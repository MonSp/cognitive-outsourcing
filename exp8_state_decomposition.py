#!/usr/bin/env python3
"""
EXP-1 (Paper 8): State Decomposition Analysis
==============================================
Static analysis of the CO+SIG EdgeKitchen codebase to quantify module
management state and token overhead.

Three parts:
  Part 1 — Static Code Audit: enumerate management functions, classify
           externalizability (F_ext / F_partial / F_sem)
  Part 2 — Dynamic Token Counting: use MeaningCompiler to measure actual
           token counts for a 35-step Kitchen scenario
  Part 3 — Classification Report: generate JSON files + console tables

Outputs:
  data/exp8_state_audit/function_taxonomy.json
  data/exp8_state_audit/token_overhead_summary.json

Usage:
  python exp8_state_decomposition.py
  python exp8_state_decomposition.py --steps 50
"""

import json
import os
import sys
import time
import argparse
from pathlib import Path
from typing import List, Dict

SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR / "data" / "exp8_state_audit"
MODEL_PATH = SCRIPT_DIR / "models" / "Qwen3.5-0.8B-Q4_K_M.gguf"

sys.path.insert(0, str(SCRIPT_DIR))
from edge_agent_bench import (
    KITCHEN_SYSTEM_PROMPT,
    KitchenToolRegistry,
    build_kitchen_scenario,
)
from core import MeaningCompiler


# ============================================================
# Part 1: Static Code Audit — Module Management Function Taxonomy
# ============================================================
#
# Each management function represents a distinct cognitive or
# infrastructural operation the agent must perform. Classification:
#   F_ext    = Fully externalizable to a harness (structured state)
#   F_partial = Partially externalizable (mixed semantic + structural)
#   F_sem    = Requires semantic judgment (irreducible model load)

MANAGEMENT_FUNCTIONS = [
    {
        "id": "M01",
        "name": "System Context Initialization",
        "name_zh": "系统上下文初始化",
        "description": (
            "One-time tokenization and KV-cache injection of the system prompt "
            "that defines the agent's role and behavioral constraints. Uses "
            "MeaningCompiler.tokenize + eval + InjectionEngine.update_cache."
        ),
        "classification": "F_ext",
        "phase": "init",
        "frequency": "once",
        "token_overhead": {
            "type": "fixed",
            "measured_tokens": None,
            "note": "Injected once at session start; persists in KV cache",
        },
        "harness_externalizable": True,
        "externalization_mechanism": (
            "Harness pre-fills system prompt into KV cache before first turn. "
            "Model never needs to manage its own role definition."
        ),
        "code_refs": [
            "edge_agent_bench.py:510-515 (EdgeKitchenSIG.run)",
            "core/injection.py:211-214 (update_cache)",
        ],
        "state_managed": "agent_role",
    },
    {
        "id": "M02",
        "name": "User Query Injection",
        "name_zh": "用户查询注入",
        "description": (
            "Per-step tokenization and KV injection of the user's natural "
            "language query. Format: '\\nUser: {query}\\nAssistant:'. "
            "The model must parse intent from this text."
        ),
        "classification": "F_ext",
        "phase": "per_step",
        "frequency": "every_step",
        "token_overhead": {
            "type": "variable",
            "measured_tokens": None,
            "note": "10-30 tokens per step depending on query complexity",
        },
        "harness_externalizable": True,
        "externalization_mechanism": (
            "Harness injects pre-formatted query into KV cache. Could also "
            "pre-parse intent into structured representation."
        ),
        "code_refs": [
            "edge_agent_bench.py:523-527 (EdgeKitchenSIG.run)",
        ],
        "state_managed": "user_intent",
    },
    {
        "id": "M03",
        "name": "Tool Dispatch & Execution",
        "name_zh": "工具调度与执行",
        "description": (
            "Harness executes the tool via KitchenToolRegistry.execute() and "
            "returns the result text. In the benchmark, tool selection is "
            "pre-scripted from scenario; in production the model generates "
            "a tool call that the harness parses."
        ),
        "classification": "F_ext",
        "phase": "per_step",
        "frequency": "every_step",
        "token_overhead": {
            "type": "zero",
            "measured_tokens": 0,
            "note": "Execution happens in harness; no tokens consumed directly",
        },
        "harness_externalizable": True,
        "externalization_mechanism": (
            "Already fully externalized. Harness executes tools and returns "
            "structured results. Model only sees the formatted output."
        ),
        "code_refs": [
            "edge_agent_bench.py:529 (EdgeKitchenSIG.run)",
            "edge_agent_bench.py:146-153 (KitchenToolRegistry.execute)",
        ],
        "state_managed": "tool_io",
    },
    {
        "id": "M04",
        "name": "Tool Result Integration",
        "name_zh": "工具结果整合",
        "description": (
            "Tokenization and KV injection of tool execution results. "
            "Format: '\\n[Tool: {name}] {result}\\n'. This is the primary "
            "mechanism by which structured state enters the model's context. "
            "Results range from short status messages to full recipe listings."
        ),
        "classification": "F_ext",
        "phase": "per_step",
        "frequency": "every_step",
        "token_overhead": {
            "type": "variable",
            "measured_tokens": None,
            "note": "15-200 tokens per step; this is the largest variable cost",
        },
        "harness_externalizable": True,
        "externalization_mechanism": (
            "Harness formats results. Could compress to structured format "
            "(JSON/dict) instead of verbose natural language to reduce tokens."
        ),
        "code_refs": [
            "edge_agent_bench.py:530-534 (EdgeKitchenSIG.run)",
        ],
        "state_managed": "all_state_components",
    },
    {
        "id": "M05",
        "name": "Response Generation",
        "name_zh": "响应生成",
        "description": (
            "Model generates a natural language response using "
            "generate_until_str with repetition detection (rep_threshold=3). "
            "This is the core semantic function requiring language "
            "understanding and generation capabilities."
        ),
        "classification": "F_sem",
        "phase": "per_step",
        "frequency": "every_step",
        "token_overhead": {
            "type": "generated",
            "measured_tokens": None,
            "note": "20-80 tokens generated per step; adds to cumulative context",
        },
        "harness_externalizable": False,
        "externalization_mechanism": (
            "Cannot be externalized — requires the model's language generation "
            "capability. This is the irreducible cognitive load."
        ),
        "code_refs": [
            "edge_agent_bench.py:536-541 (EdgeKitchenSIG.run)",
            "core/compiler.py:143-161 (generate_until_str)",
        ],
        "state_managed": "response_content",
    },
    {
        "id": "M06",
        "name": "Generation State Tracking",
        "name_zh": "生成状态追踪",
        "description": (
            "After generation, produced token IDs are appended to "
            "InjectionEngine.cached_ids. Pure bookkeeping — tracking what "
            "is in the KV cache for future eviction or compaction."
        ),
        "classification": "F_ext",
        "phase": "per_step",
        "frequency": "every_step",
        "token_overhead": {
            "type": "zero",
            "measured_tokens": 0,
            "note": "Metadata tracking only; no additional context tokens",
        },
        "harness_externalizable": True,
        "externalization_mechanism": (
            "Already externalized in InjectionEngine. Pure data-structure "
            "management with no semantic content."
        ),
        "code_refs": [
            "edge_agent_bench.py:541 (EdgeKitchenSIG.run)",
            "core/injection.py:211-214 (update_cache)",
        ],
        "state_managed": "cache_metadata",
    },
    {
        "id": "M07",
        "name": "User Profile State Maintenance",
        "name_zh": "用户档案状态维护",
        "description": (
            "User dietary profile (allergies, diet, servings, cuisine "
            "preference) established via set_user_profile. Persists as "
            "natural language text in KV cache. The model must attend to "
            "this when reasoning about recipes and ingredients."
        ),
        "classification": "F_ext",
        "phase": "state",
        "frequency": "once_written_many_reads",
        "token_overhead": {
            "type": "fixed_write_variable_read",
            "measured_tokens": None,
            "note": "~20 tokens to write; ongoing attention cost per step",
        },
        "harness_externalizable": True,
        "externalization_mechanism": (
            "Store as structured dict. Inject summary into tool context "
            "on demand instead of relying on attention over historical text."
        ),
        "code_refs": [
            "edge_agent_bench.py:155-158 (_tool_set_user_profile)",
        ],
        "state_managed": "user_profile",
    },
    {
        "id": "M08",
        "name": "Inventory State Maintenance",
        "name_zh": "库存状态维护",
        "description": (
            "Pantry and fridge contents tracked via add_to_pantry, "
            "add_to_fridge, check_pantry, check_fridge, check_ingredients. "
            "Model must maintain awareness of available ingredients across "
            "many steps with growing item lists."
        ),
        "classification": "F_ext",
        "phase": "state",
        "frequency": "many_writes_many_reads",
        "token_overhead": {
            "type": "cumulative",
            "measured_tokens": None,
            "note": "10-50 tokens per inventory operation; grows with item count",
        },
        "harness_externalizable": True,
        "externalization_mechanism": (
            "Store as Dict[str, float] in harness. Query on demand. "
            "Reduces context from O(n_items) text to O(1) structured query."
        ),
        "code_refs": [
            "edge_agent_bench.py:160-165 (_tool_check_pantry)",
            "edge_agent_bench.py:167-172 (_tool_check_fridge)",
            "edge_agent_bench.py:174-176 (_tool_add_to_pantry)",
            "edge_agent_bench.py:178-180 (_tool_add_to_fridge)",
            "edge_agent_bench.py:223-232 (_tool_check_ingredients)",
        ],
        "state_managed": "inventory",
    },
    {
        "id": "M09",
        "name": "Cooking Progress Tracking",
        "name_zh": "烹饪进度追踪",
        "description": (
            "Current recipe and cooking step tracked via start_cooking and "
            "next_step. Model must remember which recipe is being cooked "
            "and the current step number across the cooking sequence."
        ),
        "classification": "F_ext",
        "phase": "state",
        "frequency": "once_written_per_recipe",
        "token_overhead": {
            "type": "variable",
            "measured_tokens": None,
            "note": "~20-40 tokens per cooking step result",
        },
        "harness_externalizable": True,
        "externalization_mechanism": (
            "Store as (recipe_id, step_index) tuple. Harness tracks progress. "
            "Model only needs the current step instruction text."
        ),
        "code_refs": [
            "edge_agent_bench.py:248-255 (_tool_start_cooking)",
            "edge_agent_bench.py:257-267 (_tool_next_step)",
        ],
        "state_managed": "cooking_progress",
    },
    {
        "id": "M10",
        "name": "Oven & Timer State Management",
        "name_zh": "烤箱与定时器状态管理",
        "description": (
            "Oven temperature and on/off state, plus timer labels. "
            "Tracked via set_oven, get_oven_status, set_timer."
        ),
        "classification": "F_ext",
        "phase": "state",
        "frequency": "occasional",
        "token_overhead": {
            "type": "variable",
            "measured_tokens": None,
            "note": "~10-30 tokens per query",
        },
        "harness_externalizable": True,
        "externalization_mechanism": (
            "Store as structured dict: {oven_temp, oven_on, timers[]}. "
            "Harness answers get_oven_status without model involvement."
        ),
        "code_refs": [
            "edge_agent_bench.py:234-237 (_tool_set_oven)",
            "edge_agent_bench.py:239-242 (_tool_get_oven_status)",
            "edge_agent_bench.py:244-246 (_tool_set_timer)",
        ],
        "state_managed": "oven_timer",
    },
    {
        "id": "M11",
        "name": "Shopping List Management",
        "name_zh": "购物清单管理",
        "description": (
            "Shopping list items with quantities and price estimates. "
            "Managed via add_shopping_item, get_shopping_list, compare_prices."
        ),
        "classification": "F_ext",
        "phase": "state",
        "frequency": "occasional",
        "token_overhead": {
            "type": "cumulative",
            "measured_tokens": None,
            "note": "15-60 tokens per query; grows with list length",
        },
        "harness_externalizable": True,
        "externalization_mechanism": (
            "Store as Dict[str, int] with price lookup table. "
            "Harness returns formatted list on demand."
        ),
        "code_refs": [
            "edge_agent_bench.py:275-278 (_tool_add_shopping_item)",
            "edge_agent_bench.py:280-290 (_tool_get_shopping_list)",
            "edge_agent_bench.py:292-300 (_tool_compare_prices)",
        ],
        "state_managed": "shopping_list",
    },
    {
        "id": "M12",
        "name": "Recipe Knowledge Retrieval",
        "name_zh": "食谱知识检索",
        "description": (
            "Recipe details (ingredients, steps, nutrition, allergens) "
            "retrieved via get_recipe, find_recipes, get_nutrition. "
            "These are database lookups producing verbose text output."
        ),
        "classification": "F_ext",
        "phase": "state",
        "frequency": "frequent",
        "token_overhead": {
            "type": "variable",
            "measured_tokens": None,
            "note": "50-200 tokens per recipe retrieval",
        },
        "harness_externalizable": True,
        "externalization_mechanism": (
            "Recipes are already structured dicts in RECIPES. "
            "Harness can return targeted fields instead of full text."
        ),
        "code_refs": [
            "edge_agent_bench.py:182-193 (_tool_get_recipe)",
            "edge_agent_bench.py:195-221 (_tool_find_recipes)",
            "edge_agent_bench.py:302-307 (_tool_get_nutrition)",
        ],
        "state_managed": "recipe_knowledge",
    },
    {
        "id": "M13",
        "name": "Substitution & Price Knowledge",
        "name_zh": "替代品与价格知识",
        "description": (
            "Ingredient substitution mappings and price database. "
            "Retrieved via get_substitution, compare_prices."
        ),
        "classification": "F_ext",
        "phase": "state",
        "frequency": "occasional",
        "token_overhead": {
            "type": "variable",
            "measured_tokens": None,
            "note": "~10-40 tokens per query",
        },
        "harness_externalizable": True,
        "externalization_mechanism": (
            "Already structured as SUBSTITUTIONS and PRICES dicts. "
            "Harness can answer directly without model."
        ),
        "code_refs": [
            "edge_agent_bench.py:269-273 (_tool_get_substitution)",
            "edge_agent_bench.py:292-300 (_tool_compare_prices)",
            "edge_agent_bench.py:111-135 (PRICES, SUBSTITUTIONS)",
        ],
        "state_managed": "substitution_price",
    },
    {
        "id": "M14",
        "name": "Context Window Accumulation",
        "name_zh": "上下文窗口累积",
        "description": (
            "KV cache grows linearly with each step as user queries, "
            "tool results, and responses are appended. This is the "
            "fundamental source of token overhead in any context-based "
            "agent. Each cached token consumes memory and attention FLOPs."
        ),
        "classification": "F_ext",
        "phase": "infrastructure",
        "frequency": "continuous",
        "token_overhead": {
            "type": "cumulative",
            "measured_tokens": None,
            "note": "Grows ~50-150 tokens per step; total depends on scenario length",
        },
        "harness_externalizable": True,
        "externalization_mechanism": (
            "SIG incremental injection avoids re-encoding. "
            "InjectionEngine manages cache with eviction/compaction. "
            "This is SIG's core value proposition over AppLoop."
        ),
        "code_refs": [
            "edge_agent_bench.py:509-515 (EdgeKitchenSIG.run)",
            "core/injection.py:139-325 (InjectionEngine full class)",
        ],
        "state_managed": "kv_cache",
    },
    {
        "id": "M15",
        "name": "Recall Verification (Probe Testing)",
        "name_zh": "召回验证（探测测试）",
        "description": (
            "Periodic probes test whether the model recalls facts from "
            "earlier in the conversation (e.g., oven temp set 20 steps ago). "
            "Requires attending to distant context. Measures recall fidelity."
        ),
        "classification": "F_sem",
        "phase": "evaluation",
        "frequency": "conditional",
        "token_overhead": {
            "type": "variable",
            "measured_tokens": None,
            "note": "~20 tokens probe query + ~30 tokens generation per probe",
        },
        "harness_externalizable": False,
        "externalization_mechanism": (
            "Probe testing inherently requires the model to demonstrate recall. "
            "The test harness/framework is externalizable, but the actual "
            "recall capability is a model-intrinsic function."
        ),
        "code_refs": [
            "edge_agent_bench.py:545-556 (EdgeKitchenSIG.run)",
            "edge_agent_bench.py:468-492 (build_probe_queries)",
        ],
        "state_managed": "recall_capability",
    },
    {
        "id": "M16",
        "name": "Interruption Context Switching",
        "name_zh": "中断上下文切换",
        "description": (
            "When an interruption task type is encountered, the agent must "
            "switch context (e.g., from cooking to handling a guest dietary "
            "restriction). The model must reconcile the interruption with "
            "ongoing cooking state while maintaining both threads."
        ),
        "classification": "F_partial",
        "phase": "per_step",
        "frequency": "occasional",
        "token_overhead": {
            "type": "variable",
            "measured_tokens": None,
            "note": "Same token cost as regular steps; higher cognitive load",
        },
        "harness_externalizable": False,
        "externalization_mechanism": (
            "Harness can detect interruption events and route to different "
            "tool chains, but the model must still understand and integrate "
            "the new context with the ongoing task."
        ),
        "code_refs": [
            "edge_agent_bench.py:423-438 (interruption scenario steps)",
        ],
        "state_managed": "context_switching",
    },
    {
        "id": "M17",
        "name": "Repetition Detection",
        "name_zh": "重复检测",
        "description": (
            "The generate_until_str method includes _detect_repetition with "
            "rep_threshold=3 to prevent degenerate loops. This is a heuristic "
            "check on the generated text for repeating patterns."
        ),
        "classification": "F_partial",
        "phase": "per_step",
        "frequency": "every_step",
        "token_overhead": {
            "type": "zero",
            "measured_tokens": 0,
            "note": "Computational cost only; no token overhead",
        },
        "harness_externalizable": True,
        "externalization_mechanism": (
            "Rule-based pattern matching that the harness can perform. "
            "Does not require semantic understanding, just string analysis."
        ),
        "code_refs": [
            "core/compiler.py:105-121 (_detect_repetition)",
            "core/compiler.py:143-161 (generate_until_str, rep_threshold)",
        ],
        "state_managed": "generation_quality",
    },
]

STATE_COMPONENTS = [
    {
        "id": "S_user_profile",
        "name": "User Profile",
        "name_zh": "用户档案",
        "structured_format": '{"allergies": str, "diet": str, "servings": int, "cuisine_pref": str}',
        "natural_language_tokens_est": 20,
        "structured_tokens_est": 5,
        "savings_pct": 75.0,
        "tools": ["set_user_profile"],
    },
    {
        "id": "S_inventory",
        "name": "Pantry & Fridge Inventory",
        "name_zh": "食品柜与冰箱库存",
        "structured_format": '{"pantry": {item: amount_g}, "fridge": {item: amount_g}}',
        "natural_language_tokens_est": 40,
        "structured_tokens_est": 5,
        "savings_pct": 87.5,
        "tools": ["add_to_pantry", "add_to_fridge", "check_pantry", "check_fridge", "check_ingredients"],
    },
    {
        "id": "S_cooking_progress",
        "name": "Cooking Progress",
        "name_zh": "烹饪进度",
        "structured_format": '{"recipe_id": str, "step": int, "total_steps": int}',
        "natural_language_tokens_est": 30,
        "structured_tokens_est": 5,
        "savings_pct": 83.3,
        "tools": ["start_cooking", "next_step"],
    },
    {
        "id": "S_oven_timer",
        "name": "Oven & Timer State",
        "name_zh": "烤箱与定时器状态",
        "structured_format": '{"oven_temp": int, "oven_on": bool, "timers": [...]}',
        "natural_language_tokens_est": 20,
        "structured_tokens_est": 5,
        "savings_pct": 75.0,
        "tools": ["set_oven", "get_oven_status", "set_timer"],
    },
    {
        "id": "S_shopping",
        "name": "Shopping List",
        "name_zh": "购物清单",
        "structured_format": '{"items": {name: qty}, "total_price": float}',
        "natural_language_tokens_est": 40,
        "structured_tokens_est": 8,
        "savings_pct": 80.0,
        "tools": ["add_shopping_item", "get_shopping_list", "compare_prices"],
    },
    {
        "id": "S_recipes",
        "name": "Recipe Knowledge",
        "name_zh": "食谱知识",
        "structured_format": "Dict lookup by recipe_id; return targeted fields",
        "natural_language_tokens_est": 100,
        "structured_tokens_est": 15,
        "savings_pct": 85.0,
        "tools": ["get_recipe", "find_recipes", "get_nutrition"],
    },
    {
        "id": "S_substitution",
        "name": "Substitution & Price Knowledge",
        "name_zh": "替代品与价格知识",
        "structured_format": '{"substitute": str, "price": float}',
        "natural_language_tokens_est": 15,
        "structured_tokens_est": 3,
        "savings_pct": 80.0,
        "tools": ["get_substitution", "compare_prices"],
    },
]


# ============================================================
# Part 2: Dynamic Token Counting
# ============================================================

def measure_token_overhead(compiler: MeaningCompiler, total_steps: int = 35) -> Dict:
    """Simulate a Kitchen scenario and measure actual token counts.

    Does NOT run the model for generation — only uses tokenize() to count
    how many tokens each piece of context consumes.
    """
    scenario = build_kitchen_scenario(total_steps)
    scenario = scenario[:total_steps]

    results = {
        "total_steps": len(scenario),
        "system_prompt": {},
        "tool_descriptions": {},
        "extended_system_prompt": {},
        "per_step_detail": [],
        "cumulative_trajectory": [],
        "cumulative_at_checkpoints": {},
        "by_task_type": {},
        "token_budget_analysis": {},
        "apploop_vs_sig_analysis": {},
    }

    # --- Static components ---
    sys_text = f"{KITCHEN_SYSTEM_PROMPT}\n\n"
    sys_tokens = list(compiler.tokenize(sys_text, add_bos=False))
    results["system_prompt"] = {
        "text": KITCHEN_SYSTEM_PROMPT,
        "token_count": len(sys_tokens),
    }

    tool_desc_text = KitchenToolRegistry.TOOL_DESCRIPTIONS
    tool_desc_tokens = list(compiler.tokenize(tool_desc_text, add_bos=False))
    results["tool_descriptions"] = {
        "text_preview": tool_desc_text[:200] + "...",
        "num_tools": 18,
        "token_count": len(tool_desc_tokens),
    }

    extended_sys = f"{KITCHEN_SYSTEM_PROMPT}\n\n{tool_desc_text}\n\n"
    ext_sys_tokens = list(compiler.tokenize(extended_sys, add_bos=False))
    results["extended_system_prompt"] = {
        "token_count": len(ext_sys_tokens),
        "note": "System prompt + all 18 tool descriptions (hypothetical complete prompt)",
    }

    # --- Per-step simulation ---
    tools = KitchenToolRegistry()
    cumulative_tokens = len(sys_tokens)
    task_type_data = {}
    checkpoints = {5: None, 10: None, 20: None, 35: None, 50: None}
    step_user_tokens = []
    step_tool_tokens = []

    for step_i, step in enumerate(scenario):
        user_line = f"\nUser: {step.user_query}\nAssistant:"
        u_tokens = list(compiler.tokenize(user_line, add_bos=False))

        result = tools.execute(step.tool_name, step.tool_args)
        tool_line = f"\n[Tool: {step.tool_name}] {result}\n"
        t_tokens = list(compiler.tokenize(tool_line, add_bos=False))

        step_total = len(u_tokens) + len(t_tokens)
        cumulative_tokens += step_total

        step_user_tokens.append(len(u_tokens))
        step_tool_tokens.append(len(t_tokens))

        step_info = {
            "step": step_i + 1,
            "task_type": step.task_type,
            "tool_name": step.tool_name,
            "user_query_tokens": len(u_tokens),
            "tool_result_tokens": len(t_tokens),
            "step_injected_tokens": step_total,
            "cumulative_context_tokens": cumulative_tokens,
            "user_query_preview": step.user_query[:80],
            "result_preview": result[:100],
        }
        results["per_step_detail"].append(step_info)

        results["cumulative_trajectory"].append({
            "step": step_i + 1,
            "cumulative_tokens": cumulative_tokens,
            "step_tokens": step_total,
        })

        # By task type
        tt = step.task_type
        if tt not in task_type_data:
            task_type_data[tt] = {"count": 0, "total_user": 0, "total_tool": 0, "max_tool": 0}
        task_type_data[tt]["count"] += 1
        task_type_data[tt]["total_user"] += len(u_tokens)
        task_type_data[tt]["total_tool"] += len(t_tokens)
        task_type_data[tt]["max_tool"] = max(task_type_data[tt]["max_tool"], len(t_tokens))

        # Checkpoints
        step_num = step_i + 1
        if step_num in checkpoints:
            checkpoints[step_num] = cumulative_tokens

    # --- Aggregate statistics ---
    results["cumulative_at_checkpoints"] = {
        f"step_{k}": v for k, v in sorted(checkpoints.items()) if v is not None
    }

    for tt, data in task_type_data.items():
        n = data["count"]
        results["by_task_type"][tt] = {
            "step_count": n,
            "avg_user_tokens": round(data["total_user"] / n, 1),
            "avg_tool_tokens": round(data["total_tool"] / n, 1),
            "max_tool_tokens": data["max_tool"],
            "total_tokens": data["total_user"] + data["total_tool"],
        }

    # --- Token budget analysis ---
    total_user = sum(step_user_tokens)
    total_tool = sum(step_tool_tokens)
    total_injected = total_user + total_tool
    avg_user = total_user / max(len(step_user_tokens), 1)
    avg_tool = total_tool / max(len(step_tool_tokens), 1)

    results["token_budget_analysis"] = {
        "system_prompt_tokens": len(sys_tokens),
        "total_user_query_tokens": total_user,
        "total_tool_result_tokens": total_tool,
        "total_injected_tokens": total_injected,
        "final_context_size": cumulative_tokens,
        "avg_user_tokens_per_step": round(avg_user, 1),
        "avg_tool_tokens_per_step": round(avg_tool, 1),
        "tool_result_share_pct": round(total_tool / max(total_injected, 1) * 100, 1),
        "user_query_share_pct": round(total_user / max(total_injected, 1) * 100, 1),
        "management_overhead_share_pct": round(
            total_tool / max(cumulative_tokens, 1) * 100, 1
        ),
    }

    # --- AppLoop vs SIG analysis ---
    # AppLoop re-encodes the ENTIRE context each step (quadratic total compute)
    # SIG only injects NEW tokens each step (linear total compute)
    sig_total_eval = cumulative_tokens
    apploop_total_eval = 0
    apploop_trajectory = []
    ctx_so_far = len(sys_tokens)
    for si, step in enumerate(scenario):
        user_line = f"\nUser: {step.user_query}\nAssistant:"
        u_len = len(list(compiler.tokenize(user_line, add_bos=False)))
        result = tools.execute(step.tool_name, step.tool_args)
        tool_line = f"\n[Tool: {step.tool_name}] {result}\n"
        t_len = len(list(compiler.tokenize(tool_line, add_bos=False)))
        # Assume ~30 tokens generated per step for the response
        gen_est = 30
        ctx_so_far += u_len + t_len + gen_est
        apploop_total_eval += ctx_so_far
        apploop_trajectory.append({
            "step": si + 1,
            "full_context_tokens": ctx_so_far,
        })

    results["apploop_vs_sig_analysis"] = {
        "sig_total_tokens_evaluated": sig_total_eval,
        "apploop_total_tokens_evaluated": apploop_total_eval,
        "compute_ratio_apploop_over_sig": round(
            apploop_total_eval / max(sig_total_eval, 1), 2
        ),
        "note": (
            "AppLoop re-encodes full context each step (O(n^2) total). "
            "SIG injects only new tokens incrementally (O(n) total). "
            "Ratio quantifies SIG's compute advantage."
        ),
    }

    return results


# ============================================================
# Part 3: Classification Report & Console Output
# ============================================================

def compute_classification_summary(functions: List[Dict]) -> Dict:
    counts = {"F_ext": 0, "F_partial": 0, "F_sem": 0}
    externalizable = 0
    for f in functions:
        cls = f["classification"]
        counts[cls] = counts.get(cls, 0) + 1
        if f["harness_externalizable"]:
            externalizable += 1

    total = len(functions)
    return {
        "total_functions": total,
        "classification_distribution": counts,
        "externalizable_count": externalizable,
        "externalizable_pct": round(externalizable / total * 100, 1),
        "f_ext_pct": round(counts["F_ext"] / total * 100, 1),
        "f_partial_pct": round(counts["F_partial"] / total * 100, 1),
        "f_sem_pct": round(counts["F_sem"] / total * 100, 1),
    }


def estimate_savings(state_components: List[Dict], token_data: Dict) -> Dict:
    """Estimate token savings if all F_ext state were externalized."""
    per_step_tool = token_data["token_budget_analysis"]["avg_tool_tokens_per_step"]
    total_steps = token_data["total_steps"]

    total_savings = 0
    savings_by_component = []
    for sc in state_components:
        nl_est = sc["natural_language_tokens_est"]
        st_est = sc["structured_tokens_est"]
        saving = nl_est - st_est
        pct = sc["savings_pct"]
        savings_by_component.append({
            "id": sc["id"],
            "name": sc["name"],
            "nl_tokens": nl_est,
            "structured_tokens": st_est,
            "saving_per_query": saving,
            "savings_pct": pct,
        })
        total_savings += saving

    avg_saving_per_step = total_savings / max(len(state_components), 1)
    estimated_total_saving = avg_saving_per_step * total_steps

    return {
        "by_component": savings_by_component,
        "avg_saving_per_step_tokens": round(avg_saving_per_step, 1),
        "estimated_total_saving_tokens": round(estimated_total_saving),
        "current_final_context": token_data["token_budget_analysis"]["final_context_size"],
        "estimated_optimized_context": round(
            token_data["token_budget_analysis"]["final_context_size"] - estimated_total_saving
        ),
        "context_reduction_pct": round(
            estimated_total_saving
            / max(token_data["token_budget_analysis"]["final_context_size"], 1)
            * 100,
            1,
        ),
    }


def generate_reports(functions, state_components, token_data):
    os.makedirs(DATA_DIR, exist_ok=True)

    classification = compute_classification_summary(functions)
    savings = estimate_savings(state_components, token_data)

    taxonomy = {
        "experiment": "EXP-1 State Decomposition Analysis",
        "paper": "Paper 8: Cognitive Outsourcing + SIG",
        "codebase": "edge_agent_bench.py (EdgeKitchenSIG + KitchenToolRegistry)",
        "model": str(MODEL_PATH.name),
        "management_functions": functions,
        "state_components": state_components,
        "classification_summary": classification,
        "savings_estimate": savings,
    }
    taxonomy_path = DATA_DIR / "function_taxonomy.json"
    with open(taxonomy_path, "w", encoding="utf-8") as f:
        json.dump(taxonomy, f, indent=2, ensure_ascii=False)

    token_summary = {
        "experiment": "EXP-1 State Decomposition Analysis",
        "model": str(MODEL_PATH.name),
        "measurement_method": "MeaningCompiler.tokenize (llama-cpp-python, CPU)",
        "scenario_steps": token_data["total_steps"],
        "system_prompt_tokens": token_data["system_prompt"]["token_count"],
        "tool_description_tokens": token_data["tool_descriptions"]["token_count"],
        "extended_system_prompt_tokens": token_data["extended_system_prompt"]["token_count"],
        "per_step_detail": token_data["per_step_detail"],
        "cumulative_trajectory": token_data["cumulative_trajectory"],
        "cumulative_at_checkpoints": token_data["cumulative_at_checkpoints"],
        "by_task_type": token_data["by_task_type"],
        "token_budget_analysis": token_data["token_budget_analysis"],
        "apploop_vs_sig_analysis": token_data["apploop_vs_sig_analysis"],
        "savings_estimate": savings,
    }
    token_path = DATA_DIR / "token_overhead_summary.json"
    with open(token_path, "w", encoding="utf-8") as f:
        json.dump(token_summary, f, indent=2, ensure_ascii=False)

    return str(taxonomy_path), str(token_path)


def print_summary(functions, state_components, token_data):
    classification = compute_classification_summary(functions)
    savings = estimate_savings(state_components, token_data)
    budget = token_data["token_budget_analysis"]

    print()
    print("=" * 80)
    print("  EXP-1: STATE DECOMPOSITION ANALYSIS — Paper 8")
    print("=" * 80)

    # --- Management Function Taxonomy ---
    print()
    print("  PART 1: MANAGEMENT FUNCTION TAXONOMY")
    print("  " + "-" * 76)
    print(f"  {'ID':<5} {'Function':<38} {'Class':<10} {'Phase':<12} {'Externalizable'}")
    print(f"  {'-'*5} {'-'*38} {'-'*10} {'-'*12} {'-'*14}")
    for f in functions:
        ext_str = "YES" if f["harness_externalizable"] else "NO"
        print(
            f"  {f['id']:<5} {f['name']:<38} {f['classification']:<10} "
            f"{f['phase']:<12} {ext_str}"
        )

    print()
    print("  Classification Distribution:")
    print(f"    F_ext (fully externalizable):   {classification['f_ext_pct']:>5.1f}%  "
          f"({classification['classification_distribution']['F_ext']} functions)")
    print(f"    F_partial (partially):          {classification['f_partial_pct']:>5.1f}%  "
          f"({classification['classification_distribution']['F_partial']} functions)")
    print(f"    F_sem (semantic, irreducible):  {classification['f_sem_pct']:>5.1f}%  "
          f"({classification['classification_distribution']['F_sem']} functions)")
    print(f"    Total externalizable:           {classification['externalizable_pct']:>5.1f}%  "
          f"({classification['externalizable_count']}/{classification['total_functions']})")

    # --- Token Overhead ---
    print()
    print("  PART 2: TOKEN OVERHEAD MEASUREMENT")
    print("  " + "-" * 76)
    print(f"  System prompt:          {token_data['system_prompt']['token_count']:>6} tokens")
    print(f"  Tool descriptions (18): {token_data['tool_descriptions']['token_count']:>6} tokens")
    print(f"  Extended sys prompt:    {token_data['extended_system_prompt']['token_count']:>6} tokens")
    print()
    print(f"  Avg user query tokens/step:    {budget['avg_user_tokens_per_step']:>6.1f}")
    print(f"  Avg tool result tokens/step:   {budget['avg_tool_tokens_per_step']:>6.1f}")
    print(f"  Tool result share:             {budget['tool_result_share_pct']:>5.1f}%")
    print(f"  User query share:              {budget['user_query_share_pct']:>5.1f}%")

    # By task type
    print()
    print("  Token Overhead by Task Type:")
    print(f"  {'Task Type':<22} {'Steps':>5} {'Avg User':>10} {'Avg Tool':>10} {'Max Tool':>10} {'Total':>8}")
    print(f"  {'-'*22} {'-'*5} {'-'*10} {'-'*10} {'-'*10} {'-'*8}")
    for tt, data in sorted(token_data["by_task_type"].items()):
        print(
            f"  {tt:<22} {data['step_count']:>5} "
            f"{data['avg_user_tokens']:>10.1f} {data['avg_tool_tokens']:>10.1f} "
            f"{data['max_tool_tokens']:>10} {data['total_tokens']:>8}"
        )

    # Cumulative trajectory at checkpoints
    print()
    print("  Cumulative Context Trajectory:")
    print(f"  {'Checkpoint':<12} {'Tokens':>8}")
    print(f"  {'-'*12} {'-'*8}")
    for cp, toks in token_data["cumulative_at_checkpoints"].items():
        print(f"  {cp:<12} {toks:>8}")
    print(f"  {'Final':<12} {budget['final_context_size']:>8}")

    # AppLoop vs SIG
    sig_data = token_data["apploop_vs_sig_analysis"]
    print()
    print("  AppLoop vs SIG Compute Analysis:")
    print(f"    SIG total tokens evaluated:      {sig_data['sig_total_tokens_evaluated']:>10,}")
    print(f"    AppLoop total tokens evaluated:  {sig_data['apploop_total_tokens_evaluated']:>10,}")
    print(f"    Compute ratio (AppLoop/SIG):     {sig_data['compute_ratio_apploop_over_sig']:>10.2f}x")

    # --- Savings Estimate ---
    print()
    print("  PART 3: EXTERNALIZATION SAVINGS ESTIMATE")
    print("  " + "-" * 76)
    print(f"  {'Component':<32} {'NL Tok':>8} {'Struct':>8} {'Save':>8} {'%':>6}")
    print(f"  {'-'*32} {'-'*8} {'-'*8} {'-'*8} {'-'*6}")
    for sc in savings["by_component"]:
        print(
            f"  {sc['name']:<32} {sc['nl_tokens']:>8} {sc['structured_tokens']:>8} "
            f"{sc['saving_per_query']:>8} {sc['savings_pct']:>5.0f}%"
        )
    print()
    print(f"  Avg saving per step:       {savings['avg_saving_per_step_tokens']:>8.1f} tokens")
    print(f"  Est. total saving:         {savings['estimated_total_saving_tokens']:>8} tokens")
    print(f"  Current final context:     {savings['current_final_context']:>8} tokens")
    print(f"  Est. optimized context:    {savings['estimated_optimized_context']:>8} tokens")
    print(f"  Context reduction:         {savings['context_reduction_pct']:>7.1f}%")

    print()
    print("=" * 80)
    print("  KEY FINDINGS")
    print("=" * 80)
    print(f"  1. {classification['f_ext_pct']:.0f}% of management functions are fully externalizable")
    print(f"     (F_ext), meaning they could be replaced by structured harness state.")
    print(f"  2. Only {classification['f_sem_pct']:.0f}% require irreducible semantic capability")
    print(f"     (F_sem): response generation and recall verification.")
    print(f"  3. Tool results dominate per-step token overhead at "
          f"{budget['tool_result_share_pct']:.0f}% of injected tokens.")
    print(f"  4. Externalizing structured state could reduce final context by "
          f"~{savings['context_reduction_pct']:.0f}%.")
    print(f"  5. SIG's incremental injection yields "
          f"{sig_data['compute_ratio_apploop_over_sig']:.1f}x less total compute "
          f"than AppLoop full re-encoding.")
    print()


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="EXP-1: State Decomposition Analysis (Paper 8)")
    parser.add_argument("--model", type=str, default=str(MODEL_PATH),
                        help="Path to GGUF model file")
    parser.add_argument("--steps", type=int, default=35,
                        help="Number of scenario steps to simulate")
    parser.add_argument("--n-ctx", type=int, default=512,
                        help="Context window for model (tokenization only)")
    parser.add_argument("--n-threads", type=int, default=4)
    args = parser.parse_args()

    print(f"Loading model for tokenization: {args.model}")
    print(f"  n_ctx={args.n_ctx}, n_threads={args.n_threads}, n_gpu_layers=0")
    t0 = time.time()
    compiler = MeaningCompiler(
        model_path=args.model,
        n_ctx=args.n_ctx,
        n_threads=args.n_threads,
        n_gpu_layers=0,
    )
    print(f"  Model loaded in {time.time() - t0:.1f}s")

    print(f"\nSimulating {args.steps}-step Kitchen scenario (token counting only)...")
    t1 = time.time()
    token_data = measure_token_overhead(compiler, total_steps=args.steps)
    print(f"  Measurement complete in {time.time() - t1:.1f}s")

    taxonomy_path, token_path = generate_reports(
        MANAGEMENT_FUNCTIONS, STATE_COMPONENTS, token_data
    )
    print(f"\n  Reports saved:")
    print(f"    {taxonomy_path}")
    print(f"    {token_path}")

    print_summary(MANAGEMENT_FUNCTIONS, STATE_COMPONENTS, token_data)

    print("Done.")


if __name__ == "__main__":
    main()
