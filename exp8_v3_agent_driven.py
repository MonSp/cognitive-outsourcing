#!/usr/bin/env python3
"""
EXP-11/12/13/14/15: Agent-Driven Tool Selection
================================================
Addresses the fundamental design flaw in previous kitchen benchmark:
pre-scripted tool calls never tested the model's ability to SELECT tools.

Key change: The MODEL decides which tool to call based on the user query,
and we measure tool selection accuracy as the primary metric.

Agent loop:
  1. User query + tool list → model generates ACTION: tool_name(args)
  2. Parse ACTION, compare with ground truth tool
  3. Execute ground truth tool (keeps scenario deterministic)
  4. Inject tool result
  5. Model generates response

Conditions:
  SIG              — baseline, no harness state
  SIG+SECMH-full   — full SECM-H state before ACTION generation
  SIG+SECMH-selective — selective hint at decision points

Experiments:
  EXP-11: Clean kitchen — tool selection accuracy
  EXP-12: Noisy kitchen — 15% tool failures, tests SECM-H failure tracking
  EXP-13: Path A/B disentanglement — forced selection + SECM-H state
  EXP-14: Natural language state rendering — template vs NL format
  EXP-15: 0.8B agent-driven — generalization to smaller model

Usage:
  python exp8_v3_agent_driven.py --task exp11 --n-runs 3 --n-steps 35
  python exp8_v3_agent_driven.py --task exp12 --n-runs 3 --n-steps 35
  python exp8_v3_agent_driven.py --task exp13 --n-runs 3 --n-steps 35
  python exp8_v3_agent_driven.py --task exp14 --n-runs 3 --n-steps 35
  python exp8_v3_agent_driven.py --task exp15 --n-runs 3 --n-steps 35
  python exp8_v3_agent_driven.py --task all --n-runs 3 --n-steps 35
"""

import os, sys, time, json, random, re, argparse
from typing import List, Dict, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core import MeaningCompiler, InjectionEngine, init_metrics
from core.harness import SECMHarness
from core.quality import ContentQualityEvaluator
from edge_agent_bench import (
    KitchenToolRegistry, KitchenStep, build_kitchen_scenario,
)


# ======================================================================
# Tool Descriptions & System Prompt
# ======================================================================

TOOL_INFO = [
    ("set_user_profile", "name, allergies, diet, servings, cuisine_pref",
     "Set user dietary profile"),
    ("check_pantry", "", "List pantry inventory"),
    ("check_fridge", "", "List fridge inventory"),
    ("add_to_pantry", "ingredient, amount_g",
     "Add ingredient to pantry"),
    ("add_to_fridge", "ingredient, amount_g",
     "Add ingredient to fridge"),
    ("get_recipe", "recipe_id",
     "Get full recipe with ingredients and steps"),
    ("find_recipes", "cuisine, max_time, diet, exclude_allergens",
     "Search recipes by criteria"),
    ("check_ingredients", "recipe_id",
     "Check if ingredients are available"),
    ("set_oven", "temp_c, on", "Preheat or turn off oven"),
    ("get_oven_status", "", "Check oven temperature"),
    ("set_timer", "minutes, label", "Set kitchen timer"),
    ("start_cooking", "recipe_id", "Begin cooking a recipe"),
    ("next_step", "", "Advance to next cooking step"),
    ("get_substitution", "ingredient",
     "Find ingredient substitution"),
    ("add_shopping_item", "ingredient, quantity",
     "Add item to shopping list"),
    ("get_shopping_list", "", "View shopping list with prices"),
    ("compare_prices", "ingredients", "Compare ingredient prices"),
    ("get_nutrition", "recipe_id", "Get nutritional information"),
]

VALID_TOOL_NAMES = {t[0] for t in TOOL_INFO}
VALID_TOOL_NAMES.add("none")


def _format_tool_descriptions() -> str:
    lines = []
    for name, params, desc in TOOL_INFO:
        if params:
            lines.append(f"- {name}({params}) - {desc}")
        else:
            lines.append(f"- {name}() - {desc}")
    lines.append("- none - No tool needed")
    return "\n".join(lines)


TOOL_SELECTION_PROMPT = (
    "You are a kitchen assistant. Given a user request, choose the best tool to call.\n"
    "\n"
    "Available tools:\n"
    "{tool_descriptions}\n"
    "\n"
    "Respond with EXACTLY one tool call in this format:\n"
    "ACTION: tool_name(param1=value1, param2=value2)\n"
    "\n"
    "If no tool is needed, respond with:\n"
    "ACTION: none"
)


# ======================================================================
# Tool Call Parser
# ======================================================================

def parse_tool_call(text: str) -> Tuple[Optional[str], Optional[Dict]]:
    text = text.strip()

    match = re.search(r'ACTION:\s*(\w+)(?:\((.*?)\))?', text, re.DOTALL)
    if match:
        tool_name = match.group(1).strip()
        args_str = (match.group(2) or "").strip()
        if tool_name.lower() == "none":
            return "none", {}
        return _parse_tool_name_and_args(tool_name, args_str)

    for tool_name in VALID_TOOL_NAMES:
        if tool_name == "none":
            continue
        if tool_name in text.lower():
            args = {}
            pattern = rf'{tool_name}\s*\((.*?)\)'
            args_match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
            if args_match:
                _, args = _parse_tool_name_and_args(tool_name, args_match.group(1))
            return tool_name, args

    return None, None


def _parse_tool_name_and_args(tool_name: str, args_str: str) -> Tuple[str, Dict]:
    if not args_str:
        return tool_name, {}

    args = {}
    parts = []
    depth = 0
    current = ""
    for ch in args_str:
        if ch == '(':
            depth += 1
            current += ch
        elif ch == ')':
            depth -= 1
            current += ch
        elif ch == ',' and depth == 0:
            parts.append(current.strip())
            current = ""
        else:
            current += ch
    if current.strip():
        parts.append(current.strip())

    for part in parts:
        if '=' in part:
            key, val = part.split('=', 1)
            key = key.strip()
            val = val.strip().strip('"').strip("'")

            if val.lower() == 'true':
                val = True
            elif val.lower() == 'false':
                val = False
            else:
                try:
                    val = int(val)
                except ValueError:
                    try:
                        val = float(val)
                    except ValueError:
                        pass

            args[key] = val

    return tool_name, args


def _args_match(parsed: Dict, gt: Dict) -> bool:
    if not gt:
        return True
    if not parsed:
        return False
    for key, gt_val in gt.items():
        if key not in parsed:
            return False
        if str(parsed[key]).lower() != str(gt_val).lower():
            return False
    return True


# ======================================================================
# Agent: AgentDrivenSIG (baseline — no harness state)
# ======================================================================

class AgentDrivenSIG:
    def __init__(self, compiler, tools):
        self.compiler = compiler
        self.tools = tools
        self.engine = InjectionEngine(compiler)

    def run(self, scenario, max_new_action=120, max_new_response=80,
            debug=False):
        metrics = init_metrics()
        metrics["gen_texts"] = []
        metrics["tool_results"] = []
        metrics["tool_selections"] = []

        self.engine.reset()
        prompt = TOOL_SELECTION_PROMPT.format(
            tool_descriptions=_format_tool_descriptions())
        sys_ids = list(self.compiler.tokenize(f"{prompt}\n\n", add_bos=False))
        pf_t0 = time.time()
        self.compiler.eval(sys_ids)
        metrics["total_prefill_time"] += time.time() - pf_t0
        metrics["total_prefill_tokens"] += len(sys_ids)
        self.engine.update_cache(sys_ids)

        wc_start = time.time()

        for step_i, step in enumerate(scenario):
            step_t0 = time.time()

            user_line = f"\nUser: {step.user_query}\nAssistant:"
            u_ids = list(self.compiler.tokenize(user_line, add_bos=False))
            self.compiler.eval(u_ids)
            metrics["total_prefill_tokens"] += len(u_ids)
            self.engine.update_cache(u_ids)

            gen_t0 = time.time()
            action_text, action_ids = self.compiler.generate_until_str(
                "\nUser:", max_new=max_new_action, rep_threshold=3)
            metrics["total_gen_time"] += time.time() - gen_t0
            metrics["total_gen_tokens"] += len(action_ids)
            self.engine.update_cache(list(action_ids))

            parsed_tool, parsed_args = parse_tool_call(action_text)
            correct_tool = (parsed_tool == step.tool_name)
            correct_args = (_args_match(parsed_args or {}, step.tool_args)
                            if correct_tool else False)

            metrics["tool_selections"].append({
                "step_id": step.step_id,
                "parsed_tool": parsed_tool,
                "gt_tool": step.tool_name,
                "correct_tool": correct_tool,
                "parsed_args": parsed_args or {},
                "gt_args": step.tool_args,
                "correct_args": correct_args,
                "action_text": action_text[:200],
            })

            result = self.tools.execute(step.tool_name, step.tool_args)
            metrics["tool_results"].append(result)

            tool_line = f"\n[Tool: {step.tool_name}] {result}\n"
            t_ids = list(self.compiler.tokenize(tool_line, add_bos=False))
            self.compiler.eval(t_ids)
            metrics["total_prefill_tokens"] += len(t_ids)
            self.engine.update_cache(t_ids)

            gen_t0 = time.time()
            gen_text, gen_ids = self.compiler.generate_until_str(
                "\nUser:", max_new=max_new_response, rep_threshold=3)
            metrics["total_gen_time"] += time.time() - gen_t0
            metrics["total_gen_tokens"] += len(gen_ids)
            metrics["gen_texts"].append(gen_text)
            self.engine.update_cache(list(gen_ids))

            metrics["per_turn_ttf"].append(time.time() - step_t0)

            if debug and step_i % 10 == 0:
                acc = sum(1 for s in metrics["tool_selections"]
                          if s["correct_tool"])
                total = len(metrics["tool_selections"])
                print(f"  SIG step {step_i + 1}/{len(scenario)} — "
                      f"tool_acc: {acc}/{total} "
                      f"({acc / max(total, 1) * 100:.0f}%), "
                      f"cache: {self.engine.cache_size} tok")

        metrics["total_ttf"] = time.time() - wc_start
        metrics["completion_count"] = len(scenario)
        metrics["cache_size"] = self.engine.cache_size
        return metrics


# ======================================================================
# Agent: AgentDrivenSECMH (full SECM-H state injection)
# ======================================================================

class AgentDrivenSECMH:
    def __init__(self, compiler, tools, harness: SECMHarness):
        self.compiler = compiler
        self.tools = tools
        self.engine = InjectionEngine(compiler)
        self.harness = harness

    def run(self, scenario, max_new_action=120, max_new_response=80,
            debug=False):
        metrics = init_metrics()
        metrics["gen_texts"] = []
        metrics["tool_results"] = []
        metrics["tool_selections"] = []
        metrics["harness_overhead_ms"] = []
        metrics["rendered_state_tokens"] = []

        self.engine.reset()
        prompt = TOOL_SELECTION_PROMPT.format(
            tool_descriptions=_format_tool_descriptions())
        sys_ids = list(self.compiler.tokenize(f"{prompt}\n\n", add_bos=False))
        pf_t0 = time.time()
        self.compiler.eval(sys_ids)
        metrics["total_prefill_time"] += time.time() - pf_t0
        metrics["total_prefill_tokens"] += len(sys_ids)
        self.engine.update_cache(sys_ids)

        wc_start = time.time()

        for step_i, step in enumerate(scenario):
            step_t0 = time.time()

            harness_t0 = time.time()
            self.harness.pre_invoke(step.tool_name, step.tool_args)
            state_text = self.harness.render_state()
            harness_ms = (time.time() - harness_t0) * 1000
            metrics["harness_overhead_ms"].append(harness_ms)

            if state_text:
                state_line = f"\n[System: {state_text}]"
                metrics["rendered_state_tokens"].append(
                    len(list(self.compiler.tokenize(state_line, add_bos=False))))
                user_line = (f"\nUser: {step.user_query}{state_line}"
                             f"\nAssistant: ACTION:")
            else:
                metrics["rendered_state_tokens"].append(0)
                user_line = f"\nUser: {step.user_query}\nAssistant: ACTION:"

            u_ids = list(self.compiler.tokenize(user_line, add_bos=False))
            self.compiler.eval(u_ids)
            metrics["total_prefill_tokens"] += len(u_ids)
            self.engine.update_cache(u_ids)

            gen_t0 = time.time()
            action_text, action_ids = self.compiler.generate_until_str(
                "\nUser:", max_new=max_new_action, rep_threshold=3)
            metrics["total_gen_time"] += time.time() - gen_t0
            metrics["total_gen_tokens"] += len(action_ids)
            self.engine.update_cache(list(action_ids))

            parsed_tool, parsed_args = parse_tool_call(action_text)
            correct_tool = (parsed_tool == step.tool_name)
            correct_args = (_args_match(parsed_args or {}, step.tool_args)
                            if correct_tool else False)

            metrics["tool_selections"].append({
                "step_id": step.step_id,
                "parsed_tool": parsed_tool,
                "gt_tool": step.tool_name,
                "correct_tool": correct_tool,
                "parsed_args": parsed_args or {},
                "gt_args": step.tool_args,
                "correct_args": correct_args,
                "action_text": action_text[:200],
            })

            result = self.tools.execute(step.tool_name, step.tool_args)
            metrics["tool_results"].append(result)

            tool_line = f"\n[Tool: {step.tool_name}] {result}\n"
            t_ids = list(self.compiler.tokenize(tool_line, add_bos=False))
            self.compiler.eval(t_ids)
            metrics["total_prefill_tokens"] += len(t_ids)
            self.engine.update_cache(t_ids)

            success = not result.startswith("[Error]")
            quality_est = min(1.0, len(result) / 200.0) if success else 0.0
            h2_t0 = time.time()
            self.harness.post_invoke(
                step.tool_name, step.tool_args, result,
                latency_ms=harness_ms, success=success, quality=quality_est)
            harness_ms += (time.time() - h2_t0) * 1000

            gen_t0 = time.time()
            gen_text, gen_ids = self.compiler.generate_until_str(
                "\nUser:", max_new=max_new_response, rep_threshold=3)
            metrics["total_gen_time"] += time.time() - gen_t0
            metrics["total_gen_tokens"] += len(gen_ids)
            metrics["gen_texts"].append(gen_text)
            self.engine.update_cache(list(gen_ids))

            metrics["per_turn_ttf"].append(time.time() - step_t0)

            if debug and step_i % 10 == 0:
                acc = sum(1 for s in metrics["tool_selections"]
                          if s["correct_tool"])
                total = len(metrics["tool_selections"])
                print(f"  SECMH step {step_i + 1}/{len(scenario)} — "
                      f"tool_acc: {acc}/{total} "
                      f"({acc / max(total, 1) * 100:.0f}%), "
                      f"cache: {self.engine.cache_size} tok")

        metrics["total_ttf"] = time.time() - wc_start
        metrics["completion_count"] = len(scenario)
        metrics["cache_size"] = self.engine.cache_size
        oh = metrics["harness_overhead_ms"]
        metrics["avg_harness_overhead_ms"] = sum(oh) / len(oh) if oh else 0
        return metrics


# ======================================================================
# Agent: AgentDrivenSelective (selective injection at decision points)
# ======================================================================

class AgentDrivenSelective:
    def __init__(self, compiler, tools, harness: SECMHarness):
        self.compiler = compiler
        self.tools = tools
        self.engine = InjectionEngine(compiler)
        self.harness = harness

    def run(self, scenario, max_new_action=120, max_new_response=80,
            debug=False):
        metrics = init_metrics()
        metrics["gen_texts"] = []
        metrics["tool_results"] = []
        metrics["tool_selections"] = []
        metrics["harness_overhead_ms"] = []
        metrics["rendered_state_tokens"] = []
        metrics["injection_count"] = 0

        self.engine.reset()
        prompt = TOOL_SELECTION_PROMPT.format(
            tool_descriptions=_format_tool_descriptions())
        sys_ids = list(self.compiler.tokenize(f"{prompt}\n\n", add_bos=False))
        pf_t0 = time.time()
        self.compiler.eval(sys_ids)
        metrics["total_prefill_time"] += time.time() - pf_t0
        metrics["total_prefill_tokens"] += len(sys_ids)
        self.engine.update_cache(sys_ids)

        prev_tool = None
        wc_start = time.time()

        for step_i, step in enumerate(scenario):
            step_t0 = time.time()

            harness_t0 = time.time()
            self.harness.pre_invoke(step.tool_name, step.tool_args)

            is_decision_point = (prev_tool is not None
                                 and step.tool_name != prev_tool)
            hint_text = ""
            if is_decision_point:
                top = self.harness.confidence.get_top(1)
                if top:
                    top_name, top_conf = top[0]
                    hint_text = (f"[HINT] Confidence: "
                                 f"{top_name}({top_conf:.2f})")

            harness_ms = (time.time() - harness_t0) * 1000
            metrics["harness_overhead_ms"].append(harness_ms)

            if hint_text:
                hint_line = f"\n[System: {hint_text}]"
                metrics["rendered_state_tokens"].append(
                    len(list(self.compiler.tokenize(hint_line, add_bos=False))))
                user_line = (f"\nUser: {step.user_query}{hint_line}"
                             f"\nAssistant: ACTION:")
                metrics["injection_count"] += 1
            else:
                metrics["rendered_state_tokens"].append(0)
                user_line = f"\nUser: {step.user_query}\nAssistant: ACTION:"

            u_ids = list(self.compiler.tokenize(user_line, add_bos=False))
            self.compiler.eval(u_ids)
            metrics["total_prefill_tokens"] += len(u_ids)
            self.engine.update_cache(u_ids)

            gen_t0 = time.time()
            action_text, action_ids = self.compiler.generate_until_str(
                "\nUser:", max_new=max_new_action, rep_threshold=3)
            metrics["total_gen_time"] += time.time() - gen_t0
            metrics["total_gen_tokens"] += len(action_ids)
            self.engine.update_cache(list(action_ids))

            parsed_tool, parsed_args = parse_tool_call(action_text)
            correct_tool = (parsed_tool == step.tool_name)
            correct_args = (_args_match(parsed_args or {}, step.tool_args)
                            if correct_tool else False)

            metrics["tool_selections"].append({
                "step_id": step.step_id,
                "parsed_tool": parsed_tool,
                "gt_tool": step.tool_name,
                "correct_tool": correct_tool,
                "parsed_args": parsed_args or {},
                "gt_args": step.tool_args,
                "correct_args": correct_args,
                "action_text": action_text[:200],
            })

            result = self.tools.execute(step.tool_name, step.tool_args)
            metrics["tool_results"].append(result)

            tool_line = f"\n[Tool: {step.tool_name}] {result}\n"
            t_ids = list(self.compiler.tokenize(tool_line, add_bos=False))
            self.compiler.eval(t_ids)
            metrics["total_prefill_tokens"] += len(t_ids)
            self.engine.update_cache(t_ids)

            success = not result.startswith("[Error]")
            quality_est = min(1.0, len(result) / 200.0) if success else 0.0
            h2_t0 = time.time()
            self.harness.post_invoke(
                step.tool_name, step.tool_args, result,
                latency_ms=harness_ms, success=success, quality=quality_est)
            harness_ms += (time.time() - h2_t0) * 1000

            gen_t0 = time.time()
            gen_text, gen_ids = self.compiler.generate_until_str(
                "\nUser:", max_new=max_new_response, rep_threshold=3)
            metrics["total_gen_time"] += time.time() - gen_t0
            metrics["total_gen_tokens"] += len(gen_ids)
            metrics["gen_texts"].append(gen_text)
            self.engine.update_cache(list(gen_ids))

            prev_tool = step.tool_name
            metrics["per_turn_ttf"].append(time.time() - step_t0)

            if debug and step_i % 10 == 0:
                acc = sum(1 for s in metrics["tool_selections"]
                          if s["correct_tool"])
                total = len(metrics["tool_selections"])
                print(f"  Selective step {step_i + 1}/{len(scenario)} — "
                      f"tool_acc: {acc}/{total} "
                      f"({acc / max(total, 1) * 100:.0f}%), "
                      f"inj: {metrics['injection_count']}")

        metrics["total_ttf"] = time.time() - wc_start
        metrics["completion_count"] = len(scenario)
        metrics["cache_size"] = self.engine.cache_size
        oh = metrics["harness_overhead_ms"]
        metrics["avg_harness_overhead_ms"] = sum(oh) / len(oh) if oh else 0
        return metrics


# ======================================================================
# NL State Renderer
# ======================================================================

def _render_state_natural_language(state_text: str) -> str:
    """Convert template SECM-H state to natural language."""
    lines = []
    m = re.search(
        r'Modules:\s*(\d+)\s*available,\s*(\d+)\s*invoked,\s*(\d+)\s*pending',
        state_text)
    if m:
        avail, invoked, pending = m.groups()
        lines.append(
            f"You have {avail} tools available, {invoked} already used.")
    m = re.search(r'Budget:\s*(\d+)/(\d+)', state_text)
    if m:
        used, total = m.groups()
        lines.append(f"Budget: {used} of {total} tokens used.")
    m = re.search(r'Top:\s*(\w+)\(([0-9.]+)\)', state_text)
    if m:
        tool, score = m.groups()
        lines.append(f"Most reliable tool: {tool} (confidence {score}).")
    m = re.search(r'Patterns:\s*(\d+)', state_text)
    if m:
        lines.append(f"{m.group(1)} cognitive patterns cached.")
    return " ".join(lines) if lines else state_text


# ======================================================================
# Agent: AgentForcedSelectionSIG (forced tool selection, no harness)
# ======================================================================

class AgentForcedSelectionSIG:
    def __init__(self, compiler, tools):
        self.compiler = compiler
        self.tools = tools
        self.engine = InjectionEngine(compiler)

    def run(self, scenario, max_new_action=120, max_new_response=80,
            debug=False):
        metrics = init_metrics()
        metrics["gen_texts"] = []
        metrics["tool_results"] = []
        metrics["tool_selections"] = []

        self.engine.reset()
        prompt = TOOL_SELECTION_PROMPT.format(
            tool_descriptions=_format_tool_descriptions())
        sys_ids = list(self.compiler.tokenize(f"{prompt}\n\n", add_bos=False))
        pf_t0 = time.time()
        self.compiler.eval(sys_ids)
        metrics["total_prefill_time"] += time.time() - pf_t0
        metrics["total_prefill_tokens"] += len(sys_ids)
        self.engine.update_cache(sys_ids)

        wc_start = time.time()

        for step_i, step in enumerate(scenario):
            step_t0 = time.time()

            user_line = (f"\nUser: {step.user_query} "
                         f"Use the {step.tool_name} tool.\nAssistant:")
            u_ids = list(self.compiler.tokenize(user_line, add_bos=False))
            self.compiler.eval(u_ids)
            metrics["total_prefill_tokens"] += len(u_ids)
            self.engine.update_cache(u_ids)

            gen_t0 = time.time()
            action_text, action_ids = self.compiler.generate_until_str(
                "\nUser:", max_new=max_new_action, rep_threshold=3)
            metrics["total_gen_time"] += time.time() - gen_t0
            metrics["total_gen_tokens"] += len(action_ids)
            self.engine.update_cache(list(action_ids))

            parsed_tool, parsed_args = parse_tool_call(action_text)
            correct_tool = (parsed_tool == step.tool_name)
            correct_args = (_args_match(parsed_args or {}, step.tool_args)
                            if correct_tool else False)

            metrics["tool_selections"].append({
                "step_id": step.step_id,
                "parsed_tool": parsed_tool,
                "gt_tool": step.tool_name,
                "correct_tool": correct_tool,
                "parsed_args": parsed_args or {},
                "gt_args": step.tool_args,
                "correct_args": correct_args,
                "action_text": action_text[:200],
            })

            result = self.tools.execute(step.tool_name, step.tool_args)
            metrics["tool_results"].append(result)

            tool_line = f"\n[Tool: {step.tool_name}] {result}\n"
            t_ids = list(self.compiler.tokenize(tool_line, add_bos=False))
            self.compiler.eval(t_ids)
            metrics["total_prefill_tokens"] += len(t_ids)
            self.engine.update_cache(t_ids)

            gen_t0 = time.time()
            gen_text, gen_ids = self.compiler.generate_until_str(
                "\nUser:", max_new=max_new_response, rep_threshold=3)
            metrics["total_gen_time"] += time.time() - gen_t0
            metrics["total_gen_tokens"] += len(gen_ids)
            metrics["gen_texts"].append(gen_text)
            self.engine.update_cache(list(gen_ids))

            metrics["per_turn_ttf"].append(time.time() - step_t0)

            if debug and step_i % 10 == 0:
                acc = sum(1 for s in metrics["tool_selections"]
                          if s["correct_tool"])
                total = len(metrics["tool_selections"])
                print(f"  ForcedSIG step {step_i + 1}/{len(scenario)} — "
                      f"tool_acc: {acc}/{total} "
                      f"({acc / max(total, 1) * 100:.0f}%), "
                      f"cache: {self.engine.cache_size} tok")

        metrics["total_ttf"] = time.time() - wc_start
        metrics["completion_count"] = len(scenario)
        metrics["cache_size"] = self.engine.cache_size
        return metrics


# ======================================================================
# Agent: AgentForcedSelectionSECMH (forced selection + SECM-H state)
# ======================================================================

class AgentForcedSelectionSECMH:
    def __init__(self, compiler, tools, harness: SECMHarness):
        self.compiler = compiler
        self.tools = tools
        self.engine = InjectionEngine(compiler)
        self.harness = harness

    def run(self, scenario, max_new_action=120, max_new_response=80,
            debug=False):
        metrics = init_metrics()
        metrics["gen_texts"] = []
        metrics["tool_results"] = []
        metrics["tool_selections"] = []
        metrics["harness_overhead_ms"] = []
        metrics["rendered_state_tokens"] = []

        self.engine.reset()
        prompt = TOOL_SELECTION_PROMPT.format(
            tool_descriptions=_format_tool_descriptions())
        sys_ids = list(self.compiler.tokenize(f"{prompt}\n\n", add_bos=False))
        pf_t0 = time.time()
        self.compiler.eval(sys_ids)
        metrics["total_prefill_time"] += time.time() - pf_t0
        metrics["total_prefill_tokens"] += len(sys_ids)
        self.engine.update_cache(sys_ids)

        wc_start = time.time()

        for step_i, step in enumerate(scenario):
            step_t0 = time.time()

            harness_t0 = time.time()
            self.harness.pre_invoke(step.tool_name, step.tool_args)
            state_text = self.harness.render_state()
            harness_ms = (time.time() - harness_t0) * 1000
            metrics["harness_overhead_ms"].append(harness_ms)

            if state_text:
                state_line = f"\n[System: {state_text}]"
                metrics["rendered_state_tokens"].append(
                    len(list(self.compiler.tokenize(
                        state_line, add_bos=False))))
                user_line = (f"\nUser: {step.user_query} "
                             f"Use the {step.tool_name} tool."
                             f"{state_line}\nAssistant: ACTION:")
            else:
                metrics["rendered_state_tokens"].append(0)
                user_line = (f"\nUser: {step.user_query} "
                             f"Use the {step.tool_name} tool."
                             f"\nAssistant: ACTION:")

            u_ids = list(self.compiler.tokenize(user_line, add_bos=False))
            self.compiler.eval(u_ids)
            metrics["total_prefill_tokens"] += len(u_ids)
            self.engine.update_cache(u_ids)

            gen_t0 = time.time()
            action_text, action_ids = self.compiler.generate_until_str(
                "\nUser:", max_new=max_new_action, rep_threshold=3)
            metrics["total_gen_time"] += time.time() - gen_t0
            metrics["total_gen_tokens"] += len(action_ids)
            self.engine.update_cache(list(action_ids))

            parsed_tool, parsed_args = parse_tool_call(action_text)
            correct_tool = (parsed_tool == step.tool_name)
            correct_args = (_args_match(parsed_args or {}, step.tool_args)
                            if correct_tool else False)

            metrics["tool_selections"].append({
                "step_id": step.step_id,
                "parsed_tool": parsed_tool,
                "gt_tool": step.tool_name,
                "correct_tool": correct_tool,
                "parsed_args": parsed_args or {},
                "gt_args": step.tool_args,
                "correct_args": correct_args,
                "action_text": action_text[:200],
            })

            result = self.tools.execute(step.tool_name, step.tool_args)
            metrics["tool_results"].append(result)

            tool_line = f"\n[Tool: {step.tool_name}] {result}\n"
            t_ids = list(self.compiler.tokenize(tool_line, add_bos=False))
            self.compiler.eval(t_ids)
            metrics["total_prefill_tokens"] += len(t_ids)
            self.engine.update_cache(t_ids)

            success = not result.startswith("[Error]")
            quality_est = min(1.0, len(result) / 200.0) if success else 0.0
            h2_t0 = time.time()
            self.harness.post_invoke(
                step.tool_name, step.tool_args, result,
                latency_ms=harness_ms, success=success, quality=quality_est)
            harness_ms += (time.time() - h2_t0) * 1000

            gen_t0 = time.time()
            gen_text, gen_ids = self.compiler.generate_until_str(
                "\nUser:", max_new=max_new_response, rep_threshold=3)
            metrics["total_gen_time"] += time.time() - gen_t0
            metrics["total_gen_tokens"] += len(gen_ids)
            metrics["gen_texts"].append(gen_text)
            self.engine.update_cache(list(gen_ids))

            metrics["per_turn_ttf"].append(time.time() - step_t0)

            if debug and step_i % 10 == 0:
                acc = sum(1 for s in metrics["tool_selections"]
                          if s["correct_tool"])
                total = len(metrics["tool_selections"])
                print(f"  ForcedSECMH step {step_i + 1}/{len(scenario)} — "
                      f"tool_acc: {acc}/{total} "
                      f"({acc / max(total, 1) * 100:.0f}%), "
                      f"cache: {self.engine.cache_size} tok")

        metrics["total_ttf"] = time.time() - wc_start
        metrics["completion_count"] = len(scenario)
        metrics["cache_size"] = self.engine.cache_size
        oh = metrics["harness_overhead_ms"]
        metrics["avg_harness_overhead_ms"] = sum(oh) / len(oh) if oh else 0
        return metrics


# ======================================================================
# Agent: AgentDrivenSECMH_NL (natural language state rendering)
# ======================================================================

class AgentDrivenSECMH_NL:
    def __init__(self, compiler, tools, harness: SECMHarness):
        self.compiler = compiler
        self.tools = tools
        self.engine = InjectionEngine(compiler)
        self.harness = harness

    def run(self, scenario, max_new_action=120, max_new_response=80,
            debug=False):
        metrics = init_metrics()
        metrics["gen_texts"] = []
        metrics["tool_results"] = []
        metrics["tool_selections"] = []
        metrics["harness_overhead_ms"] = []
        metrics["rendered_state_tokens"] = []

        self.engine.reset()
        prompt = TOOL_SELECTION_PROMPT.format(
            tool_descriptions=_format_tool_descriptions())
        sys_ids = list(self.compiler.tokenize(f"{prompt}\n\n", add_bos=False))
        pf_t0 = time.time()
        self.compiler.eval(sys_ids)
        metrics["total_prefill_time"] += time.time() - pf_t0
        metrics["total_prefill_tokens"] += len(sys_ids)
        self.engine.update_cache(sys_ids)

        wc_start = time.time()

        for step_i, step in enumerate(scenario):
            step_t0 = time.time()

            harness_t0 = time.time()
            self.harness.pre_invoke(step.tool_name, step.tool_args)
            state_text = self.harness.render_state()
            nl_text = _render_state_natural_language(state_text)
            harness_ms = (time.time() - harness_t0) * 1000
            metrics["harness_overhead_ms"].append(harness_ms)

            if nl_text:
                state_line = f"\n[System: {nl_text}]"
                metrics["rendered_state_tokens"].append(
                    len(list(self.compiler.tokenize(
                        state_line, add_bos=False))))
                user_line = (f"\nUser: {step.user_query}{state_line}"
                             f"\nAssistant: ACTION:")
            else:
                metrics["rendered_state_tokens"].append(0)
                user_line = f"\nUser: {step.user_query}\nAssistant: ACTION:"

            u_ids = list(self.compiler.tokenize(user_line, add_bos=False))
            self.compiler.eval(u_ids)
            metrics["total_prefill_tokens"] += len(u_ids)
            self.engine.update_cache(u_ids)

            gen_t0 = time.time()
            action_text, action_ids = self.compiler.generate_until_str(
                "\nUser:", max_new=max_new_action, rep_threshold=3)
            metrics["total_gen_time"] += time.time() - gen_t0
            metrics["total_gen_tokens"] += len(action_ids)
            self.engine.update_cache(list(action_ids))

            parsed_tool, parsed_args = parse_tool_call(action_text)
            correct_tool = (parsed_tool == step.tool_name)
            correct_args = (_args_match(parsed_args or {}, step.tool_args)
                            if correct_tool else False)

            metrics["tool_selections"].append({
                "step_id": step.step_id,
                "parsed_tool": parsed_tool,
                "gt_tool": step.tool_name,
                "correct_tool": correct_tool,
                "parsed_args": parsed_args or {},
                "gt_args": step.tool_args,
                "correct_args": correct_args,
                "action_text": action_text[:200],
            })

            result = self.tools.execute(step.tool_name, step.tool_args)
            metrics["tool_results"].append(result)

            tool_line = f"\n[Tool: {step.tool_name}] {result}\n"
            t_ids = list(self.compiler.tokenize(tool_line, add_bos=False))
            self.compiler.eval(t_ids)
            metrics["total_prefill_tokens"] += len(t_ids)
            self.engine.update_cache(t_ids)

            success = not result.startswith("[Error]")
            quality_est = min(1.0, len(result) / 200.0) if success else 0.0
            h2_t0 = time.time()
            self.harness.post_invoke(
                step.tool_name, step.tool_args, result,
                latency_ms=harness_ms, success=success, quality=quality_est)
            harness_ms += (time.time() - h2_t0) * 1000

            gen_t0 = time.time()
            gen_text, gen_ids = self.compiler.generate_until_str(
                "\nUser:", max_new=max_new_response, rep_threshold=3)
            metrics["total_gen_time"] += time.time() - gen_t0
            metrics["total_gen_tokens"] += len(gen_ids)
            metrics["gen_texts"].append(gen_text)
            self.engine.update_cache(list(gen_ids))

            metrics["per_turn_ttf"].append(time.time() - step_t0)

            if debug and step_i % 10 == 0:
                acc = sum(1 for s in metrics["tool_selections"]
                          if s["correct_tool"])
                total = len(metrics["tool_selections"])
                print(f"  SECMH-NL step {step_i + 1}/{len(scenario)} — "
                      f"tool_acc: {acc}/{total} "
                      f"({acc / max(total, 1) * 100:.0f}%), "
                      f"cache: {self.engine.cache_size} tok")

        metrics["total_ttf"] = time.time() - wc_start
        metrics["completion_count"] = len(scenario)
        metrics["cache_size"] = self.engine.cache_size
        oh = metrics["harness_overhead_ms"]
        metrics["avg_harness_overhead_ms"] = sum(oh) / len(oh) if oh else 0
        return metrics


# ======================================================================
# Noisy Tool Registry
# ======================================================================

class NoisyKitchenToolRegistry:
    def __init__(self, base: KitchenToolRegistry, fail_steps: set):
        self._base = base
        self._fail_steps = fail_steps
        self._step_counter = 0

    def execute(self, tool_name: str, tool_args: Dict) -> str:
        self._step_counter += 1
        if self._step_counter in self._fail_steps:
            return (f"[Error] {tool_name} failed: device timeout after 5000ms. "
                    f"Retrying may help.")
        return self._base.execute(tool_name, tool_args)

    def __getattr__(self, name):
        return getattr(self._base, name)


# ======================================================================
# Shared: create SECMHarness with kitchen patterns
# ======================================================================

def _make_harness(n_steps: int = 35) -> SECMHarness:
    scenario = build_kitchen_scenario(n_steps)
    tool_descs = {s.tool_name: s.tool_name for s in scenario}
    harness = SECMHarness(tool_descs, total_budget=2048)
    harness.seed_patterns([
        {"id": "recipe_planning",
         "steps": ["find_recipes", "get_recipe", "check_ingredients"],
         "importance": 0.8},
        {"id": "cooking_guidance",
         "steps": ["set_oven", "start_cooking", "next_step", "set_timer"],
         "importance": 0.9},
    ])
    return harness


# ======================================================================
# Summary Printers
# ======================================================================

def _print_exp11_summary(results, model_tag):
    print("\n" + "=" * 100)
    print(f"EXP-11 Summary: Agent-Driven Tool Selection ({model_tag})")
    print("=" * 100)

    header = (f"{'Condition':<24} {'ToolAcc':<10} {'ArgsAcc':<10} "
              f"{'ContentQ':<10} {'Coverage':<10} {'RespQ':<10} "
              f"{'WallClock':<11} {'GenTok':<8}")
    print(header)
    print("-" * len(header))

    for cond in ["SIG", "SIG+SECMH-full", "SIG+SECMH-selective"]:
        runs = [r for r in results if r["condition"] == cond]
        if not runs:
            continue
        n = len(runs)
        ta = sum(r["tool_selection_accuracy"] for r in runs) / n
        aa = sum(r["tool_selection_with_args_accuracy"] for r in runs) / n
        cq = sum(r["content_composite"] for r in runs) / n
        cov = sum(r["information_coverage"] for r in runs) / n
        rq = sum(r["response_quality"] for r in runs) / n
        wc = sum(r["wall_clock_s"] for r in runs) / n
        gt = sum(r["gen_tokens"] for r in runs) / n
        print(f"{cond:<24} {ta:<10.3f} {aa:<10.3f} "
              f"{cq:<10.3f} {cov:<10.3f} {rq:<10.3f} "
              f"{wc:<11.1f} {gt:<8.0f}")

    sig_runs = [r for r in results if r["condition"] == "SIG"]
    if sig_runs:
        sig_ta = (sum(r["tool_selection_accuracy"] for r in sig_runs)
                  / len(sig_runs))
        sig_cq = (sum(r["content_composite"] for r in sig_runs)
                  / len(sig_runs))
        print(f"\nΔ vs SIG (ToolAcc={sig_ta:.3f}, ContentQ={sig_cq:.3f}):")
        for cond in ["SIG+SECMH-full", "SIG+SECMH-selective"]:
            runs = [r for r in results if r["condition"] == cond]
            if not runs:
                continue
            ta = (sum(r["tool_selection_accuracy"] for r in runs)
                  / len(runs))
            cq = (sum(r["content_composite"] for r in runs)
                  / len(runs))
            wc = sum(r["wall_clock_s"] for r in runs) / len(runs)
            sig_wc = (sum(r["wall_clock_s"] for r in sig_runs)
                      / len(sig_runs))
            wc_ratio = wc / max(sig_wc, 0.001)
            print(f"  {cond:<24} ΔAcc={ta - sig_ta:+.4f}  "
                  f"ΔCQ={cq - sig_cq:+.4f}  WC={wc_ratio:.2f}x")

    print(f"\nPer-step tool selection detail (first run of each condition):")
    for cond in ["SIG", "SIG+SECMH-full", "SIG+SECMH-selective"]:
        runs = [r for r in results if r["condition"] == cond]
        if not runs:
            continue
        details = runs[0].get("tool_selection_details", [])
        mismatches = [d for d in details if not d["correct_tool"]]
        if mismatches:
            print(f"\n  {cond}: {len(mismatches)} mismatches:")
            for m in mismatches[:8]:
                print(f"    step {m['step_id']}: "
                      f"expected={m['gt_tool']}, "
                      f"got={m['parsed_tool'] or '(unparseable)'}")
                if m["action_text"]:
                    print(f"      raw: {m['action_text'][:100]}")
        else:
            print(f"\n  {cond}: all tools selected correctly!")


def _print_exp12_summary(results, model_tag):
    print("\n" + "=" * 100)
    print(f"EXP-12 Summary: Noisy Agent-Driven Tool Selection ({model_tag})")
    print("=" * 100)

    header = (f"{'Condition':<24} {'ToolAcc':<10} {'ArgsAcc':<10} "
              f"{'ContentQ':<10} {'Failures':<10} {'WallClock':<11}")
    print(header)
    print("-" * len(header))

    for cond in ["SIG", "SIG+SECMH-full", "SIG+SECMH-selective"]:
        runs = [r for r in results if r["condition"] == cond]
        if not runs:
            continue
        n = len(runs)
        ta = sum(r["tool_selection_accuracy"] for r in runs) / n
        aa = sum(r["tool_selection_with_args_accuracy"] for r in runs) / n
        cq = sum(r["content_composite"] for r in runs) / n
        fc = sum(r["actual_failures"] for r in runs) / n
        wc = sum(r["wall_clock_s"] for r in runs) / n
        print(f"{cond:<24} {ta:<10.3f} {aa:<10.3f} "
              f"{cq:<10.3f} {fc:<10.1f} {wc:<11.1f}")

    sig_runs = [r for r in results if r["condition"] == "SIG"]
    if sig_runs:
        sig_ta = (sum(r["tool_selection_accuracy"] for r in sig_runs)
                  / len(sig_runs))
        print(f"\nΔ vs SIG (ToolAcc={sig_ta:.3f}):")
        for cond in ["SIG+SECMH-full", "SIG+SECMH-selective"]:
            runs = [r for r in results if r["condition"] == cond]
            if not runs:
                continue
            ta = (sum(r["tool_selection_accuracy"] for r in runs)
                  / len(runs))
            cq = (sum(r["content_composite"] for r in runs)
                  / len(runs))
            print(f"  {cond:<24} ΔAcc={ta - sig_ta:+.4f}  "
                  f"ΔCQ={cq - sig_runs[0]['content_composite']:+.4f}")


def _print_exp13_summary(results, model_tag):
    print("\n" + "=" * 100)
    print(f"EXP-13 Summary: Path A/B Disentanglement — "
          f"Forced Selection ({model_tag})")
    print("=" * 100)

    header = (f"{'Condition':<24} {'ToolAcc':<10} {'ContentQ':<10} "
              f"{'Coverage':<10} {'GenTok':<10} {'WallClock':<11}")
    print(header)
    print("-" * len(header))

    for cond in ["ForcedSIG", "ForcedSECMH"]:
        runs = [r for r in results if r["condition"] == cond]
        if not runs:
            continue
        n = len(runs)
        ta = sum(r["tool_selection_accuracy"] for r in runs) / n
        cq = sum(r["content_composite"] for r in runs) / n
        cov = sum(r["information_coverage"] for r in runs) / n
        gt = sum(r["gen_tokens"] for r in runs) / n
        wc = sum(r["wall_clock_s"] for r in runs) / n
        print(f"{cond:<24} {ta:<10.3f} {cq:<10.3f} "
              f"{cov:<10.3f} {gt:<10.0f} {wc:<11.1f}")

    sig_runs = [r for r in results if r["condition"] == "ForcedSIG"]
    secmh_runs = [r for r in results if r["condition"] == "ForcedSECMH"]
    if sig_runs and secmh_runs:
        sig_cq = (sum(r["content_composite"] for r in sig_runs)
                  / len(sig_runs))
        secmh_cq = (sum(r["content_composite"] for r in secmh_runs)
                    / len(secmh_runs))
        sig_cov = (sum(r["information_coverage"] for r in sig_runs)
                   / len(sig_runs))
        secmh_cov = (sum(r["information_coverage"] for r in secmh_runs)
                     / len(secmh_runs))
        print(f"\nPath B isolation (forced selection, same tool):")
        print(f"  ΔContentQ  = {secmh_cq - sig_cq:+.4f}")
        print(f"  ΔCoverage  = {secmh_cov - sig_cov:+.4f}")
        if secmh_cq > sig_cq:
            print(f"  → SECM-H improves generation quality (Path B positive)")
        else:
            print(f"  → SECM-H degrades or has no effect on generation "
                  f"(Path B neutral/negative)")


def _print_exp14_summary(results, model_tag):
    print("\n" + "=" * 100)
    print(f"EXP-14 Summary: Natural Language State Rendering ({model_tag})")
    print("=" * 100)

    header = (f"{'Condition':<24} {'ToolAcc':<10} {'ContentQ':<10} "
              f"{'Coverage':<10} {'GenTok':<10} {'WallClock':<11}")
    print(header)
    print("-" * len(header))

    for cond in ["SECMH-template", "SECMH-natural"]:
        runs = [r for r in results if r["condition"] == cond]
        if not runs:
            continue
        n = len(runs)
        ta = sum(r["tool_selection_accuracy"] for r in runs) / n
        cq = sum(r["content_composite"] for r in runs) / n
        cov = sum(r["information_coverage"] for r in runs) / n
        gt = sum(r["gen_tokens"] for r in runs) / n
        wc = sum(r["wall_clock_s"] for r in runs) / n
        print(f"{cond:<24} {ta:<10.3f} {cq:<10.3f} "
              f"{cov:<10.3f} {gt:<10.0f} {wc:<11.1f}")

    tmpl_runs = [r for r in results if r["condition"] == "SECMH-template"]
    nl_runs = [r for r in results if r["condition"] == "SECMH-natural"]
    if tmpl_runs and nl_runs:
        tmpl_cq = (sum(r["content_composite"] for r in tmpl_runs)
                   / len(tmpl_runs))
        nl_cq = (sum(r["content_composite"] for r in nl_runs)
                 / len(nl_runs))
        tmpl_ta = (sum(r["tool_selection_accuracy"] for r in tmpl_runs)
                   / len(tmpl_runs))
        nl_ta = (sum(r["tool_selection_accuracy"] for r in nl_runs)
                 / len(nl_runs))
        print(f"\nFormat interference analysis:")
        print(f"  ΔContentQ  = {nl_cq - tmpl_cq:+.4f}")
        print(f"  ΔToolAcc   = {nl_ta - tmpl_ta:+.4f}")
        if nl_cq > tmpl_cq:
            print(f"  → NL rendering reduces format interference")
        else:
            print(f"  → Template format is equally or more effective")


# ======================================================================
# Shared experiment runner
# ======================================================================

def _run_condition(compiler, tools, scenario, cond_name, agent_factory,
                   run_id, data_dir, model_tag, noisy=False):
    random.seed(42)
    compiler.reset_cache()

    agent = agent_factory()
    agent_metrics = agent.run(scenario, debug=True)

    content_eval = ContentQualityEvaluator(scenario)
    content_result = content_eval.evaluate(
        agent_metrics["gen_texts"], agent_metrics["tool_results"])

    selections = agent_metrics["tool_selections"]
    correct_tools = sum(1 for s in selections if s["correct_tool"])
    tool_accuracy = correct_tools / max(len(selections), 1)
    correct_args = sum(1 for s in selections
                       if s["correct_tool"] and s["correct_args"])
    args_accuracy = correct_args / max(len(selections), 1)

    result = {
        "experiment": "exp12" if noisy else "exp11",
        "condition": cond_name,
        "model": model_tag,
        "run_id": run_id,
        "n_steps": len(scenario),
        "tool_selection_accuracy": round(tool_accuracy, 4),
        "tool_selection_with_args_accuracy": round(args_accuracy, 4),
        "content_composite": round(
            content_result.get("content_composite", 0), 4),
        "information_coverage": round(
            content_result.get("information_coverage", 0), 4),
        "response_quality": round(
            content_result.get("response_quality", 0), 4),
        "semantic_adequacy": round(
            content_result.get("semantic_adequacy", 0), 4),
        "information_density": round(
            content_result.get("information_density", 0), 4),
        "context_utilisation": round(
            content_result.get("context_utilisation", 0), 4),
        "wall_clock_s": round(agent_metrics["total_ttf"], 3),
        "gen_tokens": agent_metrics["total_gen_tokens"],
        "prefill_tokens": agent_metrics["total_prefill_tokens"],
        "completion_count": agent_metrics["completion_count"],
        "cache_size": agent_metrics.get("cache_size", 0),
        "tool_selection_details": selections,
    }

    if "avg_harness_overhead_ms" in agent_metrics:
        result["avg_harness_overhead_ms"] = round(
            agent_metrics["avg_harness_overhead_ms"], 2)

    if noisy:
        result["actual_failures"] = sum(
            1 for r in agent_metrics["tool_results"]
            if r.startswith("[Error]"))

    safe_cond = cond_name.replace('+', '_').replace('-', '_').lower()
    prefix = "exp12" if noisy else "exp11"
    fname = f"{data_dir}/{prefix}_{safe_cond}_{model_tag}_run{run_id}.json"
    with open(fname, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"  Run {run_id}: tool_acc={tool_accuracy:.3f}, "
          f"args_acc={args_accuracy:.3f}, "
          f"CQ={content_result.get('content_composite', 0):.3f}, "
          f"wc={agent_metrics['total_ttf']:.1f}s")

    return result


# ======================================================================
# EXP-11: Agent-Driven Tool Selection
# ======================================================================

def run_exp11_agent_driven(args):
    print("\n" + "=" * 70)
    print("EXP-11: Agent-Driven Tool Selection")
    print("=" * 70)

    data_dir = "data/exp8_v3"
    os.makedirs(data_dir, exist_ok=True)

    compiler = MeaningCompiler(args.model, n_ctx=16384,
                                n_gpu_layers=args.n_gpu_layers)
    tools = KitchenToolRegistry()
    scenario = build_kitchen_scenario(args.n_steps)
    model_tag = (os.path.basename(args.model)
                 .replace("-Q4_K_M.gguf", "")
                 .replace("Qwen3.5-", ""))

    print(f"  Scenario: {len(scenario)} steps")
    print(f"  Model tag: {model_tag}")

    all_results = []

    conditions = [
        ("SIG",
         lambda: AgentDrivenSIG(compiler, tools)),
        ("SIG+SECMH-full",
         lambda: AgentDrivenSECMH(
             compiler, tools, _make_harness(args.n_steps))),
        ("SIG+SECMH-selective",
         lambda: AgentDrivenSelective(
             compiler, tools, _make_harness(args.n_steps))),
    ]

    for cond_name, agent_factory in conditions:
        print(f"\n--- Condition: {cond_name} ---")
        for run_id in range(args.n_runs):
            result = _run_condition(
                compiler, tools, scenario, cond_name, agent_factory,
                run_id, data_dir, model_tag, noisy=False)
            all_results.append(result)

    _print_exp11_summary(all_results, model_tag)

    summary_path = f"{data_dir}/exp11_summary_{model_tag}.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    return all_results


# ======================================================================
# EXP-12: Noisy Agent-Driven Tool Selection
# ======================================================================

def run_exp12_noisy_agent_driven(args):
    print("\n" + "=" * 70)
    print("EXP-12: Noisy Agent-Driven Tool Selection (15% failures)")
    print("=" * 70)

    data_dir = "data/exp8_v3"
    os.makedirs(data_dir, exist_ok=True)

    compiler = MeaningCompiler(args.model, n_ctx=16384,
                                n_gpu_layers=args.n_gpu_layers)
    model_tag = (os.path.basename(args.model)
                 .replace("-Q4_K_M.gguf", "")
                 .replace("Qwen3.5-", ""))

    scenario = build_kitchen_scenario(args.n_steps)
    fail_steps = {7, 14, 21, 28}
    fail_steps = {s for s in fail_steps if s <= len(scenario)}

    print(f"  Failure steps: {sorted(fail_steps)} "
          f"({len(fail_steps)}/{len(scenario)} = "
          f"{100 * len(fail_steps) / max(len(scenario), 1):.0f}%)")
    print(f"  Model tag: {model_tag}")

    all_results = []

    for cond_name, agent_factory in [
        ("SIG",
         lambda tools: AgentDrivenSIG(compiler, tools)),
        ("SIG+SECMH-full",
         lambda tools: AgentDrivenSECMH(
             compiler, tools, _make_harness(args.n_steps))),
        ("SIG+SECMH-selective",
         lambda tools: AgentDrivenSelective(
             compiler, tools, _make_harness(args.n_steps))),
    ]:
        print(f"\n--- Condition: {cond_name} ---")
        for run_id in range(args.n_runs):
            random.seed(42)
            compiler.reset_cache()

            base_tools = KitchenToolRegistry()
            noisy_tools = NoisyKitchenToolRegistry(base_tools, fail_steps)
            noisy_tools._step_counter = 0

            agent = agent_factory(noisy_tools)
            agent_metrics = agent.run(scenario, debug=True)

            content_eval = ContentQualityEvaluator(scenario)
            content_result = content_eval.evaluate(
                agent_metrics["gen_texts"], agent_metrics["tool_results"])

            selections = agent_metrics["tool_selections"]
            correct_tools = sum(1 for s in selections if s["correct_tool"])
            tool_accuracy = correct_tools / max(len(selections), 1)
            correct_args = sum(
                1 for s in selections
                if s["correct_tool"] and s["correct_args"])
            args_accuracy = correct_args / max(len(selections), 1)

            failure_count = sum(
                1 for r in agent_metrics["tool_results"]
                if r.startswith("[Error]"))

            result = {
                "experiment": "exp12",
                "condition": cond_name,
                "model": model_tag,
                "run_id": run_id,
                "n_steps": len(scenario),
                "fail_steps": sorted(fail_steps),
                "actual_failures": failure_count,
                "tool_selection_accuracy": round(tool_accuracy, 4),
                "tool_selection_with_args_accuracy": round(args_accuracy, 4),
                "content_composite": round(
                    content_result.get("content_composite", 0), 4),
                "information_coverage": round(
                    content_result.get("information_coverage", 0), 4),
                "response_quality": round(
                    content_result.get("response_quality", 0), 4),
                "semantic_adequacy": round(
                    content_result.get("semantic_adequacy", 0), 4),
                "information_density": round(
                    content_result.get("information_density", 0), 4),
                "context_utilisation": round(
                    content_result.get("context_utilisation", 0), 4),
                "wall_clock_s": round(agent_metrics["total_ttf"], 3),
                "gen_tokens": agent_metrics["total_gen_tokens"],
                "prefill_tokens": agent_metrics["total_prefill_tokens"],
                "completion_count": agent_metrics["completion_count"],
                "cache_size": agent_metrics.get("cache_size", 0),
                "tool_selection_details": selections,
            }

            if "avg_harness_overhead_ms" in agent_metrics:
                result["avg_harness_overhead_ms"] = round(
                    agent_metrics["avg_harness_overhead_ms"], 2)

            all_results.append(result)

            safe_cond = (cond_name.replace('+', '_')
                         .replace('-', '_').lower())
            fname = (f"{data_dir}/exp12_{safe_cond}"
                     f"_{model_tag}_run{run_id}.json")
            with open(fname, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)

            print(f"  Run {run_id}: tool_acc={tool_accuracy:.3f}, "
                  f"failures={failure_count}, "
                  f"CQ={content_result.get('content_composite', 0):.3f}, "
                  f"wc={agent_metrics['total_ttf']:.1f}s")

    _print_exp12_summary(all_results, model_tag)

    summary_path = f"{data_dir}/exp12_summary_{model_tag}.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    return all_results


# ======================================================================
# EXP-13: Path A/B Disentanglement (Forced Selection + SECM-H)
# ======================================================================

def run_exp13_path_ab_disentanglement(args):
    print("\n" + "=" * 70)
    print("EXP-13: Path A/B Disentanglement (Forced Selection + SECM-H)")
    print("=" * 70)

    data_dir = "data/exp8_v3"
    os.makedirs(data_dir, exist_ok=True)

    compiler = MeaningCompiler(args.model, n_ctx=16384,
                                n_gpu_layers=args.n_gpu_layers)
    tools = KitchenToolRegistry()
    scenario = build_kitchen_scenario(args.n_steps)
    model_tag = (os.path.basename(args.model)
                 .replace("-Q4_K_M.gguf", "")
                 .replace("Qwen3.5-", ""))

    print(f"  Scenario: {len(scenario)} steps")
    print(f"  Model tag: {model_tag}")

    all_results = []

    for cond_name, agent_factory in [
        ("ForcedSIG",
         lambda: AgentForcedSelectionSIG(compiler, tools)),
        ("ForcedSECMH",
         lambda: AgentForcedSelectionSECMH(
             compiler, tools, _make_harness(args.n_steps))),
    ]:
        print(f"\n--- Condition: {cond_name} ---")
        for run_id in range(args.n_runs):
            random.seed(42)
            compiler.reset_cache()

            agent = agent_factory()
            agent_metrics = agent.run(scenario, debug=True)

            content_eval = ContentQualityEvaluator(scenario)
            content_result = content_eval.evaluate(
                agent_metrics["gen_texts"], agent_metrics["tool_results"])

            selections = agent_metrics["tool_selections"]
            correct_tools = sum(1 for s in selections if s["correct_tool"])
            tool_accuracy = correct_tools / max(len(selections), 1)
            correct_args = sum(
                1 for s in selections
                if s["correct_tool"] and s["correct_args"])
            args_accuracy = correct_args / max(len(selections), 1)

            result = {
                "experiment": "exp13",
                "condition": cond_name,
                "model": model_tag,
                "run_id": run_id,
                "n_steps": len(scenario),
                "tool_selection_accuracy": round(tool_accuracy, 4),
                "tool_selection_with_args_accuracy": round(
                    args_accuracy, 4),
                "content_composite": round(
                    content_result.get("content_composite", 0), 4),
                "information_coverage": round(
                    content_result.get("information_coverage", 0), 4),
                "response_quality": round(
                    content_result.get("response_quality", 0), 4),
                "semantic_adequacy": round(
                    content_result.get("semantic_adequacy", 0), 4),
                "information_density": round(
                    content_result.get("information_density", 0), 4),
                "context_utilisation": round(
                    content_result.get("context_utilisation", 0), 4),
                "wall_clock_s": round(agent_metrics["total_ttf"], 3),
                "gen_tokens": agent_metrics["total_gen_tokens"],
                "prefill_tokens": agent_metrics["total_prefill_tokens"],
                "completion_count": agent_metrics["completion_count"],
                "cache_size": agent_metrics.get("cache_size", 0),
                "tool_selection_details": selections,
            }

            if "avg_harness_overhead_ms" in agent_metrics:
                result["avg_harness_overhead_ms"] = round(
                    agent_metrics["avg_harness_overhead_ms"], 2)

            all_results.append(result)

            safe_cond = (cond_name.replace('+', '_')
                         .replace('-', '_').lower())
            fname = (f"{data_dir}/exp13_{safe_cond}"
                     f"_{model_tag}_run{run_id}.json")
            with open(fname, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)

            print(f"  Run {run_id}: tool_acc={tool_accuracy:.3f}, "
                  f"args_acc={args_accuracy:.3f}, "
                  f"CQ={content_result.get('content_composite', 0):.3f}, "
                  f"wc={agent_metrics['total_ttf']:.1f}s")

    _print_exp13_summary(all_results, model_tag)

    summary_path = f"{data_dir}/exp13_summary_{model_tag}.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    return all_results


# ======================================================================
# EXP-14: Natural Language State Rendering
# ======================================================================

def run_exp14_natural_language_state(args):
    print("\n" + "=" * 70)
    print("EXP-14: Natural Language State Rendering")
    print("=" * 70)

    data_dir = "data/exp8_v3"
    os.makedirs(data_dir, exist_ok=True)

    compiler = MeaningCompiler(args.model, n_ctx=16384,
                                n_gpu_layers=args.n_gpu_layers)
    tools = KitchenToolRegistry()
    scenario = build_kitchen_scenario(args.n_steps)
    model_tag = (os.path.basename(args.model)
                 .replace("-Q4_K_M.gguf", "")
                 .replace("Qwen3.5-", ""))

    print(f"  Scenario: {len(scenario)} steps")
    print(f"  Model tag: {model_tag}")

    all_results = []

    for cond_name, agent_factory in [
        ("SECMH-template",
         lambda: AgentDrivenSECMH(
             compiler, tools, _make_harness(args.n_steps))),
        ("SECMH-natural",
         lambda: AgentDrivenSECMH_NL(
             compiler, tools, _make_harness(args.n_steps))),
    ]:
        print(f"\n--- Condition: {cond_name} ---")
        for run_id in range(args.n_runs):
            random.seed(42)
            compiler.reset_cache()

            agent = agent_factory()
            agent_metrics = agent.run(scenario, debug=True)

            content_eval = ContentQualityEvaluator(scenario)
            content_result = content_eval.evaluate(
                agent_metrics["gen_texts"], agent_metrics["tool_results"])

            selections = agent_metrics["tool_selections"]
            correct_tools = sum(1 for s in selections if s["correct_tool"])
            tool_accuracy = correct_tools / max(len(selections), 1)
            correct_args = sum(
                1 for s in selections
                if s["correct_tool"] and s["correct_args"])
            args_accuracy = correct_args / max(len(selections), 1)

            result = {
                "experiment": "exp14",
                "condition": cond_name,
                "model": model_tag,
                "run_id": run_id,
                "n_steps": len(scenario),
                "tool_selection_accuracy": round(tool_accuracy, 4),
                "tool_selection_with_args_accuracy": round(
                    args_accuracy, 4),
                "content_composite": round(
                    content_result.get("content_composite", 0), 4),
                "information_coverage": round(
                    content_result.get("information_coverage", 0), 4),
                "response_quality": round(
                    content_result.get("response_quality", 0), 4),
                "semantic_adequacy": round(
                    content_result.get("semantic_adequacy", 0), 4),
                "information_density": round(
                    content_result.get("information_density", 0), 4),
                "context_utilisation": round(
                    content_result.get("context_utilisation", 0), 4),
                "wall_clock_s": round(agent_metrics["total_ttf"], 3),
                "gen_tokens": agent_metrics["total_gen_tokens"],
                "prefill_tokens": agent_metrics["total_prefill_tokens"],
                "completion_count": agent_metrics["completion_count"],
                "cache_size": agent_metrics.get("cache_size", 0),
                "tool_selection_details": selections,
            }

            if "avg_harness_overhead_ms" in agent_metrics:
                result["avg_harness_overhead_ms"] = round(
                    agent_metrics["avg_harness_overhead_ms"], 2)

            all_results.append(result)

            safe_cond = (cond_name.replace('+', '_')
                         .replace('-', '_').lower())
            fname = (f"{data_dir}/exp14_{safe_cond}"
                     f"_{model_tag}_run{run_id}.json")
            with open(fname, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)

            print(f"  Run {run_id}: tool_acc={tool_accuracy:.3f}, "
                  f"args_acc={args_accuracy:.3f}, "
                  f"CQ={content_result.get('content_composite', 0):.3f}, "
                  f"wc={agent_metrics['total_ttf']:.1f}s")

    _print_exp14_summary(all_results, model_tag)

    summary_path = f"{data_dir}/exp14_summary_{model_tag}.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    return all_results


# ======================================================================
# EXP-15: 0.8B Agent-Driven
# ======================================================================

def run_exp15_08b(args):
    original_model = args.model
    args.model = ("d:/trunk/SIG/output/cognitive-outsourcing/"
                  "models/Qwen3.5-0.8B-Q4_K_M.gguf")
    print("\n" + "=" * 70)
    print("EXP-15: Agent-Driven Tool Selection (0.8B)")
    print("=" * 70)
    run_exp11_agent_driven(args)
    run_exp12_noisy_agent_driven(args)
    args.model = original_model


# ======================================================================
# Main
# ======================================================================

def main():
    parser = argparse.ArgumentParser(
        description="EXP-11/12/13/14/15: Agent-Driven Tool Selection")
    parser.add_argument("--task",
                        choices=["exp11", "exp12", "exp13", "exp14",
                                 "exp15", "all"],
                        default="all")
    parser.add_argument("--model",
                        default="d:/trunk/SIG/Qwen3.5-2B-Q4_K_M.gguf")
    parser.add_argument("--n-gpu-layers", type=int, default=99)
    parser.add_argument("--n-runs", type=int, default=3)
    parser.add_argument("--n-steps", type=int, default=35)
    args = parser.parse_args()

    print(f"Agent-Driven Tool Selection Experiments")
    print(f"Model: {args.model}")
    print(f"Runs per condition: {args.n_runs}")
    print(f"Steps: {args.n_steps}")

    t0 = time.time()

    if args.task in ("exp11", "all"):
        run_exp11_agent_driven(args)
    if args.task in ("exp12", "all"):
        run_exp12_noisy_agent_driven(args)
    if args.task in ("exp13", "all"):
        run_exp13_path_ab_disentanglement(args)
    if args.task in ("exp14", "all"):
        run_exp14_natural_language_state(args)
    if args.task in ("exp15", "all"):
        run_exp15_08b(args)

    elapsed = time.time() - t0
    print(f"\n{'=' * 70}")
    print(f"Total elapsed: {elapsed:.1f}s ({elapsed / 60:.1f} min)")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
