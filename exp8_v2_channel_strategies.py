#!/usr/bin/env python3
"""
EXP-9/10: SECM-H Channel Strategy Comparison
=============================================
Addresses reviewer criticism that naive KV-cache injection degrades quality.

Strategies:
  Selective — inject only at tool-switching decision points (≤20 tokens)
  OOB       — out-of-band status hint prepended to user query (≤15 tokens)
  Sweep     — inject truncated state at configurable token budgets (0–120)

Experiments:
  EXP-9: Injection amount sweep across all strategies
  EXP-10: Noisy Kitchen — 15% tool-call failures to stress-test strategies

Usage:
  python exp8_v2_channel_strategies.py --task sweep --n-runs 3
  python exp8_v2_channel_strategies.py --task noisy --n-runs 3
  python exp8_v2_channel_strategies.py --task all --n-runs 3
"""

import time, json, argparse, random, os, sys, re
from typing import List, Dict, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core import MeaningCompiler, InjectionEngine, init_metrics
from core.harness import (
    SECMHarness, ModuleRegistry, InvocationHistory,
    ConfidenceTracker, DependencyGraph, PatternCache,
    BudgetTracker, StateRenderer,
)

KITCHEN_SYSTEM_PROMPT = """You are an intelligent kitchen assistant running on an edge device.
You help users with recipe planning, real-time cooking guidance, inventory management,
and handling interruptions. Always consider dietary profile and kitchen state.
Be concise and specific."""

from edge_agent_bench import (
    KitchenToolRegistry, KitchenStep, build_kitchen_scenario,
    _check_hit, build_probe_queries, EdgeKitchenSIG,
)
from exp8_kitchen_benchmark import evaluate_quality


# ======================================================================
# Noisy tool registry wrapper
# ======================================================================

class NoisyKitchenToolRegistry:
    """KitchenToolRegistry wrapper that injects deterministic failures."""

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
# Agent: SIG + SECM-H Selective
# ======================================================================

class EdgeKitchenSIG_Selective:
    """Selective injection — only at tool-switching decision points.

    Injects a minimal hint (≤20 tokens) when the tool name changes
    from the previous step, indicating a task-type transition.
    """

    def __init__(self, compiler, tools, harness: SECMHarness):
        self.compiler = compiler
        self.tools = tools
        self.engine = InjectionEngine(compiler)
        self.harness = harness

    def run(self, scenario, probes=None, max_new=80, debug=False):
        metrics = init_metrics()
        metrics["harness_overhead_ms"] = []
        metrics["rendered_state_tokens"] = []
        metrics["gen_texts"] = []
        metrics["tool_results"] = []
        metrics["injection_count"] = 0

        self.engine.reset()
        sys_ids = list(self.compiler.tokenize(f"{KITCHEN_SYSTEM_PROMPT}\n\n", add_bos=False))
        pf_t0 = time.time()
        self.compiler.eval(sys_ids)
        metrics["total_prefill_time"] += time.time() - pf_t0
        metrics["total_prefill_tokens"] += len(sys_ids)
        self.engine.update_cache(sys_ids)

        prev_tool = None
        probe_idx = 0
        probe_results = []
        wc_start = time.time()

        for step_i, step in enumerate(scenario):
            step_t0 = time.time()
            harness_t0 = time.time()

            self.harness.pre_invoke(step.tool_name, step.tool_args)

            is_decision_point = (prev_tool is not None and step.tool_name != prev_tool)
            hint_text = ""
            if is_decision_point:
                top = self.harness.confidence.get_top(1)
                if top:
                    top_name, top_conf = top[0]
                    hint_text = f"\n[HINT] try: {top_name}({top_conf:.1f})\n"

            harness_ms = (time.time() - harness_t0) * 1000
            metrics["harness_overhead_ms"].append(harness_ms)

            user_line = f"\nUser: {step.user_query}\nAssistant:"
            u_ids = list(self.compiler.tokenize(user_line, add_bos=False))
            self.compiler.eval(u_ids)
            metrics["total_prefill_tokens"] += len(u_ids)
            self.engine.update_cache(u_ids)

            result = self.tools.execute(step.tool_name, step.tool_args)
            metrics["tool_results"].append(result)
            tool_line = f"\n[Tool: {step.tool_name}] {result}\n"
            t_ids = list(self.compiler.tokenize(tool_line, add_bos=False))
            self.compiler.eval(t_ids)
            metrics["total_prefill_tokens"] += len(t_ids)
            self.engine.update_cache(t_ids)

            if hint_text:
                hint_ids = list(self.compiler.tokenize(hint_text, add_bos=False))
                ht0 = time.time()
                self.compiler.eval(hint_ids)
                metrics["total_prefill_time"] += time.time() - ht0
                metrics["total_prefill_tokens"] += len(hint_ids)
                self.engine.update_cache(hint_ids)
                metrics["rendered_state_tokens"].append(len(hint_ids))
                metrics["injection_count"] += 1
            else:
                metrics["rendered_state_tokens"].append(0)

            quality_est = min(1.0, len(result) / 200.0)
            h2_t0 = time.time()
            self.harness.post_invoke(
                step.tool_name, step.tool_args, result,
                latency_ms=harness_ms, success=True, quality=quality_est
            )
            harness_ms += (time.time() - h2_t0) * 1000

            gen_t0 = time.time()
            gen_text, gen_ids = self.compiler.generate_until_str(
                "\nUser:", max_new=max_new, rep_threshold=3)
            metrics["total_gen_time"] += time.time() - gen_t0
            metrics["total_gen_tokens"] += len(gen_ids)
            metrics["gen_texts"].append(gen_text)
            self.engine.update_cache(list(gen_ids))

            metrics["per_turn_ttf"].append(time.time() - step_t0)

            if probes and probe_idx < len(probes):
                p = probes[probe_idx]
                if p["probe_at_step"] <= step_i:
                    probe_q = f"\nUser: {p['query']}\nAssistant:"
                    pq_ids = list(self.compiler.tokenize(probe_q, add_bos=False))
                    self.compiler.eval(pq_ids)
                    self.engine.update_cache(pq_ids)
                    p_text, _ = self.compiler.generate_until_str("\nUser:", max_new=30)
                    hit = _check_hit(p["expected"], p_text)
                    probe_results.append(dict(step=step_i, expected=p["expected"],
                        actual=p_text.strip()[:60], hit=hit, source_step=p["source_step"]))
                    probe_idx += 1

            prev_tool = step.tool_name

            if debug and step_i % 10 == 0:
                print(f"  Selective step {step_i + 1}/{len(scenario)} — "
                      f"injections: {metrics['injection_count']}, "
                      f"cache: {self.engine.cache_size} tok")

        metrics["total_ttf"] = time.time() - wc_start
        metrics["completion_count"] = len(scenario)
        metrics["failure_count"] = 0
        metrics["probe_results"] = probe_results
        metrics["cache_size"] = self.engine.cache_size

        oh = metrics["harness_overhead_ms"]
        metrics["avg_harness_overhead_ms"] = sum(oh) / len(oh) if oh else 0
        metrics["p95_harness_overhead_ms"] = sorted(oh)[int(0.95 * len(oh))] if oh else 0
        rt = metrics["rendered_state_tokens"]
        metrics["avg_rendered_state_tokens"] = sum(rt) / len(rt) if rt else 0

        return metrics


# ======================================================================
# Agent: SIG + SECM-H OOB (Out-of-Band)
# ======================================================================

class EdgeKitchenSIG_OOB:
    """Out-of-band query strategy — never inject into KV cache.

    Instead, prepends a one-line status hint to each user query so the
    model receives minimal task-progress information as part of the
    normal prompt (≤15 tokens).
    """

    def __init__(self, compiler, tools, harness: SECMHarness):
        self.compiler = compiler
        self.tools = tools
        self.engine = InjectionEngine(compiler)
        self.harness = harness

    def run(self, scenario, probes=None, max_new=80, debug=False):
        metrics = init_metrics()
        metrics["harness_overhead_ms"] = []
        metrics["rendered_state_tokens"] = []
        metrics["gen_texts"] = []
        metrics["tool_results"] = []
        metrics["injection_count"] = 0

        self.engine.reset()
        sys_ids = list(self.compiler.tokenize(f"{KITCHEN_SYSTEM_PROMPT}\n\n", add_bos=False))
        pf_t0 = time.time()
        self.compiler.eval(sys_ids)
        metrics["total_prefill_time"] += time.time() - pf_t0
        metrics["total_prefill_tokens"] += len(sys_ids)
        self.engine.update_cache(sys_ids)

        probe_idx = 0
        probe_results = []
        wc_start = time.time()
        n_total = len(scenario)
        state_summary = self.harness.get_state_summary()

        for step_i, step in enumerate(scenario):
            step_t0 = time.time()
            harness_t0 = time.time()

            self.harness.pre_invoke(step.tool_name, step.tool_args)
            harness_ms = (time.time() - harness_t0) * 1000
            metrics["harness_overhead_ms"].append(harness_ms)

            n_invoked = len(self.harness.history.get_invoked_modules())
            budget_used = int(self.harness.budget.get_utilization()
                              * self.harness.budget._total)
            budget_total = self.harness.budget._total
            status_hint = f"\n[STATUS] {n_invoked}/{n_total} done. Budget: {budget_used}/{budget_total}."

            hint_ids = list(self.compiler.tokenize(status_hint, add_bos=False))
            metrics["rendered_state_tokens"].append(len(hint_ids))

            user_line = f"\nUser: {step.user_query} {status_hint}\nAssistant:"
            u_ids = list(self.compiler.tokenize(user_line, add_bos=False))
            self.compiler.eval(u_ids)
            metrics["total_prefill_tokens"] += len(u_ids)
            self.engine.update_cache(u_ids)

            result = self.tools.execute(step.tool_name, step.tool_args)
            metrics["tool_results"].append(result)
            tool_line = f"\n[Tool: {step.tool_name}] {result}\n"
            t_ids = list(self.compiler.tokenize(tool_line, add_bos=False))
            self.compiler.eval(t_ids)
            metrics["total_prefill_tokens"] += len(t_ids)
            self.engine.update_cache(t_ids)

            quality_est = min(1.0, len(result) / 200.0)
            h2_t0 = time.time()
            self.harness.post_invoke(
                step.tool_name, step.tool_args, result,
                latency_ms=harness_ms, success=True, quality=quality_est
            )
            harness_ms += (time.time() - h2_t0) * 1000

            gen_t0 = time.time()
            gen_text, gen_ids = self.compiler.generate_until_str(
                "\nUser:", max_new=max_new, rep_threshold=3)
            metrics["total_gen_time"] += time.time() - gen_t0
            metrics["total_gen_tokens"] += len(gen_ids)
            metrics["gen_texts"].append(gen_text)
            self.engine.update_cache(list(gen_ids))

            metrics["per_turn_ttf"].append(time.time() - step_t0)

            if probes and probe_idx < len(probes):
                p = probes[probe_idx]
                if p["probe_at_step"] <= step_i:
                    probe_q = f"\nUser: {p['query']}\nAssistant:"
                    pq_ids = list(self.compiler.tokenize(probe_q, add_bos=False))
                    self.compiler.eval(pq_ids)
                    self.engine.update_cache(pq_ids)
                    p_text, _ = self.compiler.generate_until_str("\nUser:", max_new=30)
                    hit = _check_hit(p["expected"], p_text)
                    probe_results.append(dict(step=step_i, expected=p["expected"],
                        actual=p_text.strip()[:60], hit=hit, source_step=p["source_step"]))
                    probe_idx += 1

            if debug and step_i % 10 == 0:
                print(f"  OOB step {step_i + 1}/{len(scenario)} — "
                      f"cache: {self.engine.cache_size} tok")

        metrics["total_ttf"] = time.time() - wc_start
        metrics["completion_count"] = len(scenario)
        metrics["failure_count"] = 0
        metrics["probe_results"] = probe_results
        metrics["cache_size"] = self.engine.cache_size

        oh = metrics["harness_overhead_ms"]
        metrics["avg_harness_overhead_ms"] = sum(oh) / len(oh) if oh else 0
        metrics["p95_harness_overhead_ms"] = sorted(oh)[int(0.95 * len(oh))] if oh else 0
        rt = metrics["rendered_state_tokens"]
        metrics["avg_rendered_state_tokens"] = sum(rt) / len(rt) if rt else 0

        return metrics


# ======================================================================
# Agent: SIG + SECM-H Sweep (configurable injection budget)
# ======================================================================

class EdgeKitchenSIG_Sweep:
    """Injection amount sweep — truncate rendered state to max_inject_tokens.

    When max_inject_tokens=0, behaves identically to bare SIG (no injection).
    """

    def __init__(self, compiler, tools, harness: SECMHarness,
                 max_inject_tokens: int = 20):
        self.compiler = compiler
        self.tools = tools
        self.engine = InjectionEngine(compiler)
        self.harness = harness
        self.max_inject_tokens = max_inject_tokens

    def run(self, scenario, probes=None, max_new=80, debug=False):
        metrics = init_metrics()
        metrics["harness_overhead_ms"] = []
        metrics["rendered_state_tokens"] = []
        metrics["gen_texts"] = []
        metrics["tool_results"] = []
        metrics["injection_count"] = 0

        self.engine.reset()
        sys_ids = list(self.compiler.tokenize(f"{KITCHEN_SYSTEM_PROMPT}\n\n", add_bos=False))
        pf_t0 = time.time()
        self.compiler.eval(sys_ids)
        metrics["total_prefill_time"] += time.time() - pf_t0
        metrics["total_prefill_tokens"] += len(sys_ids)
        self.engine.update_cache(sys_ids)

        probe_idx = 0
        probe_results = []
        wc_start = time.time()

        for step_i, step in enumerate(scenario):
            step_t0 = time.time()
            harness_t0 = time.time()

            self.harness.pre_invoke(step.tool_name, step.tool_args)

            state_text = self.harness.render_state()

            inject_ids = []
            if self.max_inject_tokens > 0 and state_text:
                full_ids = list(self.compiler.tokenize(f"\n{state_text}\n", add_bos=False))
                inject_ids = full_ids[:self.max_inject_tokens]

            harness_ms = (time.time() - harness_t0) * 1000
            metrics["harness_overhead_ms"].append(harness_ms)

            user_line = f"\nUser: {step.user_query}\nAssistant:"
            u_ids = list(self.compiler.tokenize(user_line, add_bos=False))
            self.compiler.eval(u_ids)
            metrics["total_prefill_tokens"] += len(u_ids)
            self.engine.update_cache(u_ids)

            result = self.tools.execute(step.tool_name, step.tool_args)
            metrics["tool_results"].append(result)
            tool_line = f"\n[Tool: {step.tool_name}] {result}\n"
            t_ids = list(self.compiler.tokenize(tool_line, add_bos=False))
            self.compiler.eval(t_ids)
            metrics["total_prefill_tokens"] += len(t_ids)
            self.engine.update_cache(t_ids)

            if inject_ids:
                inject_t0 = time.time()
                self.compiler.eval(inject_ids)
                metrics["total_prefill_time"] += time.time() - inject_t0
                metrics["total_prefill_tokens"] += len(inject_ids)
                self.engine.update_cache(inject_ids)
                metrics["rendered_state_tokens"].append(len(inject_ids))
                metrics["injection_count"] += 1
            else:
                metrics["rendered_state_tokens"].append(0)

            quality_est = min(1.0, len(result) / 200.0)
            h2_t0 = time.time()
            self.harness.post_invoke(
                step.tool_name, step.tool_args, result,
                latency_ms=harness_ms, success=True, quality=quality_est
            )
            harness_ms += (time.time() - h2_t0) * 1000

            gen_t0 = time.time()
            gen_text, gen_ids = self.compiler.generate_until_str(
                "\nUser:", max_new=max_new, rep_threshold=3)
            metrics["total_gen_time"] += time.time() - gen_t0
            metrics["total_gen_tokens"] += len(gen_ids)
            metrics["gen_texts"].append(gen_text)
            self.engine.update_cache(list(gen_ids))

            metrics["per_turn_ttf"].append(time.time() - step_t0)

            if probes and probe_idx < len(probes):
                p = probes[probe_idx]
                if p["probe_at_step"] <= step_i:
                    probe_q = f"\nUser: {p['query']}\nAssistant:"
                    pq_ids = list(self.compiler.tokenize(probe_q, add_bos=False))
                    self.compiler.eval(pq_ids)
                    self.engine.update_cache(pq_ids)
                    p_text, _ = self.compiler.generate_until_str("\nUser:", max_new=30)
                    hit = _check_hit(p["expected"], p_text)
                    probe_results.append(dict(step=step_i, expected=p["expected"],
                        actual=p_text.strip()[:60], hit=hit, source_step=p["source_step"]))
                    probe_idx += 1

            if debug and step_i % 10 == 0:
                avg_inj = metrics["rendered_state_tokens"]
                avg_tok = sum(avg_inj) / len(avg_inj) if avg_inj else 0
                print(f"  Sweep(max={self.max_inject_tokens}) step {step_i + 1}/{len(scenario)} — "
                      f"inject: {avg_tok:.0f} tok/step, cache: {self.engine.cache_size}")

        metrics["total_ttf"] = time.time() - wc_start
        metrics["completion_count"] = len(scenario)
        metrics["failure_count"] = 0
        metrics["probe_results"] = probe_results
        metrics["cache_size"] = self.engine.cache_size

        oh = metrics["harness_overhead_ms"]
        metrics["avg_harness_overhead_ms"] = sum(oh) / len(oh) if oh else 0
        metrics["p95_harness_overhead_ms"] = sorted(oh)[int(0.95 * len(oh))] if oh else 0
        rt = metrics["rendered_state_tokens"]
        metrics["avg_rendered_state_tokens"] = sum(rt) / len(rt) if rt else 0

        return metrics


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
# EXP-9: Injection Strategy Sweep
# ======================================================================

def run_exp9_sweep(args):
    """EXP-9: Compare injection strategies and token budgets."""
    print("\n" + "=" * 70)
    print("EXP-9: SECM-H Channel Strategy Sweep")
    print("=" * 70)

    os.makedirs("data/exp8_v2", exist_ok=True)

    compiler = MeaningCompiler(args.model, n_ctx=16384, n_gpu_layers=args.n_gpu_layers)
    tools = KitchenToolRegistry()
    scenario = build_kitchen_scenario(args.n_steps)
    probes = build_probe_queries(scenario, num_probes=3)
    model_tag = os.path.basename(args.model).replace("-Q4_K_M.gguf", "").replace("Qwen3.5-", "")

    sweep_values = [0, 5, 10, 20, 40, 80, 120]

    conditions = []
    conditions.append(("SIG", None))
    for sv in sweep_values:
        conditions.append((f"Sweep-{sv}", sv))
    conditions.append(("Selective", "selective"))
    conditions.append(("OOB", "oob"))

    all_results = []

    for cond_name, cond_cfg in conditions:
        print(f"\n--- Condition: {cond_name} ---")
        for run_id in range(args.n_runs):
            random.seed(42)
            compiler.reset_cache()

            if cond_name == "SIG":
                agent = EdgeKitchenSIG(compiler, tools)
                metrics = agent.run(scenario, probes=probes, debug=True)
            elif cond_cfg == "selective":
                harness = _make_harness(args.n_steps)
                agent = EdgeKitchenSIG_Selective(compiler, tools, harness)
                metrics = agent.run(scenario, probes=probes, debug=True)
            elif cond_cfg == "oob":
                harness = _make_harness(args.n_steps)
                agent = EdgeKitchenSIG_OOB(compiler, tools, harness)
                metrics = agent.run(scenario, probes=probes, debug=True)
            else:
                harness = _make_harness(args.n_steps)
                agent = EdgeKitchenSIG_Sweep(compiler, tools, harness,
                                              max_inject_tokens=cond_cfg)
                metrics = agent.run(scenario, probes=probes, debug=True)

            quality = evaluate_quality(scenario, metrics, tools)

            inject_tok_per_step = 0.0
            if "rendered_state_tokens" in metrics and metrics["rendered_state_tokens"]:
                inject_tok_per_step = sum(metrics["rendered_state_tokens"]) / len(
                    metrics["rendered_state_tokens"])
            total_inject = sum(metrics.get("rendered_state_tokens", [0]))

            result = {
                "experiment": "exp9",
                "condition": cond_name,
                "model": model_tag,
                "run_id": run_id,
                "n_steps": len(scenario),
                "wall_clock_s": round(metrics["total_ttf"], 3),
                "prefill_tokens": metrics["total_prefill_tokens"],
                "gen_tokens": metrics["total_gen_tokens"],
                "completion_count": metrics["completion_count"],
                "quality_composite": round(quality["composite"], 4),
                "content_composite": round(quality.get("content_composite", 0), 4),
                "information_coverage": round(quality.get("information_coverage", 0), 4),
                "response_quality": round(quality.get("response_quality", 0), 4),
                "context_utilisation": round(quality.get("context_utilisation", 0), 4),
                "inject_tokens_per_step": round(inject_tok_per_step, 1),
                "total_inject_tokens": total_inject,
                "injection_count": metrics.get("injection_count",
                                               len(scenario) if cond_name != "SIG" else 0),
            }

            if "avg_harness_overhead_ms" in metrics:
                result["avg_harness_overhead_ms"] = round(
                    metrics.get("avg_harness_overhead_ms", 0), 2)

            all_results.append(result)

            fname = (f"data/exp8_v2/exp9_{cond_name.replace('-', '_').lower()}"
                     f"_{model_tag}_run{run_id}.json")
            with open(fname, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)

            print(f"  Run {run_id}: wc={result['wall_clock_s']:.1f}s, "
                  f"CQ={result['content_composite']:.3f}, "
                  f"inject={inject_tok_per_step:.0f} tok/step")

    _print_exp9_summary(all_results, model_tag)

    summary_path = f"data/exp8_v2/exp9_summary_{model_tag}.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    return all_results


def _print_exp9_summary(results, model_tag):
    print("\n" + "=" * 100)
    print(f"EXP-9 Summary: Channel Strategy Comparison ({model_tag})")
    print("=" * 100)

    cond_order = ["SIG"]
    for sv in [0, 5, 10, 20, 40, 80, 120]:
        cond_order.append(f"Sweep-{sv}")
    cond_order += ["Selective", "OOB"]

    header = (f"{'Condition':<14} {'InjTok/Step':<13} {'TotalInj':<10} "
              f"{'ContentQ':<10} {'Coverage':<10} {'RespQ':<10} "
              f"{'WallClock':<11} {'GenTok':<8}")
    print(header)
    print("-" * len(header))

    for cond in cond_order:
        runs = [r for r in results if r["condition"] == cond]
        if not runs:
            continue
        n = len(runs)
        inj_step = sum(r["inject_tokens_per_step"] for r in runs) / n
        total_inj = sum(r["total_inject_tokens"] for r in runs) / n
        cq = sum(r["content_composite"] for r in runs) / n
        cov = sum(r["information_coverage"] for r in runs) / n
        rq = sum(r["response_quality"] for r in runs) / n
        wc = sum(r["wall_clock_s"] for r in runs) / n
        gt = sum(r["gen_tokens"] for r in runs) / n
        print(f"{cond:<14} {inj_step:<13.1f} {total_inj:<10.0f} "
              f"{cq:<10.3f} {cov:<10.3f} {rq:<10.3f} "
              f"{wc:<11.1f} {gt:<8.0f}")

    sig_runs = [r for r in results if r["condition"] == "SIG"]
    if sig_runs:
        sig_cq = sum(r["content_composite"] for r in sig_runs) / len(sig_runs)
        print(f"\nΔ vs SIG baseline (ContentQ = {sig_cq:.3f}):")
        for cond in cond_order:
            if cond == "SIG":
                continue
            runs = [r for r in results if r["condition"] == cond]
            if not runs:
                continue
            cq = sum(r["content_composite"] for r in runs) / len(runs)
            delta = cq - sig_cq
            wc = sum(r["wall_clock_s"] for r in runs) / len(runs)
            sig_wc = sum(r["wall_clock_s"] for r in sig_runs) / len(sig_runs)
            wc_ratio = wc / max(sig_wc, 0.001)
            print(f"  {cond:<14} ΔCQ={delta:+.4f}  WC={wc_ratio:.2f}x")


# ======================================================================
# EXP-10: Noisy Kitchen
# ======================================================================

def run_exp10_noisy(args):
    """EXP-10: Noisy Kitchen — 15% tool-call failures."""
    print("\n" + "=" * 70)
    print("EXP-10: Noisy Kitchen — Tool Failures")
    print("=" * 70)

    os.makedirs("data/exp8_v2", exist_ok=True)

    compiler = MeaningCompiler(args.model, n_ctx=16384, n_gpu_layers=args.n_gpu_layers)
    model_tag = os.path.basename(args.model).replace("-Q4_K_M.gguf", "").replace("Qwen3.5-", "")

    scenario = build_kitchen_scenario(args.n_steps)
    fail_steps = {7, 14, 21, 28}
    fail_steps = {s for s in fail_steps if s <= len(scenario)}

    print(f"  Failure steps: {sorted(fail_steps)} "
          f"({len(fail_steps)}/{len(scenario)} = "
          f"{100 * len(fail_steps) / len(scenario):.0f}%)")

    all_results = []

    for cond_name, agent_factory in [
        ("SIG", lambda c, t: EdgeKitchenSIG(c, t)),
        ("SIG+SECMH-full", lambda c, t: _make_secmh_full(c, t, args.n_steps)),
        ("SIG+SECMH-selective", lambda c, t: _make_secmh_selective(c, t, args.n_steps)),
    ]:
        print(f"\n--- Condition: {cond_name} ---")
        for run_id in range(args.n_runs):
            random.seed(42)
            compiler.reset_cache()

            base_tools = KitchenToolRegistry()
            noisy_tools = NoisyKitchenToolRegistry(base_tools, fail_steps)
            noisy_tools._step_counter = 0

            agent = agent_factory(compiler, noisy_tools)
            metrics = agent.run(scenario, debug=True)
            quality = evaluate_quality(scenario, metrics, noisy_tools)

            inject_tok_per_step = 0.0
            if "rendered_state_tokens" in metrics and metrics["rendered_state_tokens"]:
                inject_tok_per_step = sum(metrics["rendered_state_tokens"]) / len(
                    metrics["rendered_state_tokens"])

            result = {
                "experiment": "exp10",
                "condition": cond_name,
                "model": model_tag,
                "run_id": run_id,
                "n_steps": len(scenario),
                "fail_steps": sorted(fail_steps),
                "wall_clock_s": round(metrics["total_ttf"], 3),
                "gen_tokens": metrics["total_gen_tokens"],
                "completion_count": metrics["completion_count"],
                "quality_composite": round(quality["composite"], 4),
                "content_composite": round(quality.get("content_composite", 0), 4),
                "information_coverage": round(quality.get("information_coverage", 0), 4),
                "response_quality": round(quality.get("response_quality", 0), 4),
                "context_utilisation": round(quality.get("context_utilisation", 0), 4),
                "inject_tokens_per_step": round(inject_tok_per_step, 1),
                "injection_count": metrics.get("injection_count",
                                               len(scenario) if "SECMH-full" in cond_name else 0),
            }

            all_results.append(result)

            fname = (f"data/exp8_v2/exp10_{cond_name.replace('+', '_').lower()}"
                     f"_{model_tag}_run{run_id}.json")
            with open(fname, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)

            print(f"  Run {run_id}: wc={result['wall_clock_s']:.1f}s, "
                  f"Q={result['quality_composite']:.3f}, "
                  f"CQ={result['content_composite']:.3f}")

    _print_exp10_summary(all_results, model_tag)

    summary_path = f"data/exp8_v2/exp10_summary_{model_tag}.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    return all_results


def _make_secmh_full(compiler, tools, n_steps):
    harness = _make_harness(n_steps)
    from exp8_kitchen_benchmark import EdgeKitchenSIG_SECMH
    return EdgeKitchenSIG_SECMH(compiler, tools, harness)


def _make_secmh_selective(compiler, tools, n_steps):
    harness = _make_harness(n_steps)
    return EdgeKitchenSIG_Selective(compiler, tools, harness)


def _print_exp10_summary(results, model_tag):
    print("\n" + "=" * 90)
    print(f"EXP-10 Summary: Noisy Kitchen ({model_tag})")
    print("=" * 90)

    header = (f"{'Condition':<24} {'Quality':<10} {'ContentQ':<10} "
              f"{'Coverage':<10} {'RespQ':<10} {'WallClock':<11}")
    print(header)
    print("-" * len(header))

    for cond in ["SIG", "SIG+SECMH-full", "SIG+SECMH-selective"]:
        runs = [r for r in results if r["condition"] == cond]
        if not runs:
            continue
        n = len(runs)
        q = sum(r["quality_composite"] for r in runs) / n
        cq = sum(r["content_composite"] for r in runs) / n
        cov = sum(r["information_coverage"] for r in runs) / n
        rq = sum(r["response_quality"] for r in runs) / n
        wc = sum(r["wall_clock_s"] for r in runs) / n
        print(f"{cond:<24} {q:<10.3f} {cq:<10.3f} {cov:<10.3f} {rq:<10.3f} {wc:<11.1f}")

    sig_runs = [r for r in results if r["condition"] == "SIG"]
    if sig_runs:
        sig_cq = sum(r["content_composite"] for r in sig_runs) / len(sig_runs)
        sig_q = sum(r["quality_composite"] for r in sig_runs) / len(sig_runs)
        print(f"\nΔ vs SIG (Q={sig_q:.3f}, CQ={sig_cq:.3f}):")
        for cond in ["SIG+SECMH-full", "SIG+SECMH-selective"]:
            runs = [r for r in results if r["condition"] == cond]
            if not runs:
                continue
            q = sum(r["quality_composite"] for r in runs) / len(runs)
            cq = sum(r["content_composite"] for r in runs) / len(runs)
            print(f"  {cond:<24} ΔQ={q - sig_q:+.4f}  ΔCQ={cq - sig_cq:+.4f}")


# ======================================================================
# Main
# ======================================================================

def main():
    parser = argparse.ArgumentParser(
        description="EXP-9/10: SECM-H Channel Strategy Comparison")
    parser.add_argument("--task", choices=["sweep", "noisy", "all"], default="all")
    parser.add_argument("--model", default="d:/trunk/SIG/Qwen3.5-2B-Q4_K_M.gguf")
    parser.add_argument("--n-gpu-layers", type=int, default=99)
    parser.add_argument("--n-runs", type=int, default=3)
    parser.add_argument("--n-steps", type=int, default=35)
    args = parser.parse_args()

    print(f"SECM-H Channel Strategy Experiments")
    print(f"Model: {args.model}")
    print(f"Runs per condition: {args.n_runs}")
    print(f"Steps: {args.n_steps}")

    t0 = time.time()

    if args.task in ("sweep", "all"):
        run_exp9_sweep(args)
    if args.task in ("noisy", "all"):
        run_exp10_noisy(args)

    elapsed = time.time() - t0
    print(f"\n{'=' * 70}")
    print(f"Total elapsed: {elapsed:.1f}s ({elapsed / 60:.1f} min)")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
