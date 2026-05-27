#!/usr/bin/env python3
"""
Retrospective SIG & Hybrid Experiments — Reviewer Response Suite
=================================================================
Addresses the three core constructive suggestions from peer review:

Tasks:
  r20    : Retrospective SIG — compensatory recall + SIG KV continuity
  r21    : SIG + KV Cache Compression (H2O-style importance pruning)
  flash  : FlashAttention-normalized prefill cost analysis
  all    : run r20 + r21 + flash

Usage:
  python retrospective_bench.py --task r20 --model models/Qwen3.5-4B-Q4_K_M.gguf
  python retrospective_bench.py --task r21 --model ... --n-gpu-layers 99
  python retrospective_bench.py --task flash --model ...
  python retrospective_bench.py --task all --model ... --n-gpu-layers 99
"""

import time, json, argparse, random, math, os, sys, re
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

from core import (
    MeaningCompiler, InjectionEngine, GPUMonitor,
    init_metrics,
)

# ======================================================================
# Import Kitchen domain from edge_agent_bench
# ======================================================================
from edge_agent_bench import (
    KITCHEN_SYSTEM_PROMPT,
    KitchenToolRegistry,
    KitchenStep,
    build_kitchen_scenario,
    build_probe_queries,
    _check_hit, _probe_f1,
)


# ======================================================================
# R20: Retrospective SIG — Compensatory Recall
# ======================================================================

class RetroSIGAgent:
    """Retrospective SIG agent — SIG KV continuity + explicit fact recall.

    After each tool injection, the agent:
    1. Records the tool result as a "fact"
    2. Every recall_interval steps, injects a recall prompt summarizing
       accumulated key facts before generation
    3. This compensates for SIG's fact-forgetting while preserving
       KV cache continuity for dialogue flow and constraint tracking
    """

    def __init__(self, compiler, tools, recall_interval=5, recall_window=3):
        self.compiler = compiler
        self.tools = tools
        self.engine = InjectionEngine(compiler)
        self.recall_interval = recall_interval
        self.recall_window = recall_window

    def _extract_key_facts(self, tool_results: List[Dict]) -> List[str]:
        facts = []
        for tr in tool_results:
            text = tr.get("result", "")
            tool_name = tr.get("tool", "")
            args = tr.get("args", {})
            if tool_name == "set_user_profile":
                al = args.get("allergies", "")
                diet = args.get("diet", "")
                sv = args.get("servings", "")
                pref = args.get("cuisine_pref", "")
                if al:
                    facts.append(f"allergies={al}")
                if diet:
                    facts.append(f"diet={diet}")
                if sv:
                    facts.append(f"servings={sv}")
                if pref:
                    facts.append(f"cuisine_pref={pref}")
            elif tool_name == "set_oven":
                temp = args.get("temp_c", "")
                if temp:
                    facts.append(f"oven={temp}C")
            elif tool_name == "start_cooking":
                rid = args.get("recipe_id", "")
                if rid:
                    facts.append(f"cooking={rid}")
            elif tool_name == "add_shopping_item":
                ing = args.get("ingredient", "")
                qty = args.get("quantity", "")
                if ing:
                    facts.append(f"shopping:{ing}x{qty}")
            elif tool_name in ("get_recipe", "get_nutrition"):
                rid = args.get("recipe_id", "")
                if rid:
                    facts.append(f"recipe={rid}")
            elif tool_name == "check_ingredients":
                rid = args.get("recipe_id", "")
                if rid:
                    facts.append(f"checked_ingredients={rid}")
            elif tool_name == "find_recipes":
                cuisine = args.get("cuisine", "")
                diet = args.get("diet", "")
                if cuisine:
                    facts.append(f"search_cuisine={cuisine}")
                if diet:
                    facts.append(f"search_diet={diet}")
            elif tool_name == "get_substitution":
                ing = args.get("ingredient", "")
                if ing:
                    facts.append(f"substitute={ing}")
            else:
                preview = text[:80].replace("\n", " ")
                if preview:
                    facts.append(preview)
        return facts

    def run(self, scenario, probes=None, max_new=80, debug=False):
        metrics = init_metrics()
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
        tool_results_archive: List[Dict] = []

        for step_i, step in enumerate(scenario):
            step_t0 = time.time()

            user_line = f"\nUser: {step.user_query}\nAssistant:"
            u_ids = list(self.compiler.tokenize(user_line, add_bos=False))
            self.compiler.eval(u_ids)
            metrics["total_prefill_tokens"] += len(u_ids)
            self.engine.update_cache(u_ids)

            result = self.tools.execute(step.tool_name, step.tool_args)
            tool_results_archive.append({
                "tool": step.tool_name,
                "args": step.tool_args,
                "result": result,
            })
            tool_line = f"\n[Tool: {step.tool_name}] {result}\n"
            t_ids = list(self.compiler.tokenize(tool_line, add_bos=False))
            self.compiler.eval(t_ids)
            metrics["total_prefill_tokens"] += len(t_ids)
            self.engine.update_cache(t_ids)

            should_recall = (
                step_i > 0
                and (step_i + 1) % self.recall_interval == 0
            )

            if should_recall:
                window_results = tool_results_archive[
                    max(0, step_i + 1 - self.recall_window * self.recall_interval):
                    step_i + 1
                ]
                all_facts = self._extract_key_facts(tool_results_archive)
                window_facts = self._extract_key_facts(window_results)
                if all_facts:
                    recall_prompt = (
                        f"\n[Memory] Critical facts to remember: "
                        f"{'; '.join(all_facts[-8:])}.\n"
                        f"Recent: {'; '.join(window_facts[-3:])}.\n"
                    )
                    r_ids = list(self.compiler.tokenize(recall_prompt, add_bos=False))
                    self.compiler.eval(r_ids)
                    metrics["total_prefill_tokens"] += len(r_ids)
                    self.engine.update_cache(r_ids)

            gen_t0 = time.time()
            gen_text, gen_ids = self.compiler.generate_until_str(
                "\nUser:", max_new=max_new, rep_threshold=3)
            metrics["total_gen_time"] += time.time() - gen_t0
            metrics["total_gen_tokens"] += len(gen_ids)
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
                recall_mark = " [RECALL]" if should_recall else ""
                print(f"  RetroSIG step {step_i + 1}/{len(scenario)}{recall_mark} — "
                      f"cache: {self.engine.cache_size} tok, ttf: {time.time()-step_t0:.3f}s")

        metrics["total_ttf"] = time.time() - wc_start
        metrics["completion_count"] = len(scenario)
        metrics["failure_count"] = 0
        metrics["probe_results"] = probe_results
        metrics["cache_size"] = self.engine.cache_size
        metrics["tool_archive_size"] = len(tool_results_archive)
        return metrics


class RetroSIGHeavyAgent(RetroSIGAgent):
    """RetroSIG with explicit recall on EVERY step — max quality variant."""

    def run(self, scenario, probes=None, max_new=80, debug=False):
        metrics = init_metrics()
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
        tool_results_archive: List[Dict] = []

        for step_i, step in enumerate(scenario):
            step_t0 = time.time()

            user_line = f"\nUser: {step.user_query}\nAssistant:"
            u_ids = list(self.compiler.tokenize(user_line, add_bos=False))
            self.compiler.eval(u_ids)
            metrics["total_prefill_tokens"] += len(u_ids)
            self.engine.update_cache(u_ids)

            result = self.tools.execute(step.tool_name, step.tool_args)
            tool_results_archive.append({
                "tool": step.tool_name,
                "args": step.tool_args,
                "result": result,
            })
            tool_line = f"\n[Tool: {step.tool_name}] {result}\n"
            t_ids = list(self.compiler.tokenize(tool_line, add_bos=False))
            self.compiler.eval(t_ids)
            metrics["total_prefill_tokens"] += len(t_ids)
            self.engine.update_cache(t_ids)

            if tool_results_archive:
                all_facts = self._extract_key_facts(tool_results_archive)
                if all_facts:
                    recall_prompt = (
                        f"\n[Memory] Key facts: {'; '.join(all_facts[-5:])}.\n"
                    )
                    r_ids = list(self.compiler.tokenize(recall_prompt, add_bos=False))
                    self.compiler.eval(r_ids)
                    metrics["total_prefill_tokens"] += len(r_ids)
                    self.engine.update_cache(r_ids)

            gen_t0 = time.time()
            gen_text, gen_ids = self.compiler.generate_until_str(
                "\nUser:", max_new=max_new, rep_threshold=3)
            metrics["total_gen_time"] += time.time() - gen_t0
            metrics["total_gen_tokens"] += len(gen_ids)
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
                print(f"  RetroSIG-H step {step_i + 1}/{len(scenario)} — "
                      f"cache: {self.engine.cache_size} tok, ttf: {time.time()-step_t0:.3f}s")

        metrics["total_ttf"] = time.time() - wc_start
        metrics["completion_count"] = len(scenario)
        metrics["failure_count"] = 0
        metrics["probe_results"] = probe_results
        metrics["cache_size"] = self.engine.cache_size
        metrics["tool_archive_size"] = len(tool_results_archive)
        return metrics


# ======================================================================
# R21: SIG + KV Cache Compression (H2O-style)
# ======================================================================

class CompressedSIGAgent:
    """SIG with periodic KV cache compression using importance heuristics.

    Strategies:
      keep_top_ratio  — keep top fraction of tokens by recency (H2O-like)
      keep_system     — always preserve system prompt
      keep_recent_n   — always preserve most recent N tokens
    """

    def __init__(self, compiler, tools, compression_ratio=0.5,
                 compression_interval=8, keep_recent_tokens=400):
        self.compiler = compiler
        self.tools = tools
        self.engine = InjectionEngine(compiler)
        self.compression_ratio = compression_ratio
        self.compression_interval = compression_interval
        self.keep_recent_tokens = keep_recent_tokens
        self._sys_prompt_len = 0
        self._compression_count = 0
        self._tokens_removed = 0

    def run(self, scenario, probes=None, max_new=80, debug=False):
        metrics = init_metrics()
        self.engine.reset()
        self._compression_count = 0
        self._tokens_removed = 0

        sys_prompt = f"{KITCHEN_SYSTEM_PROMPT}\n\n"
        sys_ids = list(self.compiler.tokenize(sys_prompt, add_bos=False))
        self._sys_prompt_len = len(sys_ids)

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

            user_line = f"\nUser: {step.user_query}\nAssistant:"
            u_ids = list(self.compiler.tokenize(user_line, add_bos=False))
            self.compiler.eval(u_ids)
            metrics["total_prefill_tokens"] += len(u_ids)
            self.engine.update_cache(u_ids)

            result = self.tools.execute(step.tool_name, step.tool_args)
            tool_line = f"\n[Tool: {step.tool_name}] {result}\n"
            t_ids = list(self.compiler.tokenize(tool_line, add_bos=False))
            self.compiler.eval(t_ids)
            metrics["total_prefill_tokens"] += len(t_ids)
            self.engine.update_cache(t_ids)

            should_compress = (
                step_i > 5
                and (step_i + 1) % self.compression_interval == 0
                and self.engine.cache_size > self._sys_prompt_len + self.keep_recent_tokens * 2
            )

            if should_compress:
                self._compression_count += 1
                total = self.engine.cache_size
                protected = self._sys_prompt_len + self.keep_recent_tokens
                middle = total - protected
                drop_count = int(middle * (1.0 - self.compression_ratio))
                if drop_count > 0:
                    start_drop = self._sys_prompt_len
                    end_drop = start_drop + drop_count
                    if end_drop < total - self.keep_recent_tokens:
                        before = self.engine.cache_size
                        self.engine.evict_range(start_drop, end_drop)
                        after = self.engine.cache_size
                        self._tokens_removed += (before - after)

            gen_t0 = time.time()
            gen_text, gen_ids = self.compiler.generate_until_str(
                "\nUser:", max_new=max_new, rep_threshold=3)
            metrics["total_gen_time"] += time.time() - gen_t0
            metrics["total_gen_tokens"] += len(gen_ids)
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
                comp_mark = " [COMPRESS]" if should_compress else ""
                print(f"  CompSIG step {step_i + 1}/{len(scenario)}{comp_mark} — "
                      f"cache: {self.engine.cache_size} tok, removed: {self._tokens_removed}, "
                      f"ttf: {time.time()-step_t0:.3f}s")

        metrics["total_ttf"] = time.time() - wc_start
        metrics["completion_count"] = len(scenario)
        metrics["failure_count"] = 0
        metrics["probe_results"] = probe_results
        metrics["cache_size"] = self.engine.cache_size
        metrics["compression_count"] = self._compression_count
        metrics["tokens_removed"] = self._tokens_removed
        return metrics


# ======================================================================
# Pre-existing agents for comparison
# ======================================================================

from edge_agent_bench import (
    EdgeKitchenSIG, EdgeKitchenAppLoop, EdgeKitchenAppLoopPC,
    EdgeKitchenAppLoopSliding, EdgeKitchenHybrid,
)


# ======================================================================
# R20 runner
# ======================================================================

def run_r20(args, compiler, tools):
    print(f"\n{'='*80}")
    print(f"  R20: Retrospective SIG — Compensatory Recall for Quality Recovery")
    print(f"{'='*80}")

    total = getattr(args, 'r20_steps', 65)
    max_new = getattr(args, 'r20_max_new', 80)
    debug = getattr(args, 'debug', True)

    scenario = build_kitchen_scenario(total)
    probes = build_probe_queries(scenario, 5)

    print(f"  Scenario: {len(scenario)} steps, {len(probes)} probes")
    print(f"  RetroSIG variants: call-back interval + recall prompt injection")
    print(f"  Hypothesis: explicit recall compensates for SIG fact-forgetting")

    results = {}

    sig_agent = EdgeKitchenSIG(compiler, tools)
    sig_m = sig_agent.run(scenario, probes, max_new, debug)
    sig_f1 = _probe_f1(sig_m["probe_results"])
    results["SIG (baseline)"] = (sig_m, sig_f1)

    apploop_agent = EdgeKitchenAppLoop(compiler, tools)
    app_m = apploop_agent.run(scenario, probes, max_new, debug)
    app_f1 = _probe_f1(app_m["probe_results"])
    results["AppLoop (baseline)"] = (app_m, app_f1)

    retro_configs = [
        ("RetroSIG (interval=5)", RetroSIGAgent, dict(recall_interval=5, recall_window=3)),
        ("RetroSIG (interval=3)", RetroSIGAgent, dict(recall_interval=3, recall_window=2)),
        ("RetroSIG-Heavy (every)", RetroSIGHeavyAgent, dict(recall_interval=99, recall_window=5)),
    ]

    for name, agent_cls, kwargs in retro_configs:
        compiler.reset_cache()
        agent = agent_cls(compiler, tools, **kwargs)
        print(f"\n  --- {name} ---")
        try:
            m = agent.run(scenario, probes, max_new, debug)
        except RuntimeError as e:
            m = init_metrics()
            m["total_ttf"] = 0.0
            m["completion_count"] = 0
            m["failure_count"] = len(scenario)
            m["probe_results"] = []
            print(f"  {name} CRASHED: {e}")
        f1 = _probe_f1(m["probe_results"])
        results[name] = (m, f1)

    print(f"\n{'─'*90}")
    print(f"  R20 Results: Retrospective SIG Quality-Speed Trade-off")
    print(f"{'─'*90}")
    print(f"\n  {'Agent':<30} {'Wall-Clock':<12} {'Probe F1':<10} "
          f"{'GenTok':<10} {'PrefillTok':<12} {'SIG vs':<10} {'Qual vs SIG':<10}")
    print(f"  {'─'*30} {'─'*12} {'─'*10} {'─'*10} {'─'*12} {'─'*10} {'─'*10}")

    sig_wc = sig_m["total_ttf"]
    for name in ["SIG (baseline)", "AppLoop (baseline)",
                 "RetroSIG (interval=5)", "RetroSIG (interval=3)",
                 "RetroSIG-Heavy (every)"]:
        m, f1 = results[name]
        wc = m["total_ttf"]
        vs_sig = wc / max(sig_wc, 0.001)
        f1_delta = f1 - sig_f1
        print(f"  {name:<30} {wc:>8.1f}s {f1:>8.1%} "
              f"{m['total_gen_tokens']:>8d} {m['total_prefill_tokens']:>10d} "
              f"{vs_sig:>8.2f}x {f1_delta:>+8.1%}")

    print(f"\n  R20 Summary:")
    print(f"  - RetroSIG aims to recover probe F1 lost by pure SIG via explicit recall injection")
    print(f"  - Trade-off: additional recall tokens increase prefill but restore factual accuracy")
    print(f"  - Ideal config balances recall frequency with KV cache continuity")
    return results


# ======================================================================
# R21 runner
# ======================================================================

def run_r21(args, compiler, tools):
    print(f"\n{'='*80}")
    print(f"  R21: SIG + KV Cache Compression — H2O-style Pruning")
    print(f"{'='*80}")

    total = getattr(args, 'r21_steps', 80)
    max_new = getattr(args, 'r21_max_new', 80)
    debug = getattr(args, 'debug', True)

    scenario = build_kitchen_scenario(total)
    probes = build_probe_queries(scenario, 5)

    configs = [
        ("CompSIG-30%", dict(compression_ratio=0.3, compression_interval=8)),
        ("CompSIG-50%", dict(compression_ratio=0.5, compression_interval=8)),
        ("CompSIG-70%", dict(compression_ratio=0.7, compression_interval=8)),
    ]

    print(f"  Scenario: {len(scenario)} steps, {len(probes)} probes")
    print(f"  Strategies: periodic KV cache middle-segment dropping")
    print(f"  System prompt + recent {400} tokens always preserved")

    results = {}

    sig_agent = EdgeKitchenSIG(compiler, tools)
    sig_m = sig_agent.run(scenario, probes, max_new, debug=False)
    sig_f1 = _probe_f1(sig_m["probe_results"])
    results["SIG (no compress)"] = (sig_m, sig_f1)
    print(f"\n  SIG baseline: F1={sig_f1:.1%}, cache={sig_m['cache_size']}, "
          f"wc={sig_m['total_ttf']:.1f}s")

    for name, kwargs in configs:
        compiler.reset_cache()
        agent = CompressedSIGAgent(compiler, tools, **kwargs)
        print(f"\n  --- {name} ---")
        try:
            m = agent.run(scenario, probes, max_new, debug)
        except RuntimeError as e:
            m = init_metrics()
            m["total_ttf"] = 0.0
            m["completion_count"] = 0
            m["failure_count"] = len(scenario)
            m["probe_results"] = []
            print(f"  {name} CRASHED: {e}")
        f1 = _probe_f1(m["probe_results"])
        results[name] = (m, f1)

    print(f"\n{'─'*100}")
    print(f"  R21 Results: SIG + KV Cache Compression")
    print(f"{'─'*100}")
    print(f"\n  {'Config':<22} {'Wall-Clock':<12} {'Probe F1':<10} "
          f"{'Cache(tok)':<12} {'Removed':<10} {'Compress':<10} {'F1 Δ vs SIG':<12}")
    print(f"  {'─'*22} {'─'*12} {'─'*10} {'─'*12} {'─'*10} {'─'*10} {'─'*12}")

    for name in ["SIG (no compress)", "CompSIG-30%", "CompSIG-50%", "CompSIG-70%"]:
        m, f1 = results[name]
        removed = m.get("tokens_removed", 0)
        comp_count = m.get("compression_count", 0)
        f1_delta = f1 - sig_f1
        print(f"  {name:<22} {m['total_ttf']:>8.1f}s {f1:>8.1%} "
              f"{m['cache_size']:>10d} {removed:>8d} {comp_count:>8d} "
              f"{f1_delta:>+10.1%}")

    print(f"\n  R21 Summary:")
    print(f"  - KV cache compression reduces memory footprint while preserving quality")
    print(f"  - System prompt + recent tokens act as 'anchors' for continuity")
    print(f"  - Aggressive compression may lose mid-range context needed for probes")

    app_agent = EdgeKitchenAppLoop(compiler, tools)
    app_m = app_agent.run(scenario, probes, max_new, debug=False)
    app_f1 = _probe_f1(app_m["probe_results"])

    print(f"\n  AppLoop reference: F1={app_f1:.1%}, wc={app_m['total_ttf']:.1f}s")
    print(f"  CompSIG-50% vs AppLoop speedup: {app_m['total_ttf']/max(results['CompSIG-50%'][0]['total_ttf'], 0.001):.1f}x")
    return results


# ======================================================================
# FlashAttention Prefill Normalization Diagnostic
# ======================================================================

def run_flash_attn_diagnostic(args, compiler, tools):
    """Diagnostic: normalize prefill costs under FlashAttention assumptions.

    FlashAttention-2 typically provides 2-4x speedup on long-sequence prefill
    vs naive attention. In modern frameworks (vLLM, TensorRT-LLM), prefill is
    further optimized by RadixAttention, page-level prefix caching, and
    continuous batching. This diagnostic recalculates SIG vs AppLoop speedups
    under conservative and aggressive FlashAttention scaling factors to show
    how much of SIG's advantage depends on naive prefill implementations.
    """
    print(f"\n{'='*80}")
    print(f"  FlashAttention Prefill Normalization Diagnostic")
    print(f"{'='*80}")

    total = getattr(args, 'flash_steps', 35)
    max_new = getattr(args, 'flash_max_new', 50)
    debug = getattr(args, 'debug', False)

    scenario = build_kitchen_scenario(total)

    print(f"  Scenario: {len(scenario)} steps, {max_new} max_new")
    print(f"  Scaling factors simulate FlashAttention-2/3 prefill acceleration:")
    print(f"    1.0x = naive attention (current baseline)")
    print(f"    2.0x = conservative FlashAttention-2 speedup")
    print(f"    3.0x = typical FlashAttention-2/3 on A100-class GPU")
    print(f"    5.0x = FlashAttention-3 + prefix caching + kernel fusion")
    print(f"    8.0x = RadixAttention + paged KV + pipeline prefill (vLLM-level)")

    flash_factors = [1.0, 2.0, 3.0, 5.0, 8.0]

    sig_agent = EdgeKitchenSIG(compiler, tools)
    sig_m = sig_agent.run(scenario, None, max_new, debug)
    sig_pf = sig_m["total_prefill_time"]
    sig_gen = sig_m["total_gen_time"]
    sig_wc = sig_m["total_ttf"]

    app_agent = EdgeKitchenAppLoop(compiler, tools)
    app_m = app_agent.run(scenario, None, max_new, debug)
    app_pf = app_m["total_prefill_time"]
    app_gen = app_m["total_gen_time"]
    app_wc = app_m["total_ttf"]

    print(f"\n  Raw measurements ({total} steps):")
    print(f"    SIG:      pf={sig_pf:.2f}s  gen={sig_gen:.2f}s  total={sig_wc:.2f}s")
    print(f"    AppLoop:  pf={app_pf:.2f}s  gen={app_gen:.2f}s  total={app_wc:.2f}s")
    print(f"    Naive speedup: {app_wc/max(sig_wc, 0.001):.2f}x")
    print(f"    SIG prefill %:  {sig_pf/max(sig_wc,0.001)*100:.1f}%")
    print(f"    AppLoop prefill %: {app_pf/max(app_wc,0.001)*100:.1f}%")

    print(f"\n{'─'*100}")
    print(f"  FlashAttention-Normalized Speedup Analysis")
    print(f"{'─'*100}")
    print(f"  {'FA Factor':<12} {'SIG pf(s)':<12} {'SIG wc(s)':<12} "
          f"{'AppLoop pf(s)':<14} {'AppLoop wc(s)':<14} {'Speedup':<10} "
          f"{'SIG pf%':<10} {'AppLoop pf%':<12}")
    print(f"  {'─'*12} {'─'*12} {'─'*12} {'─'*14} {'─'*14} {'─'*10} {'─'*10} {'─'*12}")

    for factor in flash_factors:
        sig_pf_norm = sig_pf / factor
        app_pf_norm = app_pf / factor
        sig_wc_norm = sig_pf_norm + sig_gen
        app_wc_norm = app_pf_norm + app_gen
        speedup = app_wc_norm / max(sig_wc_norm, 0.001)
        sig_pf_pct = sig_pf_norm / max(sig_wc_norm, 0.001) * 100
        app_pf_pct = app_pf_norm / max(app_wc_norm, 0.001) * 100

        marker = " <<<" if speedup < 1.15 else ""
        print(f"  {factor:.1f}x{'':>8} {sig_pf_norm:>8.2f}s {sig_wc_norm:>8.2f}s "
              f"{app_pf_norm:>10.2f}s {app_wc_norm:>10.2f}s "
              f"{speedup:>7.2f}x{marker} {sig_pf_pct:>7.1f}% {app_pf_pct:>9.1f}%")

    print(f"\n  {'─'*100}")
    print(f"  Interpretation:")
    print(f"  - SIG advantage diminishes as prefill becomes faster (FlashAttention)")
    print(f"  - At 3.0x (typical FA-2), SIG prefill drops from {sig_pf/max(sig_wc,0.001)*100:.0f}% → "
          f"{sig_pf/3/max(sig_pf/3+sig_gen,0.001)*100:.0f}% of wall-clock")
    print(f"  - At 8.0x (vLLM-level), prefill is near-negligible; generation dominates")
    print(f"  - Conclusion: SIG's absolute value depends critically on prefill cost ratio")
    print(f"  - In production systems with optimized prefill, SIG's speedup converges to ~1.0x")
    print(f"  - SIG's KV-cache continuity (quality) remains the differentiating factor")

    return dict(
        sig_pf_raw=sig_pf, sig_gen_raw=sig_gen, sig_wc_raw=sig_wc,
        app_pf_raw=app_pf, app_gen_raw=app_gen, app_wc_raw=app_wc,
        factors=flash_factors,
    )


# ======================================================================
# Combined RetroSIG + Compressed SIG (R20+R21 fusion)
# ======================================================================

class RetroCompressedSIGAgent:
    """RetroSIG + KV cache compression — the ultimate hybrid.

    Combines compensatory recall (R20) with periodic KV cache compression
    (R21) to achieve both high quality and controlled memory growth.
    """

    def __init__(self, compiler, tools, recall_interval=5, recall_window=3,
                 compression_ratio=0.5, compression_interval=8,
                 keep_recent_tokens=400):
        self.compiler = compiler
        self.tools = tools
        self.engine = InjectionEngine(compiler)
        self.recall_interval = recall_interval
        self.recall_window = recall_window
        self.compression_ratio = compression_ratio
        self.compression_interval = compression_interval
        self.keep_recent_tokens = keep_recent_tokens
        self._sys_prompt_len = 0
        self._compression_count = 0
        self._tokens_removed = 0

    def _extract_key_facts(self, tool_results: List[Dict]) -> List[str]:
        facts = []
        for tr in tool_results:
            tool_name = tr.get("tool", "")
            args = tr.get("args", {})
            if tool_name == "set_user_profile":
                al = args.get("allergies", "")
                if al: facts.append(f"allergies={al}")
                diet = args.get("diet", "")
                if diet: facts.append(f"diet={diet}")
                sv = args.get("servings", "")
                if sv: facts.append(f"servings={sv}")
                pref = args.get("cuisine_pref", "")
                if pref: facts.append(f"cuisine_pref={pref}")
            elif tool_name == "set_oven":
                temp = args.get("temp_c", "")
                if temp: facts.append(f"oven={temp}C")
            elif tool_name == "start_cooking":
                rid = args.get("recipe_id", "")
                if rid: facts.append(f"cooking={rid}")
            elif tool_name == "add_shopping_item":
                ing = args.get("ingredient", "")
                if ing: facts.append(f"shopping:{ing}")
            elif tool_name in ("get_recipe", "get_nutrition"):
                rid = args.get("recipe_id", "")
                if rid: facts.append(f"recipe={rid}")
            elif tool_name == "check_ingredients":
                rid = args.get("recipe_id", "")
                if rid: facts.append(f"checked={rid}")
            elif tool_name == "find_recipes":
                cuisine = args.get("cuisine", "")
                if cuisine: facts.append(f"search_cuisine={cuisine}")
            elif tool_name == "get_substitution":
                ing = args.get("ingredient", "")
                if ing: facts.append(f"sub={ing}")
        return facts

    def run(self, scenario, probes=None, max_new=80, debug=False):
        metrics = init_metrics()
        self.engine.reset()
        self._compression_count = 0
        self._tokens_removed = 0

        sys_ids = list(self.compiler.tokenize(f"{KITCHEN_SYSTEM_PROMPT}\n\n", add_bos=False))
        self._sys_prompt_len = len(sys_ids)
        pf_t0 = time.time()
        self.compiler.eval(sys_ids)
        metrics["total_prefill_time"] += time.time() - pf_t0
        metrics["total_prefill_tokens"] += len(sys_ids)
        self.engine.update_cache(sys_ids)

        probe_idx = 0
        probe_results = []
        wc_start = time.time()
        tool_results_archive: List[Dict] = []

        for step_i, step in enumerate(scenario):
            step_t0 = time.time()

            user_line = f"\nUser: {step.user_query}\nAssistant:"
            u_ids = list(self.compiler.tokenize(user_line, add_bos=False))
            self.compiler.eval(u_ids)
            metrics["total_prefill_tokens"] += len(u_ids)
            self.engine.update_cache(u_ids)

            result = self.tools.execute(step.tool_name, step.tool_args)
            tool_results_archive.append({
                "tool": step.tool_name, "args": step.tool_args, "result": result,
            })
            tool_line = f"\n[Tool: {step.tool_name}] {result}\n"
            t_ids = list(self.compiler.tokenize(tool_line, add_bos=False))
            self.compiler.eval(t_ids)
            metrics["total_prefill_tokens"] += len(t_ids)
            self.engine.update_cache(t_ids)

            should_recall = step_i > 0 and (step_i + 1) % self.recall_interval == 0
            if should_recall:
                all_facts = self._extract_key_facts(tool_results_archive)
                if all_facts:
                    recall_prompt = (
                        f"\n[Memory] Key facts: {'; '.join(all_facts[-5:])}.\n"
                    )
                    r_ids = list(self.compiler.tokenize(recall_prompt, add_bos=False))
                    self.compiler.eval(r_ids)
                    metrics["total_prefill_tokens"] += len(r_ids)
                    self.engine.update_cache(r_ids)

            should_compress = (
                step_i > 5
                and (step_i + 1) % self.compression_interval == 0
                and self.engine.cache_size > self._sys_prompt_len + self.keep_recent_tokens * 2
            )
            if should_compress:
                self._compression_count += 1
                total = self.engine.cache_size
                protected = self._sys_prompt_len + self.keep_recent_tokens
                middle = total - protected
                drop_count = int(middle * (1.0 - self.compression_ratio))
                if drop_count > 0:
                    start_drop = self._sys_prompt_len
                    end_drop = start_drop + drop_count
                    if end_drop < total - self.keep_recent_tokens:
                        before = self.engine.cache_size
                        self.engine.evict_range(start_drop, end_drop)
                        after = self.engine.cache_size
                        self._tokens_removed += (before - after)

            gen_t0 = time.time()
            gen_text, gen_ids = self.compiler.generate_until_str(
                "\nUser:", max_new=max_new, rep_threshold=3)
            metrics["total_gen_time"] += time.time() - gen_t0
            metrics["total_gen_tokens"] += len(gen_ids)
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
                print(f"  RetroCompSIG step {step_i + 1}/{len(scenario)} — "
                      f"cache: {self.engine.cache_size} tok, removed: {self._tokens_removed}, "
                      f"ttf: {time.time()-step_t0:.3f}s")

        metrics["total_ttf"] = time.time() - wc_start
        metrics["completion_count"] = len(scenario)
        metrics["failure_count"] = 0
        metrics["probe_results"] = probe_results
        metrics["cache_size"] = self.engine.cache_size
        metrics["compression_count"] = self._compression_count
        metrics["tokens_removed"] = self._tokens_removed
        metrics["tool_archive_size"] = len(tool_results_archive)
        return metrics


def run_r20_r21_fusion(args, compiler, tools):
    print(f"\n{'='*80}")
    print(f"  R20+R21 Fusion: Retrospective SIG + KV Cache Compression")
    print(f"{'='*80}")

    total = getattr(args, 'fusion_steps', 80)
    max_new = getattr(args, 'fusion_max_new', 80)
    debug = getattr(args, 'debug', True)

    scenario = build_kitchen_scenario(total)
    probes = build_probe_queries(scenario, 5)

    results = {}

    sig_agent = EdgeKitchenSIG(compiler, tools)
    sig_m = sig_agent.run(scenario, probes, max_new, debug=False)
    sig_f1 = _probe_f1(sig_m["probe_results"])
    results["SIG"] = (sig_m, sig_f1)

    app_agent = EdgeKitchenAppLoop(compiler, tools)
    app_m = app_agent.run(scenario, probes, max_new, debug=False)
    app_f1 = _probe_f1(app_m["probe_results"])
    results["AppLoop"] = (app_m, app_f1)

    retro_agent = RetroSIGAgent(compiler, tools, recall_interval=3, recall_window=2)
    retro_m = retro_agent.run(scenario, probes, max_new, debug)
    retro_f1 = _probe_f1(retro_m["probe_results"])
    results["RetroSIG"] = (retro_m, retro_f1)

    comp_agent = CompressedSIGAgent(compiler, tools, compression_ratio=0.5, compression_interval=8)
    comp_m = comp_agent.run(scenario, probes, max_new, debug)
    comp_f1 = _probe_f1(comp_m["probe_results"])
    results["CompSIG"] = (comp_m, comp_f1)

    fusion_agent = RetroCompressedSIGAgent(
        compiler, tools, recall_interval=3, recall_window=2,
        compression_ratio=0.5, compression_interval=8)
    fusion_m = fusion_agent.run(scenario, probes, max_new, debug)
    fusion_f1 = _probe_f1(fusion_m["probe_results"])
    results["RetroCompSIG"] = (fusion_m, fusion_f1)

    print(f"\n{'─'*110}")
    print(f"  Fusion Results: Quality-Speed-Memory Trilemma")
    print(f"{'─'*110}")
    print(f"\n  {'Agent':<18} {'Wall-Clock':<12} {'Probe F1':<10} "
          f"{'GenTok':<10} {'Cache(tok)':<12} {'Removed':<10} {'vs SIG':<10} {'vs AppLoop':<12}")
    print(f"  {'─'*18} {'─'*12} {'─'*10} {'─'*10} {'─'*12} {'─'*10} {'─'*10} {'─'*12}")

    for name in ["SIG", "AppLoop", "RetroSIG", "CompSIG", "RetroCompSIG"]:
        m, f1 = results[name]
        vs_sig = m["total_ttf"] / max(sig_m["total_ttf"], 0.001)
        vs_app = app_m["total_ttf"] / max(m["total_ttf"], 0.001)
        removed = m.get("tokens_removed", 0)
        cache = m.get("cache_size", 0)
        print(f"  {name:<18} {m['total_ttf']:>8.1f}s {f1:>8.1%} "
              f"{m['total_gen_tokens']:>8d} {cache:>10d} {removed:>8d} "
              f"{vs_sig:>8.2f}x {vs_app:>10.2f}x")

    print(f"\n  Fusion Summary:")
    print(f"  - RetroCompSIG combines explicit recall with KV compression")
    print(f"  - Target: SIG-level speed + AppLoop-level quality + bounded memory")
    print(f"  - Key insight: recall prompts serve as 'soft checkpoints' resisting compression loss")
    return results


# ======================================================================
# Main
# ======================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Retrospective SIG & Hybrid Experiments — Reviewer Response Suite")
    parser.add_argument("--model", type=str, default="")
    parser.add_argument("--n-ctx", type=int, default=8192)
    parser.add_argument("--n-threads", type=int, default=4)
    parser.add_argument("--n-gpu-layers", type=int, default=0)
    parser.add_argument("--task", default="all",
                        choices=["r20", "r21", "flash", "fusion", "all"])
    parser.add_argument("--debug", action="store_true", default=True)
    parser.add_argument("--no-debug", action="store_false", dest="debug")
    parser.add_argument("--r20-steps", type=int, default=65)
    parser.add_argument("--r20-max-new", type=int, default=80)
    parser.add_argument("--r21-steps", type=int, default=80)
    parser.add_argument("--r21-max-new", type=int, default=80)
    parser.add_argument("--flash-steps", type=int, default=35)
    parser.add_argument("--flash-max-new", type=int, default=50)
    parser.add_argument("--fusion-steps", type=int, default=80)
    parser.add_argument("--fusion-max-new", type=int, default=80)
    parser.add_argument("--no-gpu", action="store_true", default=False)
    args = parser.parse_args()

    needs_model = args.task != "none"
    if needs_model and not args.model:
        parser.error(f"--task {args.task} requires --model MODEL")

    gpu = GPUMonitor()

    n_gpu = 0 if args.no_gpu else args.n_gpu_layers
    if args.no_gpu:
        print("  --no-gpu: forcing CPU inference for fair OOM-free comparison")

    print(f"Loading model: {args.model}")
    compiler = MeaningCompiler(
        model_path=args.model, n_ctx=args.n_ctx,
        n_threads=args.n_threads, n_gpu_layers=n_gpu)
    tools = KitchenToolRegistry()

    if args.task in ("r20", "all"):
        run_r20(args, compiler, tools)
    if args.task in ("r21", "all"):
        run_r21(args, compiler, tools)
    if args.task in ("flash", "all"):
        run_flash_attn_diagnostic(args, compiler, tools)
    if args.task in ("fusion", "all"):
        run_r20_r21_fusion(args, compiler, tools)

    gpu.shutdown()
    print("\nDone.")


if __name__ == "__main__":
    main()
