#!/usr/bin/env python3
"""
Cross-Architecture CO/SIG Benchmark
=====================================
Reproduces core CO/SIG experiments on non-Qwen model families to
address cross-architecture generalisation concerns.

Supported models (GGUF):
  - Nemotron-3-Nano-4B (NVIDIA, hybrid Mamba+attention)
  - Gemma-4-E2B-2B      (Google, latest Gemma architecture)
  - Llama-3.2            (TheBloke/Llama-3.2-3B-Instruct-GGUF)
  - Phi-3.5              (microsoft/Phi-3.5-mini-instruct-GGUF)

Tasks:
  co_baseline   : CO teacher-precomputed benchmark (scenarios 1-9)
  r13_batch     : R13 fragmented assembly + batch injection
  quality       : Kitchen task completion quality evaluation
  all           : run all tasks

Usage:
  python cross_arch_bench.py --model models/nvidia_Nemotron-3-Nano-4B-Q4_K_M.gguf --task all
  python cross_arch_bench.py --model models/gemma-4-E2B-it-Q4_K_M.gguf --task co_baseline
  python cross_arch_bench.py --model models/Llama-3.2-3B-Q4_K_M.gguf --task r13_batch
"""

import time, json, argparse, os, sys, random, re, statistics, math
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field

from core import (
    MeaningCompiler, InjectionEngine, ToolRegistry, GPUMonitor,
    init_metrics, compute_metrics_table, mean_std, SYSTEM_PROMPT,
    KitchenQualityEvaluator, build_kitchen_ground_truth,
)
from core.compiler import PrefixCache, SEQ_ID
from core.tools import LatencyToolWrapper

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False


MODEL_REGISTRY = {
    "nemotron": {
        "description": "NVIDIA Nemotron-3-Nano-4B (hybrid Mamba+attention, 4B)",
        "hf_repo": "nvidia/Nemotron-3-Nano-4B-Instruct-GGUF",
        "filename": "nvidia_Nemotron-3-Nano-4B-Q4_K_M.gguf",
        "arch_features": ["Hybrid Mamba+attention blocks", "~3.2B Mamba + ~0.8B attention params",
                          "state-space recurrence + selective KV-cache"],
        "expected_kv_behavior": "MOST CRITICAL CROSS-ARCH VALIDATION. Hybrid SSM+attention "
                                "fundamentally different from Qwen's pure dense attention. "
                                "Mamba layers compress state into fixed-size hidden state — "
                                "SIG injection fidelity may differ sharply from Transformer-only "
                                "models. Attention layers within the hybrid should preserve "
                                "SIG's prefill savings, but the overall speedup profile may "
                                "diverge significantly from Qwen baselines.",
        "sourcescope_repo": "nvidia/Nemotron-3-Nano-4B-Instruct",
    },
    "gemma4": {
        "description": "Google Gemma 4 E2B-IT (latest Gemma architecture, 2B)",
        "hf_repo": "google/gemma-4-2b-it-GGUF",
        "filename": "gemma-4-E2B-it-Q4_K_M.gguf",
        "arch_features": ["GeGLU activation", "pre-normalization", "likely GQA",
                          "latest Gemma generation — distinct from Gemma 2"],
        "expected_kv_behavior": "Gemma 4's pre-normalization and architectural updates "
                                "may produce a different layer-sensitivity profile than Qwen. "
                                "The 2B scale sits near the 1.5-2B crossover identified in "
                                "our framework — results here will refine the deployment boundary. "
                                "GQA head structure may reduce per-head injection fidelity "
                                "compared to Qwen's dense attention.",
        "sourcescope_repo": "google/gemma-4-2b-it",
    },
    "llama": {
        "description": "Llama-3.2 series (GQA, RoPE, SwiGLU)",
        "hf_repo": "TheBloke/Llama-3.2-3B-Instruct-GGUF",
        "filename": "llama-3.2-3b-instruct-q4_k_m.gguf",
        "arch_features": ["GQA (grouped-query attention)", "RoPE position encoding"],
        "expected_kv_behavior": "GQA may reduce per-head injection fidelity but "
                                "increase inference efficiency; group-level attention "
                                "should preserve SIG's prefill savings",
    },
    "gemma": {
        "description": "Gemma 2 series (GeGLU, pre-norm)",
        "hf_repo": "bartowski/gemma-2-2b-it-GGUF",
        "filename": "gemma-2-2b-it-Q4_K_M.gguf",
        "arch_features": ["GeGLU activation", "pre-normalization"],
        "expected_kv_behavior": "Pre-normalisation may shift layer-sensitivity "
                                "profile; R1 attention gradient may differ from Qwen",
    },
    "phi": {
        "description": "Phi-3.5-mini (dense, 3.8B)",
        "hf_repo": "microsoft/Phi-3.5-mini-instruct-GGUF",
        "filename": "Phi-3.5-mini-instruct-Q4_K_M.gguf",
        "arch_features": ["Dense attention (no GQA)", "unique tokenizer"],
        "expected_kv_behavior": "Dense attention most similar to Qwen; expect "
                                "closest replication of core SIG results",
    },
}


def detect_architecture(model_path: str) -> str:
    name = os.path.basename(model_path).lower()
    if "nemotron" in name:
        return "nemotron"
    if "gemma-4" in name or "gemma4" in name:
        return "gemma4"
    if "gemma" in name:
        return "gemma"
    if "llama" in name:
        return "llama"
    if "phi" in name:
        return "phi"
    if "qwen" in name:
        return "qwen"
    return "unknown"


def run_co_baseline(args, compiler, module, arch_name):
    """Run simplified CO baseline: tool chains of varying depth (4, 8, 12 tools)."""
    print(f"\n{'='*80}")
    print(f"  Cross-Arch CO Baseline: {arch_name}")
    print(f"  Model: {args.model}")
    print(f"{'='*80}")

    n_runs = getattr(args, 'co_runs', 3)
    sys_prompt = SYSTEM_PROMPT + "\n\n"
    city_cycle = ["paris", "rome", "berlin", "tokyo", "london", "sydney"]
    tool_cycle = ["search_attractions", "get_weather", "get_flight_info", "search_attractions"]

    scenarios = {
        "shallow_4": _build_tool_chain(4, city_cycle, tool_cycle),
        "medium_8": _build_tool_chain(8, city_cycle, tool_cycle),
        "deep_12": _build_tool_chain(12, city_cycle, tool_cycle),
    }

    results = {}
    for s_name, chain in scenarios.items():
        sig_times = []
        app_times = []

        for run_i in range(n_runs):
            random.seed(42 + run_i)
            compiler.reset_cache()
            t0 = time.time()
            _run_apploop_chain(compiler, module, sys_prompt, chain)
            app_times.append(time.time() - t0)

            compiler.reset_cache()
            t0 = time.time()
            _run_sig_chain(compiler, module, sys_prompt, chain)
            sig_times.append(time.time() - t0)

        sig_m, sig_s = mean_std(sig_times)
        app_m, app_s = mean_std(app_times)
        speedup = app_m / max(sig_m, 0.001)
        results[s_name] = {
            "sig_mean": sig_m, "sig_std": sig_s,
            "app_mean": app_m, "app_std": app_s,
            "speedup": speedup, "n_tools": len(chain),
        }
        print(f"  {s_name:<12} ({len(chain)} tools)  SIG={sig_m:.3f}s  "
              f"AppLoop={app_m:.3f}s  Speedup={speedup:.2f}x")

    speeds = [r["speedup"] for r in results.values()]
    avg_speedup = sum(speeds) / max(len(speeds), 1)
    print(f"\n  {arch_name} Summary: avg speedup={avg_speedup:.2f}x "
          f"across {len(scenarios)} chain depths")
    return results


def _build_tool_chain(n, city_cycle, tool_cycle):
    chain = []
    for i in range(n):
        city = city_cycle[i % len(city_cycle)]
        tool_name = tool_cycle[i % len(tool_cycle)]
        if tool_name == "get_flight_info":
            dest = city_cycle[(i + 1) % len(city_cycle)]
            chain.append((tool_name, {"origin": city, "destination": dest}))
        else:
            chain.append((tool_name, {"city": city}))
    return chain


def _run_apploop_chain(compiler, module, sys_prompt, chain):
    prefix = list(compiler.tokenize(sys_prompt, add_bos=False))
    compiler.rebuild_cache(prefix)
    for tool_name, tool_args in chain:
        result = module.execute(tool_name, tool_args)
        result_text = f"\n[Tool: {tool_name}] {result}\nAssistant:"
        r_ids = list(compiler.tokenize(result_text, add_bos=False))
        compiler.eval(r_ids)
        compiler.generate_until_str("\n", max_new=20, rep_threshold=3)


def _run_sig_chain(compiler, module, sys_prompt, chain):
    engine = InjectionEngine(compiler)
    engine.reset()
    sys_ids = list(compiler.tokenize(sys_prompt, add_bos=False))
    engine.inject(sys_ids)
    for tool_name, tool_args in chain:
        result = module.execute(tool_name, tool_args)
        result_text = f"\n[Tool: {tool_name}] {result}\nAssistant:"
        r_ids = list(compiler.tokenize(result_text, add_bos=False))
        engine.inject(r_ids)
        compiler.generate_until_str("\n", max_new=20, rep_threshold=3)


def run_r13_batch(args, compiler, module, arch_name):
    print(f"\n{'='*80}")
    print(f"  Cross-Arch R13 Batch Injection: {arch_name}")
    print(f"  Model: {args.model}")
    print(f"{'='*80}")

    n_runs = getattr(args, 'r13_runs', 5)
    batch_sizes = [2, 4, 8]
    num_devices = 4

    expected_chain = [
        {"tool": "search_attractions", "tool_args": {"city": "paris"}},
        {"tool": "get_weather", "tool_args": {"city": "paris"}},
        {"tool": "search_attractions", "tool_args": {"city": "rome"}},
        {"tool": "get_weather", "tool_args": {"city": "rome"}},
        {"tool": "search_attractions", "tool_args": {"city": "berlin"}},
        {"tool": "get_weather", "tool_args": {"city": "berlin"}},
        {"tool": "get_flight_info", "tool_args": {"origin": "paris", "destination": "rome"}},
        {"tool": "get_flight_info", "tool_args": {"origin": "rome", "destination": "berlin"}},
    ]
    total_steps = len(expected_chain)
    sys_ids = list(compiler.tokenize(f"{SYSTEM_PROMPT}\n\n", add_bos=False))

    base_seed = 42
    per_step_times = []
    for run_i in range(n_runs):
        random.seed(base_seed + run_i)
        all_results = [module.execute(e["tool"], e["tool_args"]) for e in expected_chain]
        compiler.reset_cache()
        compiler.eval(sys_ids)
        wc_start = time.time()
        for step_i, entry in enumerate(expected_chain):
            result = all_results[step_i]
            device_id = step_i % num_devices
            result_text = (f"\n[Device {device_id}] {entry['tool']}"
                           f"({list(entry['tool_args'].values())[0]}): {result}\n")
            step_ids = list(compiler.tokenize(result_text, add_bos=False))
            gen_prompt = list(compiler.tokenize(
                f"\nBased on steps 1-{step_i+1}, provide a one-line summary.\n",
                add_bos=False))
            compiler.eval(step_ids)
            compiler.eval(gen_prompt)
            compiler.generate_until_str("\n", max_new=30, rep_threshold=3)
        per_step_times.append(time.time() - wc_start)

    ps_m, ps_s = mean_std(per_step_times)
    print(f"\n  {'Mode':<22} {'Mean(s)':<14} {'vs Per-Step':<12} {'Gen Calls':<12}")
    print(f"  {'-'*22} {'-'*14} {'-'*12} {'-'*12}")
    print(f"  {'Per-Step SIG':<22} {ps_m:.3f}±{ps_s:.3f}      1.00x         {total_steps:<12}")

    batch_results = {}
    for batch_size in batch_sizes:
        batch_wall_times = []
        gen_call_counts = []
        for run_i in range(n_runs):
            random.seed(base_seed + run_i)
            all_results = [module.execute(e["tool"], e["tool_args"]) for e in expected_chain]
            compiler.reset_cache()
            compiler.eval(sys_ids)
            wc_start = time.time()
            gen_calls = 0
            step_i = 0
            while step_i < total_steps:
                chunk_size = min(batch_size, total_steps - step_i)
                chunk_steps = expected_chain[step_i:step_i + chunk_size]
                chunk_results = all_results[step_i:step_i + chunk_size]
                for j, entry in enumerate(chunk_steps):
                    result = chunk_results[j]
                    device_id = (step_i + j) % num_devices
                    result_text = (f"\n[Device {device_id}] {entry['tool']}"
                                   f"({list(entry['tool_args'].values())[0]}): {result}\n")
                    step_ids = list(compiler.tokenize(result_text, add_bos=False))
                    compiler.eval(step_ids)
                gen_prompt = list(compiler.tokenize(
                    f"\nBased on steps 1-{step_i + chunk_size}, provide a one-line summary.\n",
                    add_bos=False))
                compiler.eval(gen_prompt)
                compiler.generate_until_str("\n", max_new=30, rep_threshold=3)
                gen_calls += 1
                step_i += chunk_size
            batch_wall_times.append(time.time() - wc_start)
            gen_call_counts.append(gen_calls)
        bm, bs_err = mean_std(batch_wall_times)
        avg_gen_calls = sum(gen_call_counts) / max(len(gen_call_counts), 1)
        speedup_vs_perstep = ps_m / max(bm, 0.001)
        batch_results[batch_size] = {
            "mean": bm, "std": bs_err,
            "speedup_vs_perstep": speedup_vs_perstep,
            "gen_calls": avg_gen_calls,
        }
        print(f"  {'Batch-SIG (bs=' + str(batch_size) + ')':<22} {bm:.3f}±{bs_err:.3f}      "
              f"{speedup_vs_perstep:.2f}x         {avg_gen_calls:.0f}{'':<12}")

    apppc_times = []
    for run_i in range(n_runs):
        random.seed(base_seed + run_i)
        all_results = [module.execute(e["tool"], e["tool_args"]) for e in expected_chain]
        pc = PrefixCache()
        compiler.reset_cache()
        compiler.eval(sys_ids)
        pc.save(compiler, list(sys_ids))
        wc_start = time.time()
        for step_i in range(total_steps):
            restored = pc.restore(compiler)
            compass_body = "\n\n".join(
                f"[Device {i % num_devices}] {expected_chain[i]['tool']}"
                f"({list(expected_chain[i]['tool_args'].values())[0]}): {all_results[i]}"
                for i in range(step_i + 1))
            ctx_ids = list(compiler.tokenize("\n\n" + compass_body, add_bos=False))
            gen_prompt_ids = list(compiler.tokenize(
                f"\nBased on steps 1-{step_i+1}, provide a one-line summary.\n",
                add_bos=False))
            compiler.eval(ctx_ids)
            compiler.eval(gen_prompt_ids)
            compiler.generate_until_str("\n", max_new=30, rep_threshold=3)
        apppc_times.append(time.time() - wc_start)
    apm, aps = mean_std(apppc_times)
    print(f"  {'AppLoop-PC':<22} {apm:.3f}±{aps:.3f}      "
          f"{ps_m / max(apm, 0.001):.2f}x         {total_steps:<12}")

    print(f"\n  {arch_name} R13 Batch Summary:")
    print(f"  Per-step SIG: {ps_m:.3f}s  AppLoop-PC: {apm:.3f}s")
    print(f"  Batch-SIG(bs=8)/AppLoop-PC ratio: {apm / max(batch_results.get(8, {}).get('mean', 0.001), 0.001):.2f}x")

    return {
        "per_step": {"mean": ps_m, "std": ps_s},
        "apploop_pc": {"mean": apm, "std": aps},
        "batch": batch_results,
    }


def run_quality(args, compiler, module, arch_name):
    print(f"\n{'='*80}")
    print(f"  Cross-Arch Kitchen Quality: {arch_name}")
    print(f"  Model: {args.model}")
    print(f"{'='*80}")

    n_runs = getattr(args, 'quality_runs', 5)
    total_steps = getattr(args, 'quality_kitchen_steps', 30)

    recipe_list = ["spaghetti_bolognese", "chicken_stir_fry", "caprese_salad",
                   "omelette", "mushroom_risotto"]
    recipes_used = set(recipe_list)

    steps = [{"user": f"Set profile: 2 people, dairy allergy, prefer Italian.",
              "tool": "set_user_profile",
              "tool_args": {"allergies": "dairy", "diet": "omnivore", "servings": 2,
                            "cuisine_pref": "italian"}}]
    for ing in ["pasta", "rice", "olive_oil", "salt", "pepper", "garlic", "onion", "tomato"]:
        steps.append({"user": f"Stock pantry: {ing}.", "tool": "add_to_pantry",
                       "tool_args": {"ingredient": ing, "amount_g": 500}})
    for ing in ["chicken_breast", "eggs", "butter"]:
        steps.append({"user": f"Stock fridge: {ing}.", "tool": "add_to_fridge",
                       "tool_args": {"ingredient": ing, "amount_g": 400}})
    for rid in recipe_list:
        steps.append({"user": f"Get recipe for {rid}.", "tool": "get_recipe",
                       "tool_args": {"recipe_id": rid}})
    for i in range(5):
        rid = recipe_list[i % len(recipe_list)]
        steps.append({"user": f"Check ingredients for {rid}.",
                       "tool": "check_ingredients", "tool_args": {"recipe_id": rid}})
    for ing in ["tomato", "garlic", "basil", "oregano"]:
        steps.append({"user": f"Add {ing} to shopping list.",
                       "tool": "add_shopping_item",
                       "tool_args": {"ingredient": ing, "quantity": 2}})
    steps.append({"user": "Show shopping list.", "tool": "get_shopping_list",
                   "tool_args": {}})
    steps.append({"user": "Preheat oven to 180C.", "tool": "set_oven",
                   "tool_args": {"temp_c": 180, "on": True}})

    final_prompt = ("Summarize everything: what recipes we have, our shopping list, "
                    "and any dietary restrictions.")
    steps.append({"user": final_prompt, "tool": None, "tool_args": None})

    while len(steps) < total_steps:
        rid = recipe_list[len(steps) % len(recipe_list)]
        steps.append({"user": f"Quick check: nutrition for {rid}?",
                       "tool": "get_nutrition", "tool_args": {"recipe_id": rid}})

    gt = {
        "expected_recipes": sorted(recipes_used),
        "shopping_items": ["basil", "garlic", "oregano", "tomato"],
        "inventory_items": ["pasta", "rice", "olive_oil", "salt", "pepper", "garlic",
                             "onion", "tomato", "chicken_breast", "eggs", "butter"],
        "allergens": ["dairy"],
    }

    evaluator = KitchenQualityEvaluator(gt)
    sig_qualities = []
    app_qualities = []
    sig_times = []
    app_times = []

    for run_i in range(n_runs):
        random.seed(42 + run_i)

        compiler.reset_cache()
        wc = time.time()
        sig_output, sig_log, sig_results = _run_kitchen_sig(compiler, module, steps)
        sig_times.append(time.time() - wc)
        sig_q = evaluator.evaluate(sig_output, sig_log, sig_results)
        sig_qualities.append(sig_q)

        compiler.reset_cache()
        wc = time.time()
        app_output, app_log, app_results = _run_kitchen_apploop(compiler, module, steps)
        app_times.append(time.time() - wc)
        app_q = evaluator.evaluate(app_output, app_log, app_results)
        app_qualities.append(app_q)

    sig_tm, sig_ts = mean_std(sig_times)
    app_tm, app_ts = mean_std(app_times)
    speedup = app_tm / max(sig_tm, 0.001)

    sig_cm, sig_cs = mean_std([q["composite"] for q in sig_qualities])
    app_cm, app_cs = mean_std([q["composite"] for q in app_qualities])
    quality_delta = sig_cm - app_cm

    print(f"\n  {arch_name} Kitchen Quality Results (n={n_runs}):")
    print(f"  {'Metric':<25} {'SIG':<18} {'AppLoop':<18} {'Delta':<12}")
    print(f"  {'-'*25} {'-'*18} {'-'*18} {'-'*12}")
    print(f"  {'Wall-Clock (s)':<25} {sig_tm:.3f}±{sig_ts:.3f}      "
          f"{app_tm:.3f}±{app_ts:.3f}      {speedup:.2f}x")
    print(f"  {'Composite Quality':<25} {sig_cm:.3f}±{sig_cs:.3f}        "
          f"{app_cm:.3f}±{app_cs:.3f}        {quality_delta:+.3f}")

    recipe_sig = sum(q.get("recipe_mentioned", 0) for q in sig_qualities) / n_runs
    recipe_app = sum(q.get("recipe_mentioned", 0) for q in app_qualities) / n_runs
    print(f"  {'Recipe Mention':<25} {recipe_sig:<18.3f} {recipe_app:<18.3f} "
          f"{recipe_sig - recipe_app:+.3f}")
    print(f"\n  Key finding for {arch_name}: speedup={speedup:.2f}x, "
          f"quality_delta={quality_delta:+.3f}")

    return {
        "sig_wc": (sig_tm, sig_ts),
        "app_wc": (app_tm, app_ts),
        "speedup": speedup,
        "sig_quality": (sig_cm, sig_cs),
        "app_quality": (app_cm, app_cs),
        "quality_delta": quality_delta,
        "recipe_sig": recipe_sig,
        "recipe_app": recipe_app,
    }


def _run_kitchen_sig(compiler, module, steps):
    engine = InjectionEngine(compiler)
    engine.reset()
    sys_prompt = ("You are a kitchen assistant running on an edge device. "
                  "Be concise but complete. Answer all parts of the request.\n\n")
    sys_ids = list(compiler.tokenize(sys_prompt, add_bos=False))
    engine.inject(sys_ids)
    tool_log = []
    tool_results_list = []

    for step in steps:
        user_line = f"\nUser: {step['user']}\nAssistant:"
        u_ids = list(compiler.tokenize(user_line, add_bos=False))
        engine.inject(u_ids)
        if step.get("tool"):
            result = module.execute(step["tool"], step["tool_args"])
            tool_line = f"\n[Tool: {step['tool']}] {result}\n"
            t_ids = list(compiler.tokenize(tool_line, add_bos=False))
            engine.inject(t_ids)
            tool_log.append({"tool": step["tool"], "args": step["tool_args"],
                             "result": result})
            tool_results_list.append(result)

    output_ids = compiler.generate_until_str("\nUser:", max_new=200, rep_threshold=3)
    return output_ids[0], tool_log, tool_results_list


def _run_kitchen_apploop(compiler, module, steps):
    sys_prompt = ("You are a kitchen assistant running on an edge device. "
                  "Be concise but complete. Answer all parts of the request.\n\n")
    context = sys_prompt
    tool_log = []
    tool_results_list = []

    for step in steps:
        context += f"\nUser: {step['user']}\nAssistant:"
        if step.get("tool"):
            result = module.execute(step["tool"], step["tool_args"])
            context += f"\n[Tool: {step['tool']}] {result}\n"
            tool_log.append({"tool": step["tool"], "args": step["tool_args"],
                             "result": result})
            tool_results_list.append(result)

    ids = list(compiler.tokenize(context, add_bos=False))
    compiler.rebuild_cache(ids)
    output_ids = compiler.generate_until_str("\nUser:", max_new=200, rep_threshold=3)
    return output_ids[0], tool_log, tool_results_list


def run_dependency_check(compiler, module, arch_name):
    print(f"\n  --- Sequential Dependency Check ({arch_name}) ---")
    context = "The assistant has access to tool results:\n"
    for step_i in range(4):
        result = module.execute("get_weather", {"city": "paris"})
        context += f"  Step {step_i+1}: weather={result}\n"
        ids = list(compiler.tokenize(context, add_bos=False))
        compiler.rebuild_cache(ids)
        gen = compiler.generate_until_str("\n", max_new=8, rep_threshold=3)
        out = gen[0].strip()
        print(f"  Step {step_i+1} output: '{out[:60]}'")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Cross-Architecture CO/SIG Benchmark")
    parser.add_argument("--model", type=str, required=True,
                        help="Path to GGUF model file")
    parser.add_argument("--task", default="all",
                        choices=["co_baseline", "r13_batch", "quality",
                                 "dependency_check", "all"])
    parser.add_argument("--n-ctx", type=int, default=8192)
    parser.add_argument("--n-threads", type=int, default=4)
    parser.add_argument("--n-gpu-layers", type=int, default=0)
    parser.add_argument("--co-runs", type=int, default=3,
                        help="Number of runs for CO baseline")
    parser.add_argument("--r13-runs", type=int, default=5,
                        help="Number of runs for R13 batch")
    parser.add_argument("--quality-runs", type=int, default=5,
                        help="Number of runs for quality eval")
    parser.add_argument("--quality-kitchen-steps", type=int, default=30,
                        help="Number of kitchen steps")
    parser.add_argument("--output", type=str, default="",
                        help="Output JSON file path")
    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"ERROR: Model not found: {args.model}")
        print(f"\n  Download GGUF models from HuggingFace:")
        for key, info in MODEL_REGISTRY.items():
            print(f"  {key:<8} {info['hf_repo']} — {info['filename']}")
        sys.exit(1)

    arch_name = detect_architecture(args.model)
    arch_info = MODEL_REGISTRY.get(arch_name, {"description": "Unknown architecture",
                                                "arch_features": ["Unknown"]})
    print(f"\n{'='*80}")
    print(f"  Cross-Architecture CO/SIG Benchmark")
    print(f"  Model: {args.model}")
    print(f"  Detected Architecture: {arch_name} — {arch_info['description']}")
    print(f"  Features: {', '.join(arch_info['arch_features'])}")
    print(f"  Expected KV Behavior: {arch_info.get('expected_kv_behavior', 'N/A')}")
    print(f"{'='*80}")

    print(f"  Loading model... (ctx={args.n_ctx}, gpu_layers={args.n_gpu_layers})")
    compiler = MeaningCompiler(
        model_path=args.model,
        n_ctx=args.n_ctx,
        n_threads=args.n_threads,
        n_gpu_layers=args.n_gpu_layers,
    )
    module = ToolRegistry()
    gpu = GPUMonitor()

    all_results = {"model": args.model, "architecture": arch_name,
                   "arch_features": arch_info["arch_features"],
                   "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}

    if args.task in ("co_baseline", "all"):
        all_results["co_baseline"] = run_co_baseline(args, compiler, module, arch_name)

    if args.task in ("r13_batch", "all"):
        all_results["r13_batch"] = run_r13_batch(args, compiler, module, arch_name)

    if args.task in ("quality", "all"):
        all_results["quality"] = run_quality(args, compiler, module, arch_name)

    if args.task in ("dependency_check", "all"):
        run_dependency_check(compiler, module, arch_name)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"\n  Results saved to {args.output}")

    print(f"\n{'='*80}")
    print(f"  Cross-Architecture Benchmark Complete")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
