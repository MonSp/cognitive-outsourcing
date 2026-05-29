#!/usr/bin/env python3
"""EXP-4/5/6: Scheduler ablation, crossover shift, drafter selection.

Uses ManualSpecDecCompiler (eval+sample path) which works on Qwen3.5.

Usage:
  python run_exp456.py --model models/Qwen3.5-4B-Q4_K_M.gguf --exp all
  python run_exp456.py --model models/Qwen3.5-0.8B-Q4_K_M.gguf --exp exp5
"""

import argparse
import random
import sys
import os
import time
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple

sys.path.insert(0, '.')

from core import MeaningCompiler, InjectionEngine, GPUMonitor, init_metrics
from core.llamacpp_specdec import ManualSpecDecCompiler
from edge_agent_bench import (
    KITCHEN_SYSTEM_PROMPT, KitchenToolRegistry, KitchenStep,
    build_kitchen_scenario,
)


def run_sig_baseline(model_path, scenario, tools, n_gpu_layers=99, max_new=60):
    """SIG baseline (no SpecDec)."""
    compiler = MeaningCompiler(model_path, n_ctx=16384, n_gpu_layers=n_gpu_layers)
    engine = InjectionEngine(compiler)
    engine.reset()

    metrics = init_metrics()
    sys_ids = list(compiler.tokenize("%s\n\n" % KITCHEN_SYSTEM_PROMPT, add_bos=False))
    compiler.eval(sys_ids)
    engine.update_cache(sys_ids)

    wc_start = time.time()
    completed = 0

    for step in scenario:
        step_t0 = time.time()
        user_line = "\nUser: %s\nAssistant:" % step.user_query
        u_ids = list(compiler.tokenize(user_line, add_bos=False))
        compiler.eval(u_ids)
        engine.update_cache(u_ids)

        result = tools.execute(step.tool_name, step.tool_args)
        tool_line = "\n[Tool: %s] %s\n" % (step.tool_name, result)
        t_ids = list(compiler.tokenize(tool_line, add_bos=False))
        compiler.eval(t_ids)
        engine.update_cache(t_ids)

        gen_t0 = time.time()
        gen_text, gen_ids = compiler.generate_until_str(
            "\nUser:", max_new=max_new, rep_threshold=3)
        metrics["total_gen_time"] += time.time() - gen_t0
        metrics["total_gen_tokens"] += len(gen_ids)
        engine.update_cache(list(gen_ids))
        completed += 1
        metrics["per_turn_ttf"].append(time.time() - step_t0)

    metrics["total_ttf"] = time.time() - wc_start
    metrics["completion_count"] = completed
    del compiler
    return metrics


def run_specdec_with_schedule(
    model_path, scenario, tools, n_gpu_layers=99, max_new=60,
    warmup_steps=3, max_k=3, ngram_size=3,
    track_acceptance=False,
):
    """SIG + SpecDec with adaptive K scheduling.

    Returns: (metrics, acceptance_records)
    """
    compiler = ManualSpecDecCompiler(
        model_path, n_ctx=16384, n_gpu_layers=n_gpu_layers,
        ngram_size=ngram_size, num_pred=max_k,
    )

    metrics = init_metrics()
    acceptance_records = []
    steps_since_injection = 0
    current_k = 1

    sys_ids = compiler.tokenize("%s\n\n" % KITCHEN_SYSTEM_PROMPT)
    compiler.reset_cache()
    compiler.eval(sys_ids)

    wc_start = time.time()
    completed = 0
    total_draft_proposed = 0
    total_draft_accepted = 0

    for step_i, step in enumerate(scenario):
        step_t0 = time.time()
        steps_since_injection = 0

        user_line = "\nUser: %s\nAssistant:" % step.user_query
        u_ids = compiler.tokenize(user_line)
        compiler.eval(u_ids)

        result = tools.execute(step.tool_name, step.tool_args)
        tool_line = "\n[Tool: %s] %s\n" % (step.tool_name, result)
        t_ids = compiler.tokenize(tool_line)
        compiler.eval(t_ids)

        # Adaptive K scheduling
        if steps_since_injection >= warmup_steps and current_k < max_k:
            current_k = min(current_k + 1, max_k)

        gen_t0 = time.time()
        gen_text, gen_ids, stats = compiler.generate_with_specdec(
            stop_str="\nUser:", max_new=max_new, num_pred=current_k)
        gen_time = time.time() - gen_t0

        metrics["total_gen_time"] += gen_time
        metrics["total_gen_tokens"] += len(gen_ids)
        total_draft_proposed += stats["draft_proposed"]
        total_draft_accepted += stats["draft_accepted"]
        completed += 1
        metrics["per_turn_ttf"].append(time.time() - step_t0)

        if track_acceptance and stats["draft_proposed"] > 0:
            acceptance_records.append({
                "step": step_i,
                "k": current_k,
                "proposed": stats["draft_proposed"],
                "accepted": stats["draft_accepted"],
                "rate": stats["draft_accepted"] / stats["draft_proposed"],
            })

        steps_since_injection += 1

    metrics["total_ttf"] = time.time() - wc_start
    metrics["completion_count"] = completed
    metrics["draft_proposed"] = total_draft_proposed
    metrics["draft_accepted"] = total_draft_accepted
    metrics["acceptance_rate"] = (
        total_draft_accepted / total_draft_proposed
        if total_draft_proposed > 0 else 0.0
    )
    metrics["warmup_steps"] = warmup_steps
    metrics["max_k"] = max_k

    del compiler
    return metrics, acceptance_records


def run_apploop(model_path, scenario, tools, n_gpu_layers=99, max_new=60):
    """AppLoop baseline."""
    compiler = MeaningCompiler(model_path, n_ctx=16384, n_gpu_layers=n_gpu_layers)
    metrics = init_metrics()
    context = "%s\n\n" % KITCHEN_SYSTEM_PROMPT
    completed = 0
    wc_start = time.time()

    for step in scenario:
        step_t0 = time.time()
        context += "\nUser: %s\n" % step.user_query
        result = tools.execute(step.tool_name, step.tool_args)
        context += "[Tool: %s] %s\nAssistant:" % (step.tool_name, result)

        full_ids = list(compiler.tokenize(context, add_bos=False))
        compiler.reset_cache()
        try:
            compiler.eval(full_ids)
            gen_t0 = time.time()
            gen_text, gen_ids = compiler.generate_until_str(
                "\nUser:", max_new=max_new, rep_threshold=3)
            metrics["total_gen_time"] += time.time() - gen_t0
            metrics["total_gen_tokens"] += len(gen_ids)
            metrics["total_prefill_tokens"] += len(full_ids)
            context += gen_text + "\n"
            completed += 1
        except RuntimeError as e:
            if "failed to find a memory slot" in str(e) or "decode returned" in str(e):
                break
            raise
        metrics["per_turn_ttf"].append(time.time() - step_t0)

    metrics["total_ttf"] = time.time() - wc_start
    metrics["completion_count"] = completed
    del compiler
    return metrics


# ===== EXP-4: Scheduler Ablation =====

def run_exp4(model_path, n_gpu_layers=99, n_steps=35, max_new=60):
    """EXP-4: Sweep (WARMUP_STEPS, MAX_K) combinations."""
    tools = KitchenToolRegistry()
    random.seed(42)
    scenario = build_kitchen_scenario(n_steps)

    results = []

    # Grid: warmup in {1, 3, 5, 8}, max_k in {1, 2, 3, 4}
    configs = []
    for w in [1, 3, 5, 8]:
        for k in [1, 2, 3, 4]:
            configs.append((w, k))

    print("EXP-4: %d configs" % len(configs))

    for i, (warmup, max_k) in enumerate(configs):
        print("  [%d/%d] warmup=%d, max_k=%d ..." % (i+1, len(configs), warmup, max_k), end="", flush=True)
        try:
            metrics, records = run_specdec_with_schedule(
                model_path, scenario, tools, n_gpu_layers=n_gpu_layers,
                max_new=max_new, warmup_steps=warmup, max_k=max_k,
                track_acceptance=True,
            )
            result = {
                "warmup_steps": warmup,
                "max_k": max_k,
                "wall_clock_s": round(metrics["total_ttf"], 3),
                "gen_time_s": round(metrics["total_gen_time"], 3),
                "gen_tokens": metrics["total_gen_tokens"],
                "tok_per_s": round(
                    metrics["total_gen_tokens"] / max(0.001, metrics["total_gen_time"]), 1),
                "completion_count": metrics["completion_count"],
                "draft_proposed": metrics.get("draft_proposed", 0),
                "draft_accepted": metrics.get("draft_accepted", 0),
                "acceptance_rate": round(metrics.get("acceptance_rate", 0), 3),
            }
            results.append(result)
            print(" %.2fs, %.1f tok/s, %d toks, AR=%.1f%%" % (
                result["wall_clock_s"], result["tok_per_s"],
                result["gen_tokens"], result["acceptance_rate"] * 100))
        except Exception as e:
            print(" ERROR: %s" % e)
            results.append({"warmup_steps": warmup, "max_k": max_k, "error": str(e)})

    return results


# ===== EXP-5: Crossover Shift =====

def run_exp5(model_path, n_gpu_layers=99, n_steps=35, max_new=60, n_runs=3):
    """EXP-5: Measure AppLoop, SIG, SIG+SpecDec for crossover analysis."""
    tools = KitchenToolRegistry()
    results = []

    for run_id in range(n_runs):
        random.seed(42)
        scenario = build_kitchen_scenario(n_steps)

        for mode in ["AppLoop", "SIG", "SIG+SpecDec"]:
            print("  Run %d/%d: %s ..." % (run_id + 1, n_runs, mode), end="", flush=True)
            try:
                if mode == "AppLoop":
                    metrics = run_apploop(model_path, scenario, tools, n_gpu_layers, max_new)
                elif mode == "SIG":
                    metrics = run_sig_baseline(model_path, scenario, tools, n_gpu_layers, max_new)
                else:
                    metrics, _ = run_specdec_with_schedule(
                        model_path, scenario, tools, n_gpu_layers, max_new,
                        warmup_steps=3, max_k=3,
                    )

                result = {
                    "run_id": run_id, "mode": mode,
                    "wall_clock_s": round(metrics["total_ttf"], 3),
                    "gen_time_s": round(metrics["total_gen_time"], 3),
                    "gen_tokens": metrics["total_gen_tokens"],
                    "tok_per_s": round(
                        metrics["total_gen_tokens"] / max(0.001, metrics["total_gen_time"]), 1),
                    "completion_count": metrics["completion_count"],
                }
                if "acceptance_rate" in metrics:
                    result["acceptance_rate"] = round(metrics["acceptance_rate"], 3)
                results.append(result)
                print(" %.2fs, %.1f tok/s" % (result["wall_clock_s"], result["tok_per_s"]))
            except Exception as e:
                print(" ERROR: %s" % e)
                results.append({"run_id": run_id, "mode": mode, "error": str(e)})

    return results


# ===== EXP-6: Drafter Selection =====

def run_exp6(model_path, n_gpu_layers=99, n_steps=35, max_new=60, n_runs=1):
    """EXP-6: Compare drafter strategies."""
    tools = KitchenToolRegistry()
    results = []

    drafters = [
        ("ngram_n3_k1", 3, 1),
        ("ngram_n3_k2", 3, 2),
        ("ngram_n3_k3", 3, 3),
        ("ngram_n3_k5", 3, 5),
        ("ngram_n2_k3", 2, 3),
        ("ngram_n4_k3", 4, 3),
    ]

    for run_id in range(n_runs):
        random.seed(42)
        scenario = build_kitchen_scenario(n_steps)

        for name, ngram_size, num_pred in drafters:
            print("  [%s] %s ..." % (name, "run %d/%d" % (run_id+1, n_runs) if n_runs > 1 else ""), end="", flush=True)
            try:
                metrics, records = run_specdec_with_schedule(
                    model_path, scenario, tools, n_gpu_layers, max_new,
                    warmup_steps=3, max_k=num_pred, ngram_size=ngram_size,
                    track_acceptance=True,
                )
                result = {
                    "run_id": run_id, "drafter": name,
                    "ngram_size": ngram_size, "num_pred": num_pred,
                    "wall_clock_s": round(metrics["total_ttf"], 3),
                    "gen_tokens": metrics["total_gen_tokens"],
                    "tok_per_s": round(
                        metrics["total_gen_tokens"] / max(0.001, metrics["total_gen_time"]), 1),
                    "draft_proposed": metrics.get("draft_proposed", 0),
                    "draft_accepted": metrics.get("draft_accepted", 0),
                    "acceptance_rate": round(metrics.get("acceptance_rate", 0), 3),
                }
                results.append(result)
                print(" AR=%.1f%%, %.1f tok/s" % (result["acceptance_rate"] * 100, result["tok_per_s"]))
            except Exception as e:
                print(" ERROR: %s" % e)
                results.append({"run_id": run_id, "drafter": name, "error": str(e)})

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--exp", default="all", choices=["all", "exp4", "exp5", "exp6"])
    parser.add_argument("--n-gpu-layers", type=int, default=99)
    parser.add_argument("--n-steps", type=int, default=35)
    parser.add_argument("--max-new", type=int, default=60)
    parser.add_argument("--n-runs", type=int, default=1)
    parser.add_argument("--output-dir", default="results/exp456")
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    model_name = Path(args.model).stem

    print("=" * 60)
    print("  EXP-4/5/6: Scheduler, Crossover, Drafter Selection")
    print("  Model: %s" % args.model)
    print("=" * 60)

    all_results = {}

    if args.exp in ("all", "exp4"):
        print("\n" + "=" * 60)
        print("  EXP-4: Scheduler Ablation")
        print("=" * 60)
        all_results["exp4"] = run_exp4(
            args.model, args.n_gpu_layers, args.n_steps, args.max_new)

    if args.exp in ("all", "exp5"):
        print("\n" + "=" * 60)
        print("  EXP-5: Crossover Shift")
        print("=" * 60)
        all_results["exp5"] = run_exp5(
            args.model, args.n_gpu_layers, args.n_steps, args.max_new, args.n_runs)

    if args.exp in ("all", "exp6"):
        print("\n" + "=" * 60)
        print("  EXP-6: Drafter Selection")
        print("=" * 60)
        all_results["exp6"] = run_exp6(
            args.model, args.n_gpu_layers, args.n_steps, args.max_new, args.n_runs)

    # Save
    outfile = os.path.join(args.output_dir, "exp456_%s.json" % model_name)
    with open(outfile, "w") as f:
        json.dump(all_results, f, indent=2)
    print("\nResults saved to %s" % outfile)

    # Print summary
    if "exp4" in all_results:
        print("\n" + "=" * 70)
        print("  EXP-4 Summary: Scheduler Ablation")
        print("=" * 70)
        print("%-8s %-8s %10s %8s %10s %8s" % ("Warmup", "MaxK", "WallClock", "Tok/s", "GenTok", "AR%"))
        print("-" * 70)
        for r in all_results["exp4"]:
            if "error" in r:
                print("%-8d %-8d ERROR" % (r["warmup_steps"], r["max_k"]))
            else:
                print("%-8d %-8d %10.2f %8.1f %10d %7.1f%%" % (
                    r["warmup_steps"], r["max_k"], r["wall_clock_s"],
                    r["tok_per_s"], r["gen_tokens"], r["acceptance_rate"] * 100))

    if "exp5" in all_results:
        print("\n" + "=" * 70)
        print("  EXP-5 Summary: Crossover Shift")
        print("=" * 70)
        by_mode = {}
        for r in all_results["exp5"]:
            if "error" not in r:
                by_mode.setdefault(r["mode"], []).append(r)
        apploop_wc = None
        for mode in ["AppLoop", "SIG", "SIG+SpecDec"]:
            if mode in by_mode:
                runs = by_mode[mode]
                wc = sum(r["wall_clock_s"] for r in runs) / len(runs)
                tps = sum(r["tok_per_s"] for r in runs) / len(runs)
                gt = sum(r["gen_tokens"] for r in runs) / len(runs)
                if mode == "AppLoop":
                    apploop_wc = wc
                speedup = "%.2fx" % (apploop_wc / wc) if apploop_wc and wc > 0 else "---"
                print("  %-15s %.2fs  %.1f tok/s  %d toks  %s" % (mode, wc, tps, gt, speedup))

    if "exp6" in all_results:
        print("\n" + "=" * 70)
        print("  EXP-6 Summary: Drafter Selection")
        print("=" * 70)
        print("%-15s %8s %8s %10s %8s" % ("Drafter", "ngram", "K", "AR%", "Tok/s"))
        print("-" * 70)
        for r in all_results["exp6"]:
            if "error" not in r:
                print("%-15s %8d %8d %9.1f%% %8.1f" % (
                    r["drafter"], r["ngram_size"], r["num_pred"],
                    r["acceptance_rate"] * 100, r["tok_per_s"]))


if __name__ == "__main__":
    main()
