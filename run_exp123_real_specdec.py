#!/usr/bin/env python3
"""EXP-1/2/3 with real speculative decoding.

Supports two drafter modes:
  1. n-gram drafter (default): zero VRAM overhead, works on all models
  2. Model-based drafter (--drafter): uses a separate model for drafting

Usage:
  python run_exp123_real_specdec.py --model models/Qwen3.5-4B-Q4_K_M.gguf --exp all
  python run_exp123_real_specdec.py --model models/Qwen3.5-4B-Q4_K_M.gguf --drafter models/Qwen3.5-0.8B-Q4_K_M.gguf --exp exp1
  python run_exp123_real_specdec.py --model models/Qwen3.5-0.8B-Q4_K_M.gguf --exp exp1
"""

import argparse
import random
import sys
import os
import time
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional

sys.path.insert(0, '.')

from core import MeaningCompiler, InjectionEngine, GPUMonitor, init_metrics
from core.llamacpp_specdec import ManualSpecDecCompiler
from edge_agent_bench import (
    KITCHEN_SYSTEM_PROMPT, KitchenToolRegistry, KitchenStep,
    build_kitchen_scenario,
)


def run_apploop(model_path, scenario, tools, n_gpu_layers=99, max_new=60):
    """AppLoop baseline (no SIG, no SpecDec)."""
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


def run_apploop_specdec(model_path, scenario, tools, n_gpu_layers=99, max_new=60,
                        ngram_size=3, num_pred=3):
    """AppLoop + n-gram SpecDec (no SIG, with SpecDec)."""
    compiler = ManualSpecDecCompiler(
        model_path, n_ctx=16384, n_gpu_layers=n_gpu_layers,
        ngram_size=ngram_size, num_pred=num_pred,
    )
    metrics = init_metrics()
    context = "%s\n\n" % KITCHEN_SYSTEM_PROMPT
    completed = 0
    total_draft_proposed = 0
    total_draft_accepted = 0
    wc_start = time.time()

    for step in scenario:
        step_t0 = time.time()
        context += "\nUser: %s\n" % step.user_query
        result = tools.execute(step.tool_name, step.tool_args)
        context += "[Tool: %s] %s\nAssistant:" % (step.tool_name, result)

        full_ids = compiler.tokenize(context)
        compiler.reset_cache()
        try:
            compiler.eval(full_ids)
            gen_t0 = time.time()
            gen_text, gen_ids, stats = compiler.generate_with_specdec(
                stop_str="\nUser:", max_new=max_new, num_pred=num_pred)
            gen_time = time.time() - gen_t0

            metrics["total_gen_time"] += gen_time
            metrics["total_gen_tokens"] += len(gen_ids)
            metrics["total_prefill_tokens"] += len(full_ids)
            total_draft_proposed += stats["draft_proposed"]
            total_draft_accepted += stats["draft_accepted"]
            context += gen_text + "\n"
            completed += 1
        except RuntimeError as e:
            if "failed to find a memory slot" in str(e) or "decode returned" in str(e):
                break
            raise
        metrics["per_turn_ttf"].append(time.time() - step_t0)

    metrics["total_ttf"] = time.time() - wc_start
    metrics["completion_count"] = completed
    metrics["draft_proposed"] = total_draft_proposed
    metrics["draft_accepted"] = total_draft_accepted
    metrics["acceptance_rate"] = (
        total_draft_accepted / total_draft_proposed
        if total_draft_proposed > 0 else 0.0
    )
    del compiler
    return metrics


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


def run_sig_specdec(model_path, scenario, tools, n_gpu_layers=99, max_new=60,
                    ngram_size=3, num_pred=3):
    """SIG + n-gram SpecDec."""
    compiler = ManualSpecDecCompiler(
        model_path, n_ctx=16384, n_gpu_layers=n_gpu_layers,
        ngram_size=ngram_size, num_pred=num_pred,
    )
    metrics = init_metrics()
    total_draft_proposed = 0
    total_draft_accepted = 0

    sys_ids = compiler.tokenize("%s\n\n" % KITCHEN_SYSTEM_PROMPT)
    compiler.reset_cache()
    compiler.eval(sys_ids)

    wc_start = time.time()
    completed = 0

    for step in scenario:
        step_t0 = time.time()

        user_line = "\nUser: %s\nAssistant:" % step.user_query
        u_ids = compiler.tokenize(user_line)
        compiler.eval(u_ids)

        result = tools.execute(step.tool_name, step.tool_args)
        tool_line = "\n[Tool: %s] %s\n" % (step.tool_name, result)
        t_ids = compiler.tokenize(tool_line)
        compiler.eval(t_ids)

        gen_t0 = time.time()
        gen_text, gen_ids, stats = compiler.generate_with_specdec(
            stop_str="\nUser:", max_new=max_new, num_pred=num_pred)
        gen_time = time.time() - gen_t0

        metrics["total_gen_time"] += gen_time
        metrics["total_gen_tokens"] += len(gen_ids)
        total_draft_proposed += stats["draft_proposed"]
        total_draft_accepted += stats["draft_accepted"]
        completed += 1
        metrics["per_turn_ttf"].append(time.time() - step_t0)

    metrics["total_ttf"] = time.time() - wc_start
    metrics["completion_count"] = completed
    metrics["draft_proposed"] = total_draft_proposed
    metrics["draft_accepted"] = total_draft_accepted
    metrics["acceptance_rate"] = (
        total_draft_accepted / total_draft_proposed
        if total_draft_proposed > 0 else 0.0
    )
    del compiler
    return metrics


# ===== EXP-1: Orthogonality Validation =====

def run_exp1(model_path, n_gpu_layers=99, n_steps=35, max_new=60, n_runs=3):
    """EXP-1: Measure orthogonality ratio with 4 conditions.

    Measures:
      S_SIG = T_AppLoop / T_SIG
      S_SpecDec = T_AppLoop / T_AppLoop+SpecDec
      S_SIG+SpecDec = T_AppLoop / T_SIG+SpecDec
      rho = S_SIG+SpecDec / (S_SIG * S_SpecDec)
    """
    tools = KitchenToolRegistry()
    results = []

    for run_id in range(n_runs):
        random.seed(42)
        scenario = build_kitchen_scenario(n_steps)

        for mode in ["AppLoop", "SIG", "AppLoop+SpecDec", "SIG+SpecDec"]:
            print("  Run %d/%d: %s ..." % (run_id + 1, n_runs, mode), end="", flush=True)
            try:
                if mode == "AppLoop":
                    metrics = run_apploop(model_path, scenario, tools, n_gpu_layers, max_new)
                elif mode == "SIG":
                    metrics = run_sig_baseline(model_path, scenario, tools, n_gpu_layers, max_new)
                elif mode == "AppLoop+SpecDec":
                    metrics = run_apploop_specdec(model_path, scenario, tools, n_gpu_layers, max_new)
                else:
                    metrics = run_sig_specdec(model_path, scenario, tools, n_gpu_layers, max_new)

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
                print(" %.2fs, %.1f tok/s, %d toks" % (
                    result["wall_clock_s"], result["tok_per_s"], result["gen_tokens"]))
            except Exception as e:
                print(" ERROR: %s" % e)
                results.append({"run_id": run_id, "mode": mode, "error": str(e)})

    return results


# ===== EXP-2: Acceptance Rate Characterization =====

def run_exp2(model_path, n_gpu_layers=99, n_steps=50, max_new=60, n_runs=3):
    """EXP-2: Measure post-injection acceptance rate recovery with n-gram drafter."""
    tools = KitchenToolRegistry()
    results = []

    for run_id in range(n_runs):
        random.seed(42)
        scenario = build_kitchen_scenario(n_steps)

        print("  Run %d/%d: SIG+SpecDec with acceptance tracking ..." % (run_id + 1, n_runs), end="", flush=True)
        try:
            metrics = run_sig_specdec(
                model_path, scenario, tools, n_gpu_layers, max_new)
            result = {
                "run_id": run_id,
                "wall_clock_s": round(metrics["total_ttf"], 3),
                "gen_tokens": metrics["total_gen_tokens"],
                "acceptance_rate": round(metrics.get("acceptance_rate", 0), 3),
            }
            results.append(result)
            print(" AR=%.1f%%" % (metrics.get("acceptance_rate", 0) * 100))
        except Exception as e:
            print(" ERROR: %s" % e)
            results.append({"run_id": run_id, "error": str(e)})

    return results


# ===== EXP-3: Compound Acceleration =====

def run_exp3(model_path, n_gpu_layers=99, n_steps=35, max_new=60, n_runs=3):
    """EXP-3: Measure compound acceleration with n-gram SpecDec."""
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
                    metrics = run_sig_specdec(model_path, scenario, tools, n_gpu_layers, max_new)

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


def compute_orthogonality(results):
    """Compute orthogonality ratio from EXP-1 results."""
    by_mode = {}
    for r in results:
        if "error" not in r:
            by_mode.setdefault(r["mode"], []).append(r)

    if "AppLoop" not in by_mode or "SIG" not in by_mode:
        return None

    apploop_wc = np.mean([r["wall_clock_s"] for r in by_mode["AppLoop"]])
    sig_wc = np.mean([r["wall_clock_s"] for r in by_mode["SIG"]])

    out = {
        "S_SIG": round(apploop_wc / sig_wc, 3) if sig_wc > 0 else 0,
    }

    if "AppLoop+SpecDec" in by_mode:
        apploop_sd_wc = np.mean([r["wall_clock_s"] for r in by_mode["AppLoop+SpecDec"]])
        out["S_SpecDec"] = round(apploop_wc / apploop_sd_wc, 3) if apploop_sd_wc > 0 else 0

    if "SIG+SpecDec" in by_mode:
        sig_sd_wc = np.mean([r["wall_clock_s"] for r in by_mode["SIG+SpecDec"]])
        out["S_SIG+SpecDec"] = round(apploop_wc / sig_sd_wc, 3) if sig_sd_wc > 0 else 0

    if "S_SpecDec" in out and "S_SIG+SpecDec" in out and out["S_SIG"] > 0 and out["S_SpecDec"] > 0:
        out["rho"] = round(out["S_SIG+SpecDec"] / (out["S_SIG"] * out["S_SpecDec"]), 3)

    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--drafter", default=None, help="Path to drafter model (optional, uses n-gram if not provided)")
    parser.add_argument("--exp", default="all", choices=["all", "exp1", "exp2", "exp3"])
    parser.add_argument("--n-gpu-layers", type=int, default=99)
    parser.add_argument("--n-steps", type=int, default=35)
    parser.add_argument("--max-new", type=int, default=60)
    parser.add_argument("--n-runs", type=int, default=3)
    parser.add_argument("--output-dir", default="data/exp123_real_specdec")
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    model_name = Path(args.model).stem

    drafter_label = "ngram"
    if args.drafter:
        drafter_label = Path(args.drafter).stem

    print("=" * 60)
    print("  EXP-1/2/3 with Real Speculative Decoding")
    print("  Model: %s" % args.model)
    print("  Drafter: %s" % (args.drafter if args.drafter else "n-gram (built-in)"))
    print("=" * 60)

    all_results = {}

    if args.exp in ("all", "exp1"):
        print("\n" + "=" * 60)
        print("  EXP-1: Orthogonality Validation")
        print("=" * 60)
        all_results["exp1"] = run_exp1(
            args.model, args.n_gpu_layers, args.n_steps, args.max_new, args.n_runs)
        ortho = compute_orthogonality(all_results["exp1"])
        if ortho:
            all_results["exp1_orthogonality"] = ortho

    if args.exp in ("all", "exp2"):
        print("\n" + "=" * 60)
        print("  EXP-2: Acceptance Rate Characterization")
        print("=" * 60)
        all_results["exp2"] = run_exp2(
            args.model, args.n_gpu_layers, args.n_steps, args.max_new, args.n_runs)

    if args.exp in ("all", "exp3"):
        print("\n" + "=" * 60)
        print("  EXP-3: Compound Acceleration")
        print("=" * 60)
        all_results["exp3"] = run_exp3(
            args.model, args.n_gpu_layers, args.n_steps, args.max_new, args.n_runs)

    outfile = os.path.join(args.output_dir, "exp123_%s_%s.json" % (model_name, drafter_label))
    with open(outfile, "w") as f:
        json.dump(all_results, f, indent=2)
    print("\nResults saved to %s" % outfile)

    if "exp1" in all_results:
        print("\n" + "=" * 70)
        print("  EXP-1 Summary: Orthogonality Validation")
        print("=" * 70)
        by_mode = {}
        for r in all_results["exp1"]:
            if "error" not in r:
                by_mode.setdefault(r["mode"], []).append(r)
        apploop_wc = None
        for mode in ["AppLoop", "SIG", "AppLoop+SpecDec", "SIG+SpecDec"]:
            if mode in by_mode:
                runs = by_mode[mode]
                wc = np.mean([r["wall_clock_s"] for r in runs])
                tps = np.mean([r["tok_per_s"] for r in runs])
                gt = np.mean([r["gen_tokens"] for r in runs])
                if mode == "AppLoop":
                    apploop_wc = wc
                speedup = "%.2fx" % (apploop_wc / wc) if apploop_wc and wc > 0 else "---"
                ar = ""
                if "acceptance_rate" in runs[0]:
                    ar = " AR=%.1f%%" % (runs[0]["acceptance_rate"] * 100)
                print("  %-18s %6.2fs  %6.1f tok/s  %5.0f toks  %s%s" % (mode, wc, tps, gt, speedup, ar))

        if "exp1_orthogonality" in all_results:
            o = all_results["exp1_orthogonality"]
            print("\n  Orthogonality Analysis:")
            print("    S_SIG          = %.3f" % o.get("S_SIG", 0))
            print("    S_SpecDec      = %.3f" % o.get("S_SpecDec", 0))
            print("    S_SIG+SpecDec  = %.3f" % o.get("S_SIG+SpecDec", 0))
            rho = o.get("rho", 0)
            status = "PASS" if rho >= 0.85 else "FAIL"
            print("    rho            = %.3f  (%s, threshold >= 0.85)" % (rho, status))

    if "exp3" in all_results:
        print("\n" + "=" * 70)
        print("  EXP-3 Summary: Compound Acceleration")
        print("=" * 70)
        by_mode = {}
        for r in all_results["exp3"]:
            if "error" not in r:
                by_mode.setdefault(r["mode"], []).append(r)
        apploop_wc = None
        for mode in ["AppLoop", "SIG", "SIG+SpecDec"]:
            if mode in by_mode:
                runs = by_mode[mode]
                wc = np.mean([r["wall_clock_s"] for r in runs])
                tps = np.mean([r["tok_per_s"] for r in runs])
                gt = np.mean([r["gen_tokens"] for r in runs])
                if mode == "AppLoop":
                    apploop_wc = wc
                speedup = "%.2fx" % (apploop_wc / wc) if apploop_wc and wc > 0 else "---"
                ar = ""
                if "acceptance_rate" in runs[0]:
                    ar = " AR=%.1f%%" % (runs[0]["acceptance_rate"] * 100)
                print("  %-18s %6.2fs  %6.1f tok/s  %5.0f toks  %s%s" % (mode, wc, tps, gt, speedup, ar))


if __name__ == "__main__":
    main()
