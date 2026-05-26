#!/usr/bin/env python3
"""
Comprehensive Data Collection Orchestrator for CO+SIG Research Vectors R6-R19 + EdgeAgent-Kitchen.

Runs ALL experiments across four test harnesses, captures stdout to
timestamped log files, and prints a summary of results.

Usage:
  python run_all_experiments.py --model models/Qwen3.5-0.8B-Q4_K_M.gguf --n-gpu-layers 99
  python run_all_experiments.py --model ... --only r6,r10,e15
  python run_all_experiments.py --model ... --skip r7,r9
  python run_all_experiments.py --dry-run
"""

import argparse
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path


PROJECT = os.path.dirname(os.path.abspath(__file__))
PYTHON = sys.executable
TORCH_LIB = os.path.join(
    os.path.dirname(sys.executable), "..", "..", "Roaming",
    "Python", "Python311", "site-packages", "torch", "lib"
)

TASK_CONFIG = {
    "r6": {
        "script": "co_benchmark.py",
        "needs_model": True,
        "description": "Dynamic Replanning — 30-tool chain, 15% failure injection",
        "default_runs": 30,
    },
    "r7": {
        "script": "sig_benchmark.py",
        "needs_model": True,
        "description": "Multimodal SIG — structured data token efficiency",
        "default_runs": 1,
    },
    "r8": {
        "script": "sig_benchmark.py",
        "needs_model": True,
        "description": "Long-Context Precise Retrieval + Spatial Reasoning",
        "default_runs": 30,
    },
    "r9": {
        "script": "sig_benchmark.py",
        "needs_model": True,
        "description": "Real-Time Constrained SIG — latency budget analysis",
        "default_runs": 1,
    },
    "r10": {
        "script": "transformer_bench.py",
        "needs_model": True,
        "description": "Injection Attacks & Defense — 10 attack vectors, rollback isolation",
        "default_runs": None,
    },
    "r11": {
        "script": "transformer_bench.py",
        "needs_model": True,
        "description": "Tool-Result Faithfulness — Token-Jaccard fidelity evaluation",
        "default_runs": None,
    },
    "r12": {
        "script": "transformer_bench.py",
        "needs_model": False,
        "description": "SIG Scaling Law — measured prefill + analytic projections",
        "default_runs": None,
    },
    "r13": {
        "script": "co_benchmark.py",
        "needs_model": True,
        "description": "Fragmented Local KV Reconstruction — end-to-end wall-clock",
        "default_runs": 30,
    },
    "r14": {
        "script": "co_benchmark.py",
        "needs_model": True,
        "description": "SIG + Reasoning Paradigms — CoT+SIG vs CoT+AppLoop vs AppLoop-PC",
        "default_runs": 30,
    },
    "kitchen": {
        "script": "edge_agent_bench.py",
        "needs_model": True,
        "description": "EdgeAgent-Kitchen — 5 baselines, 65-step interleaved agent task",
        "default_runs": 1,
    },
    "e15": {
        "script": "edge_agent_bench.py",
        "needs_model": True,
        "description": "Hybrid Scheduling — SIG vs AppLoop-PC adaptive switching (R15)",
        "default_runs": 1,
    },
    "e16": {
        "script": "edge_agent_bench.py",
        "needs_model": True,
        "description": "Multi-Sequence Concurrency — multi-tenant KV isolation (R16)",
        "default_runs": 1,
    },
    "e17": {
        "script": "edge_agent_bench.py",
        "needs_model": True,
        "description": "Context Aging & Compression — KV memory management (R17)",
        "default_runs": 1,
    },
    "e18": {
        "script": "edge_agent_bench.py",
        "needs_model": True,
        "description": "Prefill-Decode Pipeline — SIG + speculative decoding (R18)",
        "default_runs": 1,
    },
    "e19": {
        "script": "edge_agent_bench.py",
        "needs_model": True,
        "description": "Edge Cluster Fragment Routing — distributed KV (R19)",
        "default_runs": 1,
    },
}

TASK_ORDER = ["r6", "r7", "r8", "r9", "r10", "r11", "r12", "r13", "r14",
              "kitchen", "e15", "e16", "e17", "e18", "e19"]


EDGE_TASK_MAP = {
    "kitchen": "kitchen",
    "e15": "r15", "e16": "r16", "e17": "r17",
    "e18": "r18", "e19": "r19",
}


def build_command(task_id, model, n_gpu_layers, n_ctx):
    config = TASK_CONFIG[task_id]
    script = os.path.join(PROJECT, config["script"])

    actual_task = EDGE_TASK_MAP.get(task_id, task_id)
    base_args = ["--task", actual_task]

    if config["needs_model"]:
        base_args.extend(["--model", model])

    if config["script"] == "transformer_bench.py":
        base_args.extend(["--n-ctx", str(n_ctx), "--n-gpu-layers", str(n_gpu_layers)])
    else:
        base_args.extend([
            "--n-ctx", str(n_ctx),
            "--n-gpu-layers", str(n_gpu_layers),
            "--no-debug",
        ])

    return [PYTHON, "-u", script] + base_args


def run_experiment(task_id, cmd, log_dir, task_label):
    log_path = os.path.join(log_dir, f"{task_id}.log")
    start_time = time.time()

    print(f"\n{'='*70}")
    print(f"  [{task_id}] {task_label}")
    print(f"  CMD: {' '.join(cmd)}")
    print(f"  LOG: {log_path}")
    print(f"{'='*70}\n", flush=True)

    env = os.environ.copy()
    if os.path.isdir(TORCH_LIB):
        env["PATH"] = TORCH_LIB + ";" + env.get("PATH", "")

    output_lines = []

    try:
        with open(log_path, "w", encoding="utf-8") as log_f:
            log_f.write(f"# [{task_id}] {task_label}\n")
            log_f.write(f"# CMD: {' '.join(cmd)}\n")
            log_f.write(f"# Started: {datetime.now().isoformat()}\n")
            log_f.write(f"# {'='*60}\n\n")

            proc = subprocess.Popen(
                cmd, env=env,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, cwd=PROJECT,
            )

            for line in proc.stdout:
                print(line, end="", flush=True)
                log_f.write(line)
                output_lines.append(line)

            proc.wait()
            elapsed = time.time() - start_time
            rc = proc.returncode

            log_f.write(f"\n\n# Exit code: {rc}\n")
            log_f.write(f"# Elapsed: {elapsed:.1f}s\n")
            log_f.write(f"# Finished: {datetime.now().isoformat()}\n")

    except FileNotFoundError as e:
        elapsed = time.time() - start_time
        rc = -1
        msg = f"ERROR: {e}"
        print(f"\n  {msg}\n", flush=True)
        with open(log_path, "w", encoding="utf-8") as log_f:
            log_f.write(f"{msg}\n")

    except Exception as e:
        elapsed = time.time() - start_time
        rc = -1
        msg = f"ERROR: {e}"
        print(f"\n  {msg}\n", flush=True)
        with open(log_path, "a", encoding="utf-8") as log_f:
            log_f.write(f"\n\n{msg}\n")

    status = "OK" if rc == 0 else f"FAIL(rc={rc})"
    full_output = "".join(output_lines)

    return {
        "task": task_id,
        "status": status,
        "elapsed": elapsed,
        "log": log_path,
        "output": full_output,
        "returncode": rc,
    }


def print_summary(results, started_at):
    print("\n\n")
    print("=" * 70)
    print("  EXPERIMENT SUMMARY")
    print("=" * 70)

    total_elapsed = sum(r["elapsed"] for r in results)
    n_ok = sum(1 for r in results if r["returncode"] == 0)
    n_fail = sum(1 for r in results if r["returncode"] != 0)
    n_skip = sum(1 for r in results if r["status"] == "SKIPPED")

    print(f"\n  Started:  {started_at.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Total wall-clock: {total_elapsed:.0f}s ({timedelta(seconds=int(total_elapsed))})")
    print(f"  Tasks: {len(results)} total, {n_ok} OK, {n_fail} FAIL, {n_skip} SKIPPED")
    print()

    header = f"  {'Task':<6} {'Description':<50} {'Status':<14} {'Elapsed':<10} {'Log'}"
    print(header)
    print(f"  {'-'*6} {'-'*50} {'-'*14} {'-'*10} {'-'*30}")

    for r in results:
        desc = TASK_CONFIG[r["task"]]["description"][:48]
        elapsed_str = f"{r['elapsed']:.1f}s"
        log_name = os.path.basename(r["log"])
        print(f"  {r['task']:<6} {desc:<50} {r['status']:<14} {elapsed_str:<10} {log_name}")

    print(f"\n  {'='*60}")
    if n_fail > 0:
        print(f"  WARNING: {n_fail} task(s) failed. Check individual logs for details.")
        for r in results:
            if r["returncode"] != 0 and r["status"] != "SKIPPED":
                print(f"    - {r['task']}: log={r['log']}")
    else:
        print(f"  All tasks completed successfully.")


def main():
    parser = argparse.ArgumentParser(
        description="CO+SIG Comprehensive Data Collection — R6 through R14",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_all_experiments.py --model models/Qwen3.5-0.8B-Q4_K_M.gguf
  python run_all_experiments.py --model ... --only r6,r10,r12
  python run_all_experiments.py --model ... --skip r7,r9
  python run_all_experiments.py --dry-run
        """,
    )

    parser.add_argument(
        "--model", type=str, default="",
        help="Path to GGUF model file (required for model-dependent tasks: R6-R11,R13-R14)",
    )
    parser.add_argument(
        "--n-ctx", type=int, default=16384,
        help="Context window size (default: 16384)",
    )
    parser.add_argument(
        "--n-gpu-layers", type=int, default=99,
        help="Number of GPU layers for llama.cpp (default: 99)",
    )
    parser.add_argument(
        "--skip", type=str, default="",
        help="Comma-separated task IDs to skip (e.g., 'r7,r9')",
    )
    parser.add_argument(
        "--only", type=str, default="",
        help="Comma-separated task IDs to run exclusively (e.g., 'r6,r10,r12')",
    )
    parser.add_argument(
        "--r6-runs", type=int, default=30,
        help="Number of paired runs for R6 dynamic replanning (default: 30)",
    )
    parser.add_argument(
        "--r6-tool-depth", type=int, default=30,
        help="Tool chain depth for R6 (default: 30; options: 14, 22, 30)",
    )
    parser.add_argument(
        "--r8-runs", type=int, default=30,
        help="Number of paired runs for R8 long-context retrieval (default: 30)",
    )
    parser.add_argument(
        "--r13-runs", type=int, default=30,
        help="Number of paired runs for R13 fragmented KV reconstruction (default: 30)",
    )
    parser.add_argument(
        "--r14-runs", type=int, default=30,
        help="Number of paired runs for R14 CoT+SIG evaluation (default: 30)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print what would be executed without running anything",
    )
    parser.add_argument(
        "--log-dir", type=str, default="",
        help="Override log output directory (default: logs/YYYYMMDD_HHMMSS/)",
    )

    args = parser.parse_args()

    skip_set = set(x.strip().lower() for x in args.skip.split(",") if x.strip())
    only_set = set(x.strip().lower() for x in args.only.split(",") if x.strip())

    if only_set:
        unknown = only_set - set(TASK_CONFIG.keys())
        if unknown:
            print(f"ERROR: Unknown task(s) in --only: {', '.join(sorted(unknown))}")
            print(f"       Available tasks: {', '.join(TASK_ORDER)}")
            sys.exit(1)
        ordered_tasks = [t for t in TASK_ORDER if t in only_set]
    else:
        ordered_tasks = list(TASK_ORDER)

    ordered_tasks = [t for t in ordered_tasks if t not in skip_set]

    if not ordered_tasks:
        print("No tasks selected to run.")
        return

    if args.log_dir:
        log_dir = args.log_dir
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = os.path.join(PROJECT, "logs", timestamp)

    started_at = datetime.now()

    print("=" * 70)
    print("  CO+SIG Comprehensive Data Collection — R6 through R19 + Kitchen")
    print("=" * 70)
    print(f"\n  Model:     {args.model if args.model else '(not specified — model-dependent tasks will skip)'}")
    print(f"  n_ctx:     {args.n_ctx}")
    print(f"  GPU layers: {args.n_gpu_layers}")
    print(f"  Log dir:   {log_dir}")
    print(f"  Tasks:     {', '.join(ordered_tasks)}")
    print(f"  R6 runs:   {args.r6_runs} (tool depth: {args.r6_tool_depth})")
    print(f"  R8 runs:   {args.r8_runs}")
    print(f"  R13 runs:  {args.r13_runs}")
    print(f"  R14 runs:  {args.r14_runs}")
    print(f"  Dry run:   {args.dry_run}")
    print()

    if args.dry_run:
        print("DRY RUN — commands that would be executed:\n")
        for task_id in ordered_tasks:
            config = TASK_CONFIG[task_id]
            needs_model = config["needs_model"]
            if needs_model and not args.model:
                print(f"  [{task_id}] SKIP (no --model) — {config['description']}")
                continue
            cmd = build_command(task_id, args.model, args.n_gpu_layers, args.n_ctx)
            print(f"  [{task_id}] {' '.join(cmd)}")
        print("\nDry run complete. No experiments executed.")
        return

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    results = []

    for task_id in ordered_tasks:
        config = TASK_CONFIG[task_id]
        needs_model = config["needs_model"]

        if needs_model and not args.model:
            print(f"\n  [{task_id}] SKIPPED — no --model provided "
                  f"({config['description']})")
            results.append({
                "task": task_id,
                "status": "SKIPPED",
                "elapsed": 0.0,
                "log": "(none)",
                "output": "",
                "returncode": -1,
            })
            continue

        cmd = build_command(task_id, args.model, args.n_gpu_layers, args.n_ctx)

        result = run_experiment(task_id, cmd, log_dir, config["description"])
        results.append(result)
        time.sleep(1)
    print_summary(results, started_at)

    log_index = os.path.join(log_dir, "_summary.txt")
    with open(log_index, "w", encoding="utf-8") as f:
        f.write(f"CO+SIG Experiment Summary — {started_at.isoformat()}\n")
        f.write(f"{'='*60}\n\n")
        for r in results:
            f.write(f"  {r['task']}: {r['status']}  ({r['elapsed']:.1f}s)  -> {os.path.basename(r['log'])}\n")

    print(f"\n  Summary written to: {log_index}")


if __name__ == "__main__":
    main()
