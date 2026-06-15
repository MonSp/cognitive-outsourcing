"""
Paper 9 — Consolidated Experiment Orchestrator
================================================
Runs all 8 experiments (EXP-1 through EXP-8) in sequence or selectively.
Supports dry-run, single-experiment, and full-suite modes.

Usage:
  python run_all.py --dry-run                 # Print plan for all experiments
  python run_all.py --only exp1               # Run only EXP-1
  python run_all.py --only exp1,exp2          # Run EXP-1 and EXP-2
  python run_all.py                           # Run all experiments
  python run_all.py --resume exp3             # Resume from EXP-3
"""

import argparse, sys, os, time, json
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))

EXPERIMENT_REGISTRY = {
    "exp1": ("exp1_unified_speedup",     "EXP-1: Unified Kitchen Speedup",       40),
    "exp2": ("exp2_model_size_sweep",    "EXP-2: Model Size Sweep",              40),
    "exp3": ("exp3_composition_matrix",  "EXP-3: Composition Matrix",             40),
    "exp4": ("exp4_paradigm_comparison", "EXP-4: Paradigm Comparison",            60),
    "exp5": ("exp5_pareto_frontier",     "EXP-5: Pareto Frontier",               100),
    "exp6": ("exp6_generation_causation","EXP-6: Generation Causation",           20),
    "exp7": ("exp7_cross_architecture",  "EXP-7: Cross-Architecture Replication", 30),
    "exp8": ("exp8_agent_driven",        "EXP-8: Agent-Driven Kitchen",           20),
}

EXECUTION_ORDER = ["exp1", "exp2", "exp3", "exp4", "exp5", "exp6", "exp7", "exp8"]


def print_plan(selected):
    total_runs = 0
    print("=" * 70)
    print("Paper 9 — Consolidated Experiment Plan")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print("=" * 70)
    for key in EXECUTION_ORDER:
        if key not in selected:
            continue
        module_name, desc, n_runs = EXPERIMENT_REGISTRY[key]
        total_runs += n_runs
        marker = ">>>"
        print(f"  {marker} {key.upper()}: {desc} ({n_runs} runs)")
    print("-" * 70)
    print(f"  Total runs: {total_runs}")
    est_hours = total_runs * 7 / 3600
    print(f"  Estimated time: ~{est_hours:.1f} hours (assuming ~7s avg per run)")
    print("=" * 70)


def run_experiment(key, dry_run=False):
    module_name, desc, n_runs = EXPERIMENT_REGISTRY[key]
    print(f"\n{'='*70}")
    print(f"Starting {key.upper()}: {desc}")
    print(f"Expected runs: {n_runs}")
    print(f"{'='*70}\n")

    module = __import__(module_name)
    t0 = time.time()
    module.run_experiment(dry_run=dry_run)
    elapsed = time.time() - t0
    print(f"\n{key.upper()} completed in {elapsed:.1f}s")
    return elapsed


def main():
    parser = argparse.ArgumentParser(description="Paper 9 Experiment Orchestrator")
    parser.add_argument("--dry-run", action="store_true", help="Print plan without executing")
    parser.add_argument("--only", type=str, default=None,
                        help="Comma-separated experiment keys (e.g., exp1,exp2)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from this experiment (inclusive)")
    parser.add_argument("--list", action="store_true", help="List all experiments and exit")
    args = parser.parse_args()

    if args.list:
        print_plan(EXECUTION_ORDER)
        return

    if args.only:
        selected = [s.strip().lower() for s in args.only.split(",")]
        for s in selected:
            if s not in EXPERIMENT_REGISTRY:
                print(f"Unknown experiment: {s}. Valid: {', '.join(EXPERIMENT_REGISTRY.keys())}")
                sys.exit(1)
    elif args.resume:
        resume_key = args.resume.strip().lower()
        if resume_key not in EXPERIMENT_REGISTRY:
            print(f"Unknown experiment: {resume_key}")
            sys.exit(1)
        idx = EXECUTION_ORDER.index(resume_key)
        selected = EXECUTION_ORDER[idx:]
    else:
        selected = EXECUTION_ORDER

    print_plan(selected)

    if args.dry_run:
        print("\n[DRY RUN] No experiments will be executed.")
        for key in selected:
            run_experiment(key, dry_run=True)
        return

    log = {"start": datetime.now().isoformat(), "experiments": {}}
    total_elapsed = 0

    for key in selected:
        try:
            elapsed = run_experiment(key, dry_run=False)
            log["experiments"][key] = {"status": "completed", "elapsed_s": round(elapsed, 1)}
            total_elapsed += elapsed
        except Exception as e:
            log["experiments"][key] = {"status": "failed", "error": str(e)}
            print(f"\n[ERROR] {key.upper()} failed: {e}")
            print(f"Use --resume {key} to retry from this experiment.")
            break

    log["end"] = datetime.now().isoformat()
    log["total_elapsed_s"] = round(total_elapsed, 1)

    from common import RESULTS_DIR
    log_path = RESULTS_DIR / "run_all_log.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2, ensure_ascii=False)
    print(f"\nExecution log saved to {log_path}")


if __name__ == "__main__":
    main()
