"""
EXP-1: Unified Kitchen Speedup
===============================
Establishes the definitive SIG speedup under unified protocol.
Resolves Contradiction 1: 2.54x vs 3.50x vs 4.71x.

Conditions: AppLoop, SIG, AppLoop-PC, AppLoop-Sliding
Model: Qwen3.5-4B-Q4_K_M
Steps: 35, n=10, total 40 runs
"""

import argparse, sys, os
sys.path.insert(0, os.path.dirname(__file__))
from common import (
    collect_run, save_run, aggregate_experiment,
    randomized_run_order, bootstrap_ci, bootstrap_ratio_ci,
    welch_t_test, cohens_d, RESULTS_DIR
)
import json

EXPERIMENT_ID = "EXP-1"
SLUG = "unified_speedup"
MODEL = "4B"
N_RUNS = 10
KITCHEN_STEPS = 35

CONDITIONS = ["AppLoop", "SIG", "AppLoop-PC", "AppLoop-Sliding"]

BASELINE_MAP = {
    "AppLoop": "AppLoop",
    "SIG": "SIG",
    "AppLoop-PC": "AppLoop-PC",
    "AppLoop-Sliding": "Sliding",
}


def extract_baseline_metric(record, baseline_name):
    baselines = record.get("parsed_baselines", {})
    bl = baselines.get(baseline_name, {})
    return bl.get("wall_clock_s", record.get("wall_clock_s", 0.0))


def run_experiment(dry_run=False):
    order = randomized_run_order(EXPERIMENT_ID, CONDITIONS, N_RUNS)
    all_records = []

    print(f"[{EXPERIMENT_ID}] {len(order)} runs total ({len(CONDITIONS)} conditions x {N_RUNS} runs)")
    print(f"[{EXPERIMENT_ID}] Model: Qwen3.5-{MODEL}, Steps: {KITCHEN_STEPS}")

    for idx, (condition, run_id) in enumerate(order):
        print(f"  [{idx+1}/{len(order)}] {condition} run {run_id}...", end=" ", flush=True)
        if dry_run:
            print("DRY RUN - skipped")
            continue
        record = collect_run(
            experiment_id=EXPERIMENT_ID,
            condition=condition,
            model=MODEL,
            run_id=run_id,
            task="kitchen",
            kitchen_steps=KITCHEN_STEPS,
        )
        bl_name = BASELINE_MAP.get(condition, condition)
        record["target_wall_clock_s"] = extract_baseline_metric(record, bl_name)
        save_run(EXPERIMENT_ID, SLUG, run_id, record)
        all_records.append(record)
        wc = record["target_wall_clock_s"]
        print(f"ok={record['ok']} wc={wc:.2f}s")

    if dry_run:
        print(f"[{EXPERIMENT_ID}] Dry run complete. No data collected.")
        return

    for rec in all_records:
        rec["wall_clock_s"] = rec.get("target_wall_clock_s", rec["wall_clock_s"])

    agg = aggregate_experiment(EXPERIMENT_ID, SLUG, all_records)

    sig_vals = [r["wall_clock_s"] for r in all_records if r["condition"] == "SIG" and r.get("ok")]
    app_vals = [r["wall_clock_s"] for r in all_records if r["condition"] == "AppLoop" and r.get("ok")]

    if sig_vals and app_vals:
        speedup_mean, speedup_lo, speedup_hi = bootstrap_ratio_ci(app_vals, sig_vals)
        t_stat, p_val = welch_t_test(app_vals, sig_vals)
        d = cohens_d(app_vals, sig_vals)
        agg["speedup_analysis"] = {
            "speedup_mean": round(speedup_mean, 4),
            "speedup_ci95_lo": round(speedup_lo, 4),
            "speedup_ci95_hi": round(speedup_hi, 4),
            "welch_t": round(t_stat, 4),
            "welch_p": round(p_val, 6),
            "cohens_d": round(d, 4),
        }
        print(f"\n[{EXPERIMENT_ID}] SIG vs AppLoop speedup: {speedup_mean:.2f}x [{speedup_lo:.2f}, {speedup_hi:.2f}]")
        print(f"  Welch t={t_stat:.2f}, p={p_val:.6f}, Cohen's d={d:.2f}")

    analysis_path = RESULTS_DIR / f"{EXPERIMENT_ID.lower()}_{SLUG}_analysis.json"
    with open(analysis_path, "w", encoding="utf-8") as f:
        json.dump(agg, f, indent=2, ensure_ascii=False, default=str)
    print(f"[{EXPERIMENT_ID}] Results saved to {analysis_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=EXPERIMENT_ID)
    parser.add_argument("--dry-run", action="store_true", help="Print plan without executing")
    args = parser.parse_args()
    run_experiment(dry_run=args.dry_run)
