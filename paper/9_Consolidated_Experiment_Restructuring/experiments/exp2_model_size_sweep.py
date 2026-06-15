"""
EXP-2: Model Size Sweep
========================
Locates SIG-vs-AppLoop crossover point with 4 data points.
Resolves Contradiction 2: ~0.7B vs ~1B vs 1.5-2B.

Conditions: AppLoop + SIG for each of 0.5B, 0.8B, 2B, 4B
Steps: 35, n=10, total 80 runs
"""

import argparse, sys, os, json
sys.path.insert(0, os.path.dirname(__file__))
from common import (
    collect_run, save_run, aggregate_experiment,
    randomized_run_order, bootstrap_ratio_ci, RESULTS_DIR
)
import numpy as np

EXPERIMENT_ID = "EXP-2"
SLUG = "model_size_sweep"
N_RUNS = 10
KITCHEN_STEPS = 35
SIZES = ["0.8B", "4B"]
MODES = ["AppLoop", "SIG"]

BASELINE_MAP = {"AppLoop": "AppLoop", "SIG": "SIG"}


def make_conditions():
    conds = []
    for size in SIZES:
        for mode in MODES:
            conds.append(f"{mode}_{size}")
    return conds


def extract_baseline_metric(record, baseline_name):
    baselines = record.get("parsed_baselines", {})
    bl = baselines.get(baseline_name, {})
    return bl.get("wall_clock_s", record.get("wall_clock_s", 0.0))


def fit_crossover(sizes_b, speedups):
    from scipy.optimize import curve_fit
    x = np.array(sizes_b, dtype=float)
    y = np.array(speedups, dtype=float)

    def log_model(m, a, c):
        return a * np.log(m) + c

    def power_model(m, a, b, c):
        return a * np.power(m, b) + c

    results = {}
    try:
        popt, _ = curve_fit(log_model, x, y)
        x_fine = np.linspace(0.3, 5.0, 1000)
        y_fine = log_model(x_fine, *popt)
        idx = np.argmin(np.abs(y_fine - 1.0))
        crossover = float(x_fine[idx])
        residuals = y - log_model(x, *popt)
        sse = float(np.sum(residuals**2))
        results["log"] = {"crossover_b": crossover, "sse": sse, "params": [float(p) for p in popt]}
    except Exception:
        pass

    try:
        popt, _ = curve_fit(power_model, x, y, p0=[1.0, 0.5, 0.0], maxfev=5000)
        x_fine = np.linspace(0.3, 5.0, 1000)
        y_fine = power_model(x_fine, *popt)
        idx = np.argmin(np.abs(y_fine - 1.0))
        crossover = float(x_fine[idx])
        residuals = y - power_model(x, *popt)
        sse = float(np.sum(residuals**2))
        results["power"] = {"crossover_b": crossover, "sse": sse, "params": [float(p) for p in popt]}
    except Exception:
        pass

    return results


def run_experiment(dry_run=False):
    conditions = make_conditions()
    order = randomized_run_order(EXPERIMENT_ID, conditions, N_RUNS)
    all_records = []

    print(f"[{EXPERIMENT_ID}] {len(order)} runs total ({len(conditions)} conditions x {N_RUNS} runs)")

    for idx, (condition, run_id) in enumerate(order):
        parts = condition.split("_")
        mode = parts[0]
        size = parts[1]
        print(f"  [{idx+1}/{len(order)}] {condition} run {run_id}...", end=" ", flush=True)
        if dry_run:
            print("DRY RUN - skipped")
            continue
        record = collect_run(
            experiment_id=EXPERIMENT_ID,
            condition=condition,
            model=size,
            run_id=run_id,
            task="kitchen",
            kitchen_steps=KITCHEN_STEPS,
        )
        bl_name = BASELINE_MAP.get(mode, mode)
        record["target_wall_clock_s"] = extract_baseline_metric(record, bl_name)
        record["mode"] = mode
        record["model_size"] = size
        save_run(EXPERIMENT_ID, SLUG, run_id, record)
        all_records.append(record)
        wc = record["target_wall_clock_s"]
        print(f"ok={record['ok']} wc={wc:.2f}s")

    if dry_run:
        print(f"[{EXPERIMENT_ID}] Dry run complete.")
        return

    for rec in all_records:
        rec["wall_clock_s"] = rec.get("target_wall_clock_s", rec["wall_clock_s"])

    agg = aggregate_experiment(EXPERIMENT_ID, SLUG, all_records)

    speedup_data = {}
    for size in SIZES:
        sig_vals = [r["wall_clock_s"] for r in all_records
                    if r["model_size"] == size and r["mode"] == "SIG" and r.get("ok")]
        app_vals = [r["wall_clock_s"] for r in all_records
                    if r["model_size"] == size and r["mode"] == "AppLoop" and r.get("ok")]
        if sig_vals and app_vals:
            s_mean, s_lo, s_hi = bootstrap_ratio_ci(app_vals, sig_vals)
            speedup_data[size] = {
                "speedup_mean": round(s_mean, 4),
                "speedup_ci95_lo": round(s_lo, 4),
                "speedup_ci95_hi": round(s_hi, 4),
            }

    agg["speedup_by_size"] = speedup_data

    if len(speedup_data) >= 3:
        sizes_b = [float(s.replace("B", "")) for s in speedup_data.keys()]
        speedups = [d["speedup_mean"] for d in speedup_data.values()]
        fit_results = fit_crossover(sizes_b, speedups)
        agg["crossover_fit"] = fit_results
        if "log" in fit_results:
            print(f"\n[{EXPERIMENT_ID}] Crossover (log fit): {fit_results['log']['crossover_b']:.2f}B")
        if "power" in fit_results:
            print(f"[{EXPERIMENT_ID}] Crossover (power fit): {fit_results['power']['crossover_b']:.2f}B")

    analysis_path = RESULTS_DIR / f"{EXPERIMENT_ID.lower()}_{SLUG}_analysis.json"
    with open(analysis_path, "w", encoding="utf-8") as f:
        json.dump(agg, f, indent=2, ensure_ascii=False, default=str)
    print(f"[{EXPERIMENT_ID}] Results saved to {analysis_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=EXPERIMENT_ID)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    run_experiment(dry_run=args.dry_run)
