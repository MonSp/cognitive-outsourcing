"""
EXP-5: Pareto Frontier
=======================
Maps speedup-quality-memory Pareto frontier across chain depth.
Resolves Contradiction 5: coverage non-monotonicity (33% vs 1%).

5 depths x 3 models x 3 modes x 5 runs = 225 runs
"""

import argparse, sys, os, json
sys.path.insert(0, os.path.dirname(__file__))
from common import (
    collect_run, save_run, aggregate_experiment,
    randomized_run_order, bootstrap_ratio_ci, RESULTS_DIR
)

EXPERIMENT_ID = "EXP-5"
SLUG = "pareto_frontier"
N_RUNS = 5
DEPTHS = [5, 10, 20, 35, 50]
MODELS = ["0.8B", "4B"]
MODES = ["AppLoop", "SIG"]


def make_conditions():
    conds = []
    for depth in DEPTHS:
        for model in MODELS:
            for mode in MODES:
                conds.append(f"D{depth}_{model}_{mode}")
    return conds


def get_extra_args(mode):
    if mode == "SIG+CompSIG":
        return "SKIP"
    return None


def run_experiment(dry_run=False):
    conditions = make_conditions()
    order = randomized_run_order(EXPERIMENT_ID, conditions, N_RUNS)
    all_records = []

    print(f"[{EXPERIMENT_ID}] {len(order)} runs total")

    for idx, (condition, run_id) in enumerate(order):
        parts = condition.split("_")
        depth = int(parts[0][1:])
        model = parts[1]
        mode = parts[2]
        print(f"  [{idx+1}/{len(order)}] {condition} run {run_id}...", end=" ", flush=True)
        if dry_run:
            print("DRY RUN")
            continue

        extra = get_extra_args(mode)
        if extra == "SKIP":
            print(f"SKIP (unsupported)")
            continue
        record = collect_run(
            experiment_id=EXPERIMENT_ID,
            condition=condition,
            model=model,
            run_id=run_id,
            task="kitchen",
            kitchen_steps=depth,
            extra_args=extra,
        )
        record["depth"] = depth
        record["model_size"] = model
        record["mode"] = mode
        # Extract individual baseline times from parsed_baselines
        baselines = record.get("parsed_baselines", {})
        if mode in baselines:
            record["target_wall_clock_s"] = baselines[mode].get("wall_clock_s", record.get("wall_clock_s", 0.0))
            record["target_gen_s"] = baselines[mode].get("gen_s", 0.0)
            record["target_prefill_s"] = baselines[mode].get("prefill_s", 0.0)
        else:
            record["target_wall_clock_s"] = record.get("wall_clock_s", 0.0)
        save_run(EXPERIMENT_ID, SLUG, run_id, record)
        all_records.append(record)
        print(f"ok={record['ok']} wc={record['target_wall_clock_s']:.2f}s")

    if dry_run:
        return

    # Override wall_clock_s with extracted baseline times before aggregation
    for rec in all_records:
        if rec.get("target_wall_clock_s"):
            rec["wall_clock_s"] = rec["target_wall_clock_s"]

    agg = aggregate_experiment(EXPERIMENT_ID, SLUG, all_records)

    depth_speedups = {}
    for depth in DEPTHS:
        for model in MODELS:
            for mode in ["SIG", "SIG+CompSIG"]:
                sig_vals = [r["target_wall_clock_s"] for r in all_records
                            if r["depth"] == depth and r["model_size"] == model
                            and r["mode"] == mode and r.get("ok") and r.get("target_wall_clock_s", 0) > 0]
                app_vals = [r["target_wall_clock_s"] for r in all_records
                            if r["depth"] == depth and r["model_size"] == model
                            and r["mode"] == "AppLoop" and r.get("ok") and r.get("target_wall_clock_s", 0) > 0]
                if sig_vals and app_vals:
                    s_mean, s_lo, s_hi = bootstrap_ratio_ci(app_vals, sig_vals)
                    key = f"D{depth}_{model}_{mode}"
                    depth_speedups[key] = {
                        "speedup_mean": round(s_mean, 4),
                        "speedup_ci95_lo": round(s_lo, 4),
                        "speedup_ci95_hi": round(s_hi, 4),
                    }

    agg["depth_speedups"] = depth_speedups

    analysis_path = RESULTS_DIR / f"{EXPERIMENT_ID.lower()}_{SLUG}_analysis.json"
    with open(analysis_path, "w", encoding="utf-8") as f:
        json.dump(agg, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n[{EXPERIMENT_ID}] Results saved to {analysis_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=EXPERIMENT_ID)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    run_experiment(dry_run=args.dry_run)
