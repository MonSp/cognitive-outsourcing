"""
EXP-8: Agent-Driven Kitchen
=============================
Measures SIG speedup and quality under agent-driven evaluation.
Resolves Contradiction 4 (paradigm effect) with speedup focus.

3 modes x 2 noise x 2 models x 10 runs = 120 runs
"""

import argparse, sys, os, json
sys.path.insert(0, os.path.dirname(__file__))
from common import (
    collect_run, save_run, aggregate_experiment,
    randomized_run_order, bootstrap_ratio_ci, welch_t_test, cohens_d,
    RESULTS_DIR
)

EXPERIMENT_ID = "EXP-8"
SLUG = "agent_driven"
N_RUNS = 10
KITCHEN_STEPS = 35
MODELS = ["4B"]
MODES = ["AppLoop", "SIG"]
NOISE_LEVELS = ["clean"]


def make_conditions():
    conds = []
    for model in MODELS:
        for noise in NOISE_LEVELS:
            for mode in MODES:
                mode_slug = mode.replace("+", "_").replace("-", "_")
                conds.append(f"{model}_{noise}_{mode_slug}")
    return conds


def get_extra_args(mode, noise):
    return None


def run_experiment(dry_run=False):
    conditions = make_conditions()
    order = randomized_run_order(EXPERIMENT_ID, conditions, N_RUNS)
    all_records = []

    print(f"[{EXPERIMENT_ID}] {len(order)} runs total")

    for idx, (condition, run_id) in enumerate(order):
        parts = condition.split("_")
        model = parts[0]
        noise = parts[1]
        mode_slug = "_".join(parts[2:])
        mode = mode_slug.replace("_", "+") if "SECM" in mode_slug else mode_slug.replace("_", "-") if "SECM" not in mode_slug else mode_slug
        if mode_slug == "AppLoop":
            mode = "AppLoop"
        elif mode_slug == "SIG":
            mode = "SIG"
        elif "SECM" in mode_slug:
            mode = "SIG+SECM-H"

        print(f"  [{idx+1}/{len(order)}] {condition} run {run_id}...", end=" ", flush=True)
        if dry_run:
            print("DRY RUN")
            continue

        extra = get_extra_args(mode, noise)
        if extra == "SKIP":
            print(f"SKIP (unsupported)")
            continue
        record = collect_run(
            experiment_id=EXPERIMENT_ID,
            condition=condition,
            model=model,
            run_id=run_id,
            task="kitchen",
            kitchen_steps=KITCHEN_STEPS,
            extra_args=extra,
            evaluation_mode="agent-driven",
        )
        record["mode"] = mode
        record["noise"] = noise
        record["model_size"] = model

        baselines = record.get("parsed_baselines", {})
        bl_name = "AppLoop" if mode == "AppLoop" else "SIG"
        bl = baselines.get(bl_name, {})
        record["target_wall_clock_s"] = bl.get("wall_clock_s", record.get("wall_clock_s", 0.0))

        save_run(EXPERIMENT_ID, SLUG, run_id, record)
        all_records.append(record)
        print(f"ok={record['ok']} wc={record['target_wall_clock_s']:.2f}s")

    if dry_run:
        return

    for rec in all_records:
        rec["wall_clock_s"] = rec.get("target_wall_clock_s", rec["wall_clock_s"])

    agg = aggregate_experiment(EXPERIMENT_ID, SLUG, all_records)

    speedups = {}
    for model in MODELS:
        for noise in NOISE_LEVELS:
            sig_vals = [r["wall_clock_s"] for r in all_records
                        if r["model_size"] == model and r["noise"] == noise
                        and r["mode"] == "SIG" and r.get("ok")]
            app_vals = [r["wall_clock_s"] for r in all_records
                        if r["model_size"] == model and r["noise"] == noise
                        and r["mode"] == "AppLoop" and r.get("ok")]
            if sig_vals and app_vals:
                s_mean, s_lo, s_hi = bootstrap_ratio_ci(app_vals, sig_vals)
                speedups[f"{model}_{noise}"] = {
                    "speedup_mean": round(s_mean, 4),
                    "speedup_ci95_lo": round(s_lo, 4),
                    "speedup_ci95_hi": round(s_hi, 4),
                }

    agg["speedups"] = speedups

    analysis_path = RESULTS_DIR / f"{EXPERIMENT_ID.lower()}_{SLUG}_analysis.json"
    with open(analysis_path, "w", encoding="utf-8") as f:
        json.dump(agg, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n[{EXPERIMENT_ID}] Results saved to {analysis_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=EXPERIMENT_ID)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    run_experiment(dry_run=args.dry_run)
