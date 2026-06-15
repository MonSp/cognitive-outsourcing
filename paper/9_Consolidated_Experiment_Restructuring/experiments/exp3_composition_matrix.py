"""
EXP-3: Composition Matrix
==========================
Maps composition behavior of SIG with MTP, SECM-H, DiskKVCache, CompSIG.
Resolves Contradiction 6 (AppLoop-PC behavior) and characterizes RQ3.

8 configs x 2 models (0.8B, 4B) x 10 runs = 160 runs
"""

import argparse, sys, os, json
sys.path.insert(0, os.path.dirname(__file__))
from common import (
    collect_run, save_run, aggregate_experiment,
    randomized_run_order, bootstrap_ratio_ci, orthogonality_ratio,
    RESULTS_DIR
)

EXPERIMENT_ID = "EXP-3"
SLUG = "composition_matrix"
N_RUNS = 10
KITCHEN_STEPS = 35
MODELS = ["0.8B", "4B"]

CONFIGS = [
    ("C1_AppLoop",           "AppLoop"),
    ("C2_SIG",               "SIG"),
    ("C3_SIG_MTP",           "SIG+MTP"),
    ("C4_SIG_CompSIG",       "SIG+CompSIG"),
    ("C5_SIG_DiskKV",        "SIG+DiskKV"),
    ("C6_SIG_MTP_CompSIG",   "SIG+MTP+CompSIG"),
    ("C7_SIG_SECMH",         "SIG+SECM-H"),
    ("C8_FourWay",            "SIG+MTP+CompSIG+DiskKV"),
]


def make_conditions():
    conds = []
    for config_id, _ in CONFIGS:
        for model in MODELS:
            conds.append(f"{config_id}_{model}")
    return conds


SUPPORTED_CONFIGS = {"C1_AppLoop", "C2_SIG"}

def get_extra_args(config_id):
    if config_id not in SUPPORTED_CONFIGS:
        return None
    return None


def get_evaluation_mode(config_id):
    if "SECMH" in config_id:
        return "agent-driven"
    return "pre-scripted"


def run_experiment(dry_run=False):
    conditions = make_conditions()
    order = randomized_run_order(EXPERIMENT_ID, conditions, N_RUNS)
    all_records = []

    print(f"[{EXPERIMENT_ID}] {len(order)} runs total")

    for idx, (condition, run_id) in enumerate(order):
        parts = condition.rsplit("_", 1)
        config_id = parts[0]
        model = parts[1]
        print(f"  [{idx+1}/{len(order)}] {condition} run {run_id}...", end=" ", flush=True)
        if dry_run:
            print("DRY RUN")
            continue

        extra = get_extra_args(config_id)
        if extra == "SKIP" or (config_id not in SUPPORTED_CONFIGS and extra is None):
            print(f"SKIP (unsupported)")
            continue
        eval_mode = get_evaluation_mode(config_id)
        record = collect_run(
            experiment_id=EXPERIMENT_ID,
            condition=condition,
            model=model,
            run_id=run_id,
            task="kitchen",
            kitchen_steps=KITCHEN_STEPS,
            extra_args=extra,
            evaluation_mode=eval_mode,
        )
        baselines = record.get("parsed_baselines", {})
        sig_bl = baselines.get("SIG", {})
        app_bl = baselines.get("AppLoop", {})
        record["sig_wall_clock_s"] = sig_bl.get("wall_clock_s", 0.0)
        record["app_wall_clock_s"] = app_bl.get("wall_clock_s", 0.0)
        record["config_id"] = config_id
        save_run(EXPERIMENT_ID, SLUG, run_id, record)
        all_records.append(record)
        print(f"ok={record['ok']}")

    if dry_run:
        print(f"[{EXPERIMENT_ID}] Dry run complete.")
        return

    agg = aggregate_experiment(EXPERIMENT_ID, SLUG, all_records)

    speedups = {}
    for cond_key, _ in CONFIGS:
        for model in MODELS:
            key = f"{cond_key}_{model}"
            sig_vals = [r["sig_wall_clock_s"] for r in all_records
                        if r["condition"] == key and r.get("ok") and r["sig_wall_clock_s"] > 0]
            app_vals = [r["app_wall_clock_s"] for r in all_records
                        if r["condition"] == key and r.get("ok") and r["app_wall_clock_s"] > 0]
            if sig_vals and app_vals:
                s_mean, s_lo, s_hi = bootstrap_ratio_ci(app_vals, sig_vals)
                speedups[key] = {
                    "speedup_mean": round(s_mean, 4),
                    "speedup_ci95_lo": round(s_lo, 4),
                    "speedup_ci95_hi": round(s_hi, 4),
                }
    agg["speedups"] = speedups

    analysis_path = RESULTS_DIR / f"{EXPERIMENT_ID.lower()}_{SLUG}_analysis.json"
    with open(analysis_path, "w", encoding="utf-8") as f:
        json.dump(agg, f, indent=2, ensure_ascii=False, default=str)
    print(f"[{EXPERIMENT_ID}] Results saved to {analysis_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=EXPERIMENT_ID)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    run_experiment(dry_run=args.dry_run)
