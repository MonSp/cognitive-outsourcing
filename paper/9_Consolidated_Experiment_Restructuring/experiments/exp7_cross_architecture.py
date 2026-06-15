"""
EXP-7: Cross-Architecture Replication
=======================================
Verifies SIG speedup generalizes across architectures.
Replicates Paper 3 under unified protocol.

9 conditions x 5 runs = 45 runs
"""

import argparse, sys, os, json
sys.path.insert(0, os.path.dirname(__file__))
from common import (
    run_subprocess, save_run, aggregate_experiment,
    randomized_run_order, bootstrap_ratio_ci,
    MODEL_PATHS, CROSS_ARCH_MODELS, RESULTS_DIR
)

EXPERIMENT_ID = "EXP-7"
SLUG = "cross_architecture"
N_RUNS = 5
KITCHEN_STEPS = 35

ALL_MODELS = {**MODEL_PATHS, **CROSS_ARCH_MODELS}

CONDITIONS = [
    ("AppLoop_qwen4B",    "AppLoop",    "4B"),
    ("SIG_qwen4B",        "SIG",        "4B"),
    ("AppLoop_gemma",     "AppLoop",    "gemma"),
    ("SIG_gemma",         "SIG",        "gemma"),
    ("AppLoop_nemotron",  "AppLoop",    "nemotron"),
    ("SIG_nemotron",      "SIG",        "nemotron"),
]


def get_model_path(model_key):
    return ALL_MODELS.get(model_key, MODEL_PATHS.get(model_key, ""))


def get_extra_args(mode):
    if mode == "Batch-SIG":
        return "SKIP"
    return None


def run_experiment(dry_run=False):
    cond_ids = [c[0] for c in CONDITIONS]
    order = randomized_run_order(EXPERIMENT_ID, cond_ids, N_RUNS)
    all_records = []
    cond_lookup = {c[0]: c for c in CONDITIONS}

    print(f"[{EXPERIMENT_ID}] {len(order)} runs total")

    for idx, (cond_id, run_id) in enumerate(order):
        _, mode, model_key = cond_lookup[cond_id]
        model_path = get_model_path(model_key)
        print(f"  [{idx+1}/{len(order)}] {cond_id} run {run_id}...", end=" ", flush=True)
        if dry_run:
            print("DRY RUN")
            continue

        extra = get_extra_args(mode)
        if extra == "SKIP":
            print(f"SKIP (unsupported)")
            continue
        result = run_subprocess("kitchen", model_path, KITCHEN_STEPS, 60, extra)
        record = {
            "experiment_id": EXPERIMENT_ID,
            "condition": cond_id,
            "model": model_key,
            "mode": mode,
            "run_id": run_id,
            "wall_clock_s": result["elapsed"],
            "ok": result["ok"],
        }
        from common import parse_kitchen_metrics
        metrics = parse_kitchen_metrics(result["stdout"])
        if metrics:
            record["parsed_baselines"] = metrics
            bl_name = "Batch-SIG" if mode == "Batch-SIG" else mode
            bl = metrics.get(bl_name, metrics.get("SIG", {}))
            record["target_wall_clock_s"] = bl.get("wall_clock_s", result["elapsed"])
        else:
            record["target_wall_clock_s"] = result["elapsed"]
        save_run(EXPERIMENT_ID, SLUG, run_id, record)
        all_records.append(record)
        print(f"ok={record['ok']} wc={record['target_wall_clock_s']:.2f}s")

    if dry_run:
        return

    for rec in all_records:
        rec["wall_clock_s"] = rec.get("target_wall_clock_s", rec["wall_clock_s"])

    agg = aggregate_experiment(EXPERIMENT_ID, SLUG, all_records)

    speedups = {}
    for _, mode, model_key in CONDITIONS:
        if mode == "AppLoop":
            continue
        sig_vals = [r["wall_clock_s"] for r in all_records
                    if r["model"] == model_key and r["mode"] == mode and r.get("ok")]
        app_vals = [r["wall_clock_s"] for r in all_records
                    if r["model"] == model_key and r["mode"] == "AppLoop" and r.get("ok")]
        if sig_vals and app_vals:
            s_mean, s_lo, s_hi = bootstrap_ratio_ci(app_vals, sig_vals)
            speedups[f"{mode}_{model_key}"] = {
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
