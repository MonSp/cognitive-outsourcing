"""
EXP-4: Paradigm Comparison
============================
Quantifies evaluation paradigm effect (pre-scripted vs agent-driven).
Resolves Contradiction 4: SECM-H negative vs positive results.

3 modes x 2 paradigms x 2 noise levels x 10 runs = 120 runs
Model: Qwen3.5-2B
"""

import argparse, sys, os, json
sys.path.insert(0, os.path.dirname(__file__))
from common import (
    collect_run, save_run, aggregate_experiment,
    randomized_run_order, bootstrap_ratio_ci, welch_t_test, cohens_d,
    RESULTS_DIR
)

EXPERIMENT_ID = "EXP-4"
SLUG = "paradigm_comparison"
N_RUNS = 10
KITCHEN_STEPS = 35
MODEL = "4B"

CONDITIONS = [
    ("PS_clean_AppLoop",    "pre-scripted", "clean",  "AppLoop"),
    ("PS_clean_SIG",        "pre-scripted", "clean",  "SIG"),
    ("PS_clean_SECMH",      "pre-scripted", "clean",  "SIG+SECM-H"),
    ("PS_noisy_AppLoop",    "pre-scripted", "noisy",  "AppLoop"),
    ("PS_noisy_SIG",        "pre-scripted", "noisy",  "SIG"),
    ("PS_noisy_SECMH",      "pre-scripted", "noisy",  "SIG+SECM-H"),
    ("AD_clean_AppLoop",    "agent-driven", "clean",  "AppLoop"),
    ("AD_clean_SIG",        "agent-driven", "clean",  "SIG"),
    ("AD_clean_SECMH",      "agent-driven", "clean",  "SIG+SECM-H"),
    ("AD_noisy_AppLoop",    "agent-driven", "noisy",  "AppLoop"),
    ("AD_noisy_SIG",        "agent-driven", "noisy",  "SIG"),
    ("AD_noisy_SECMH",      "agent-driven", "noisy",  "SIG+SECM-H"),
]


def get_extra_args(paradigm, noise, mode):
    if paradigm == "agent-driven" or noise == "noisy" or mode == "SIG+SECM-H":
        return "SKIP"
    return None


def run_experiment(dry_run=False):
    cond_ids = [c[0] for c in CONDITIONS]
    order = randomized_run_order(EXPERIMENT_ID, cond_ids, N_RUNS)
    all_records = []
    cond_lookup = {c[0]: c for c in CONDITIONS}

    print(f"[{EXPERIMENT_ID}] {len(order)} runs total")

    for idx, (cond_id, run_id) in enumerate(order):
        paradigm, noise, mode = cond_lookup[cond_id][1], cond_lookup[cond_id][2], cond_lookup[cond_id][3]
        print(f"  [{idx+1}/{len(order)}] {cond_id} run {run_id}...", end=" ", flush=True)
        if dry_run:
            print("DRY RUN")
            continue

        extra = get_extra_args(paradigm, noise, mode)
        if extra == "SKIP":
            print(f"SKIP (unsupported)")
            continue
        record = collect_run(
            experiment_id=EXPERIMENT_ID,
            condition=cond_id,
            model=MODEL,
            run_id=run_id,
            task="kitchen",
            kitchen_steps=KITCHEN_STEPS,
            extra_args=extra,
            evaluation_mode=paradigm,
        )
        record["paradigm"] = paradigm
        record["noise"] = noise
        record["mode"] = mode
        save_run(EXPERIMENT_ID, SLUG, run_id, record)
        all_records.append(record)
        print(f"ok={record['ok']}")

    if dry_run:
        return

    for rec in all_records:
        baselines = rec.get("parsed_baselines", {})
        bl_name = rec.get("mode", "SIG")
        if bl_name == "SIG+SECM-H":
            bl_name = "SIG"
        bl = baselines.get(bl_name, {})
        rec["wall_clock_s"] = bl.get("wall_clock_s", rec.get("wall_clock_s", 0.0))

    agg = aggregate_experiment(EXPERIMENT_ID, SLUG, all_records)

    analysis = {"paradigm_effects": {}}
    for paradigm in ["pre-scripted", "agent-driven"]:
        for noise in ["clean", "noisy"]:
            sig_vals = [r["wall_clock_s"] for r in all_records
                        if r["paradigm"] == paradigm and r["noise"] == noise
                        and r["mode"] == "SIG" and r.get("ok")]
            app_vals = [r["wall_clock_s"] for r in all_records
                        if r["paradigm"] == paradigm and r["noise"] == noise
                        and r["mode"] == "AppLoop" and r.get("ok")]
            if sig_vals and app_vals:
                s_mean, s_lo, s_hi = bootstrap_ratio_ci(app_vals, sig_vals)
                analysis["paradigm_effects"][f"{paradigm}_{noise}"] = {
                    "speedup_mean": round(s_mean, 4),
                    "speedup_ci95_lo": round(s_lo, 4),
                    "speedup_ci95_hi": round(s_hi, 4),
                }

    agg["analysis"] = analysis

    analysis_path = RESULTS_DIR / f"{EXPERIMENT_ID.lower()}_{SLUG}_analysis.json"
    with open(analysis_path, "w", encoding="utf-8") as f:
        json.dump(agg, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n[{EXPERIMENT_ID}] Results saved to {analysis_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=EXPERIMENT_ID)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    run_experiment(dry_run=args.dry_run)
