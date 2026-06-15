"""
EXP-6: Generation Causation
==============================
Attributes generation time differences to prompt format vs KV-cache mechanism.
Resolves Contradiction 3: Paper 1 KV-cache expansion vs Papers 2-4 prompt-format artifact.

4 conditions x 10 runs = 40 runs
Model: Qwen3.5-4B
"""

import argparse, sys, os, json
sys.path.insert(0, os.path.dirname(__file__))
from common import (
    collect_run, save_run, aggregate_experiment,
    randomized_run_order, bootstrap_ci, welch_t_test, cohens_d,
    mean_std, RESULTS_DIR
)

EXPERIMENT_ID = "EXP-6"
SLUG = "generation_causation"
N_RUNS = 10
KITCHEN_STEPS = 35
MODEL = "4B"

CONDITIONS = ["AppLoop_standard", "SIG_standard", "AppLoop_lengthmatched", "SIG_lengthmatched"]


def get_extra_args(condition):
    if "lengthmatched" in condition:
        return "SKIP"
    return None


def run_experiment(dry_run=False):
    order = randomized_run_order(EXPERIMENT_ID, CONDITIONS, N_RUNS)
    all_records = []

    print(f"[{EXPERIMENT_ID}] {len(order)} runs total")

    for idx, (condition, run_id) in enumerate(order):
        print(f"  [{idx+1}/{len(order)}] {condition} run {run_id}...", end=" ", flush=True)
        if dry_run:
            print("DRY RUN")
            continue

        extra = get_extra_args(condition)
        if extra == "SKIP":
            print(f"SKIP (unsupported)")
            continue
        record = collect_run(
            experiment_id=EXPERIMENT_ID,
            condition=condition,
            model=MODEL,
            run_id=run_id,
            task="kitchen",
            kitchen_steps=KITCHEN_STEPS,
            extra_args=extra,
        )
        baselines = record.get("parsed_baselines", {})
        if "AppLoop" in condition:
            bl = baselines.get("AppLoop", {})
        else:
            bl = baselines.get("SIG", {})
        record["target_wall_clock_s"] = bl.get("wall_clock_s", record.get("wall_clock_s", 0.0))
        record["target_gen_s"] = bl.get("gen_s", 0.0)
        record["target_prefill_s"] = bl.get("prefill_s", 0.0)
        save_run(EXPERIMENT_ID, SLUG, run_id, record)
        all_records.append(record)
        print(f"ok={record['ok']} wc={record['target_wall_clock_s']:.2f}s")

    if dry_run:
        return

    for rec in all_records:
        rec["wall_clock_s"] = rec.get("target_wall_clock_s", rec["wall_clock_s"])

    agg = aggregate_experiment(EXPERIMENT_ID, SLUG, all_records)

    analysis = {"decomposition": {}}
    for cond in CONDITIONS:
        vals = [r["target_gen_s"] for r in all_records if r["condition"] == cond and r.get("ok")]
        pf_vals = [r["target_prefill_s"] for r in all_records if r["condition"] == cond and r.get("ok")]
        if vals:
            m, s = mean_std(vals)
            pf_m, pf_s = mean_std(pf_vals) if pf_vals else (0, 0)
            analysis["decomposition"][cond] = {
                "gen_time_mean": round(m, 4), "gen_time_std": round(s, 4),
                "prefill_time_mean": round(pf_m, 4), "prefill_time_std": round(pf_s, 4),
            }

    sig_std = [r["target_gen_s"] for r in all_records
               if r["condition"] == "SIG_standard" and r.get("ok")]
    app_std = [r["target_gen_s"] for r in all_records
               if r["condition"] == "AppLoop_standard" and r.get("ok")]
    if sig_std and app_std:
        t, p = welch_t_test(app_std, sig_std)
        d = cohens_d(app_std, sig_std)
        m_app, _ = mean_std(app_std)
        m_sig, _ = mean_std(sig_std)
        ratio = m_app / m_sig if m_sig > 0 else 0
        analysis["token_count_ratio"] = {
            "apploop_gen_mean": round(m_app, 4),
            "sig_gen_mean": round(m_sig, 4),
            "ratio": round(ratio, 4),
            "welch_p": round(p, 6),
            "cohens_d": round(d, 4),
        }
        print(f"\n[{EXPERIMENT_ID}] Gen time ratio (AppLoop/SIG): {ratio:.2f}x")

    agg["analysis"] = analysis

    analysis_path = RESULTS_DIR / f"{EXPERIMENT_ID.lower()}_{SLUG}_analysis.json"
    with open(analysis_path, "w", encoding="utf-8") as f:
        json.dump(agg, f, indent=2, ensure_ascii=False, default=str)
    print(f"[{EXPERIMENT_ID}] Results saved to {analysis_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=EXPERIMENT_ID)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    run_experiment(dry_run=args.dry_run)
