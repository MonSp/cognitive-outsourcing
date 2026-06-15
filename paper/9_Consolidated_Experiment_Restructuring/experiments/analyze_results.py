"""
Paper 9 — Results Analyzer and Figure Generator
=================================================
Reads all experiment result JSON files from results/ and produces:
  - Cross-experiment comparison tables
  - Figure specifications (matplotlib)
  - Reconciliation with prior paper data

Usage:
  python analyze_results.py                  # Analyze all available results
  python analyze_results.py --exp exp1       # Analyze only EXP-1
  python analyze_results.py --figures        # Generate figures
"""

import argparse, sys, os, json, math
from pathlib import Path
from typing import Dict, List, Any, Optional

sys.path.insert(0, os.path.dirname(__file__))
from common import RESULTS_DIR, FIGURES_DIR, mean_std, bootstrap_ci

PRIOR_PAPER_DATA = {
    "paper1_sig_speedup_4B": {"value": 3.85, "source": "Paper 1, FA-normalized 4.71x"},
    "paper4_sig_speedup_4B": {"value": 2.54, "n": 3, "steps": 32, "source": "Paper 4, n=3"},
    "paper5_sig_speedup_4B": {"value": 3.50, "n": 5, "steps": 35, "source": "Paper 5, llama-server"},
    "paper4_crossover": {"value": "1.5-2.0B", "source": "Paper 4, 2-point estimate"},
    "paper5_crossover": {"value": "~0.7B", "source": "Paper 5, linear interpolation"},
    "paper1_crossover": {"value": "~1.0B", "source": "Paper 1, FA-normalized"},
    "paper5_rho_mtp": {"value": 1.239, "source": "Paper 5, native MTP parallel verification"},
    "paper8_secmh_pre_scripted_delta": {"value": -0.141, "source": "Paper 8, full injection"},
    "paper8_secmh_agent_driven_delta": {"value": +0.101, "source": "Paper 8, noisy agent-driven"},
}


def load_experiment_results(exp_id: str) -> Optional[Dict[str, Any]]:
    pattern = f"{exp_id.lower()}_*_analysis.json"
    files = list(RESULTS_DIR.glob(pattern))
    if not files:
        return None
    with open(files[0], "r", encoding="utf-8") as f:
        return json.load(f)


def load_all_run_records(exp_id: str, slug: str) -> List[Dict[str, Any]]:
    pattern = f"{exp_id.lower()}_{slug}_run_*.json"
    files = sorted(RESULTS_DIR.glob(pattern))
    records = []
    for f in files:
        with open(f, "r", encoding="utf-8") as fh:
            records.append(json.load(fh))
    return records


def build_speedup_reconciliation_table():
    print("\n" + "=" * 80)
    print("TABLE: Cross-Paper Speedup Reconciliation (4B Kitchen)")
    print("=" * 80)
    print(f"{'Source':<25} {'Speedup':>10} {'n':>5} {'Steps':>7} {'Notes':<30}")
    print("-" * 80)
    rows = [
        ("Paper 1 (raw)", "3.85x", "10", "9 scenarios", "FA-normalized: 4.71x"),
        ("Paper 4", "2.54x", "3", "32", "Subprocess, quantized Q4_K_M"),
        ("Paper 5 (SIG)", "3.50x", "5", "35", "llama-server, prompt caching"),
        ("Paper 5 (SIG+MTP)", "4.52x", "5", "35", "Native MTP parallel verification"),
    ]
    for source, speedup, n, steps, notes in rows:
        print(f"{source:<25} {speedup:>10} {n:>5} {steps:>7} {notes:<30}")

    exp1 = load_experiment_results("EXP-1")
    if exp1 and "speedup_analysis" in exp1:
        sa = exp1["speedup_analysis"]
        print(f"{'EXP-9 (this work)':<25} {sa['speedup_mean']:.2f}x{'':<4} {'10':>5} {'35':>7} "
              f"CI=[{sa['speedup_ci95_lo']:.2f}, {sa['speedup_ci95_hi']:.2f}]")
    else:
        print(f"{'EXP-9 (this work)':<25} {'TBD':>10} {'10':>5} {'35':>7} {'Not yet executed':<30}")
    print()


def build_crossover_reconciliation_table():
    print("\n" + "=" * 80)
    print("TABLE: Cross-Paper Crossover Point Reconciliation")
    print("=" * 80)
    print(f"{'Source':<25} {'Crossover':>12} {'Method':<30} {'Notes':<20}")
    print("-" * 80)
    rows = [
        ("Paper 1", "~1.0B", "FA-normalized boundary", "0.8B inverts at 2x FA"),
        ("Paper 4", "1.5-2.0B", "Direct 2-point (0.8B, 4B)", "0.8B tied at 1.00x"),
        ("Paper 5", "~0.7B", "Linear interpolation 0.8B-4B", "SIG-only"),
    ]
    for source, crossover, method, notes in rows:
        print(f"{source:<25} {crossover:>12} {method:<30} {notes:<20}")

    exp2 = load_experiment_results("EXP-2")
    if exp2 and "crossover_fit" in exp2:
        for fit_type, fit_data in exp2["crossover_fit"].items():
            cb = fit_data.get("crossover_b", 0)
            print(f"{'EXP-9 ' + fit_type:<25} {cb:.2f}B{'':<7} {'Parametric 4-point fit':<30} "
                  f"SSE={fit_data.get('sse', 0):.4f}")
    else:
        print(f"{'EXP-9 (this work)':<25} {'TBD':>12} {'4-point parametric fit':<30} {'Not yet executed':<20}")
    print()


def build_paradigm_reconciliation_table():
    print("\n" + "=" * 80)
    print("TABLE: Evaluation Paradigm Effect Reconciliation")
    print("=" * 80)
    print(f"{'Source':<25} {'Paradigm':<15} {'SECM-H ΔQ':>12} {'Notes':<30}")
    print("-" * 80)
    rows = [
        ("Paper 8 EXP-3/9/10", "Pre-scripted", "-0.141", "Full injection, 2B"),
        ("Paper 8 EXP-11", "Agent-driven", "+0.101", "Noisy, full injection, 2B"),
        ("Paper 8 EXP-12", "Agent-driven", "+0.122", "Clean, selective injection, 2B"),
    ]
    for source, paradigm, delta, notes in rows:
        print(f"{source:<25} {paradigm:<15} {delta:>12} {notes:<30}")
    print()


def generate_figure_specs():
    specs = {
        "F1_speedup_comparison": {
            "title": "SIG Speedup Under Unified Protocol (EXP-1)",
            "x": "Condition", "y": "Wall-Clock Time (s)",
            "type": "bar", "conditions": ["AppLoop", "SIG", "AppLoop-PC", "Sliding"],
        },
        "F3_crossover_curve": {
            "title": "SIG-vs-AppLoop Crossover (EXP-2)",
            "x": "Model Size (B params)", "y": "Speedup Factor",
            "type": "line+scatter", "hline": 1.0,
        },
        "F5_orthogonality_heatmap": {
            "title": "Composition Orthogonality Ratios (EXP-3)",
            "x": "Component", "y": "Model Size",
            "type": "heatmap", "metric": "rho",
        },
        "F7_paradigm_interaction": {
            "title": "Evaluation Paradigm × Noise Interaction (EXP-4)",
            "x": "Paradigm", "y": "Content Quality (Q_content)",
            "type": "grouped_bar", "group": "noise",
        },
        "F9_pareto_frontier": {
            "title": "Speedup-Quality-Memory Pareto Frontier (EXP-5)",
            "x": "Speedup (log)", "y": "Q_content", "z": "KV-Cache Tokens",
            "type": "3d_scatter",
        },
        "F10_coverage_vs_depth": {
            "title": "Information Coverage vs Chain Depth (EXP-5)",
            "x": "Chain Depth (steps)", "y": "Information Coverage",
            "type": "line", "models": ["0.8B", "2B", "4B"],
        },
    }

    spec_path = FIGURES_DIR / "figure_specifications.json"
    with open(spec_path, "w", encoding="utf-8") as f:
        json.dump(specs, f, indent=2)
    print(f"Figure specifications saved to {spec_path}")
    return specs


def main():
    parser = argparse.ArgumentParser(description="Paper 9 Results Analyzer")
    parser.add_argument("--exp", type=str, default=None, help="Analyze specific experiment")
    parser.add_argument("--figures", action="store_true", help="Generate figure specs")
    parser.add_argument("--tables", action="store_true", help="Print reconciliation tables")
    args = parser.parse_args()

    if args.figures:
        generate_figure_specs()

    if args.tables or not args.figures:
        build_speedup_reconciliation_table()
        build_crossover_reconciliation_table()
        build_paradigm_reconciliation_table()

    if args.exp:
        exp_id = args.exp.upper()
        result = load_experiment_results(exp_id)
        if result:
            print(f"\n{exp_id} Results:")
            print(json.dumps(result, indent=2, ensure_ascii=False, default=str)[:2000])
        else:
            print(f"No results found for {exp_id}")

    if not args.exp and not args.figures and not args.tables:
        print("\n[INFO] Use --tables for reconciliation tables, --figures for figure specs, "
              "--exp EXP-N for specific experiment results.")


if __name__ == "__main__":
    main()
