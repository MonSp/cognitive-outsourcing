import json
import math
import numpy as np
from scipy import stats
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"

def load_json(name):
    with open(DATA_DIR / name) as f:
        return json.load(f)

def cohens_d(x, y):
    nx, ny = len(x), len(y)
    vx = np.var(x, ddof=1)
    vy = np.var(y, ddof=1)
    pooled_std = math.sqrt(((nx - 1) * vx + (ny - 1) * vy) / (nx + ny - 2))
    if pooled_std == 0:
        return 0.0
    return (np.mean(x) - np.mean(y)) / pooled_std

def paired_test_report(label, a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    n = len(a)
    t_stat, p_val = stats.ttest_rel(a, b)
    d = cohens_d(a, b)
    diff = a - b
    mean_diff = np.mean(diff)
    se_diff = np.std(diff, ddof=1) / math.sqrt(n)
    t_crit = stats.t.ppf(0.975, df=n - 1)
    ci_lo = mean_diff - t_crit * se_diff
    ci_hi = mean_diff + t_crit * se_diff
    sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
    return {
        "label": label,
        "n": n,
        "median_a": float(np.median(a)),
        "median_b": float(np.median(b)),
        "mean_a": float(np.mean(a)),
        "mean_b": float(np.mean(b)),
        "mean_diff": float(mean_diff),
        "ci_95": (float(ci_lo), float(ci_hi)),
        "t": float(t_stat),
        "p": float(p_val),
        "d": float(d),
        "sig": sig,
    }

def print_exp1_table(results_8b, results_4b):
    print("=" * 130)
    print("EXP 1: LATENCY COMPARISON — PAIRED T-TESTS (real per-session data)")
    print("=" * 130)
    header = f"{'Comparison':<42} {'N':>3} {'Mdn_A':>8} {'Mdn_B':>8} {'Δ (ms)':>9} {'95% CI':>20} {'t':>7} {'p':>9} {'d':>7} {'Sig':>5}"
    print(header)
    print("-" * 130)
    for label, model_label, results in [
        ("0.8B", "Qwen3.5-0.8B", results_8b),
        ("4B", "Qwen3.5-4B", results_4b),
    ]:
        print(f"  [{model_label}]")
        for r in results:
            ci_str = f"[{r['ci_95'][0]:+.2f}, {r['ci_95'][1]:+.2f}]"
            p_str = f"{r['p']:.4e}" if r['p'] < 0.001 else f"{r['p']:.4f}"
            print(
                f"  {r['label']:<40} {r['n']:>3} {r['median_a']:>8.2f} {r['median_b']:>8.2f} "
                f"{r['mean_diff']:>+9.2f} {ci_str:>20} {r['t']:>7.2f} {p_str:>9} "
                f"{r['d']:>7.2f} {r['sig']:>5}"
            )
        print()


def print_exp3_table(decomp_8b, decomp_4b):
    print("=" * 130)
    print("EXP 3: STATE SIZE DECOMPOSITION")
    print("=" * 130)
    for label, decomp in [("Qwen3.5-0.8B", decomp_8b), ("Qwen3.5-4B", decomp_4b)]:
        print(f"\n  [{label}]")
        print(f"  {'Prefix':<15} {'Tokens':>7} {'Total (MB)':>11} {'NZ%':>6} {'Zero%':>6} {'Compr.Raw':>11} {'Compr.NZ':>10} {'B/token':>8}")
        print(f"  {'-'*80}")
        for name, info in decomp.get("prefixes", {}).items():
            if isinstance(info, dict) and "state_mb" in info:
                print(
                    f"  {name:<15} {info['n_tokens']:>7} {info['state_mb']:>11.2f} "
                    f"{info['nonzero_ratio_pct']:>5.1f}% {info['zero_padding_pct']:>5.1f}% "
                    f"{info['compressed_raw_bytes']:>11,} {info['compressed_nonzero_bytes']:>10,} "
                    f"{info['bytes_per_token']:>8.0f}"
                )
            elif isinstance(info, dict) and "bytes" in info:
                print(f"  {name:<15} {'':>7} {info['mb']:>11.2f} (incremental)")
    print()


def print_exp4_table(compression_results):
    print("=" * 130)
    print("EXP 4: COMPRESSION ANALYSIS (with zero-byte analysis)")
    print("=" * 130)

    print(f"\n  {'Model':<20} {'Raw (B)':>12} {'Compressed':>12} {'NonZero':>12} {'Zero%':>7} {'Ratio(raw)':>12} {'Ratio(eff)':>12}")
    print(f"  {'-'*90}")
    for r in compression_results:
        print(
            f"  {r['model']:<20} {r['uncompressed']:>12,} {r['compressed']:>12,} "
            f"{r['nonzero_bytes']:>12,} {r['zero_padding_pct']:>6.1f}% "
            f"{r['ratio_raw']:>11.1f}x {r['ratio_effective']:>11.1f}x"
        )

    print(f"\n  NOTE: Raw compression ratios are dominated by zero-padding in unused KV-Cache slots.")
    print(f"  Effective ratio measures actual data compression (nonzero bytes / compressed bytes).")

    print(f"\n  Save/Load Latency:")
    print(f"  {'Model':<20} {'Save(plain)':>12} {'Save(zlib)':>12} {'Save Δ%':>9} {'Load(plain)':>12} {'Load(zlib)':>12} {'Load Δ%':>9}")
    print(f"  {'-'*90}")
    for r in compression_results:
        save_pct = (1 - r['save_comp'] / r['save_uncomp']) * 100 if r['save_uncomp'] > 0 else 0
        load_pct = (1 - r['load_comp'] / r['load_uncomp']) * 100 if r['load_uncomp'] > 0 else 0
        print(
            f"  {r['model']:<20} {r['save_uncomp']:>12.2f} {r['save_comp']:>12.2f} "
            f"{save_pct:>+8.1f}% {r['load_uncomp']:>12.2f} {r['load_comp']:>12.2f} "
            f"{load_pct:>+8.1f}%"
        )


def print_exp5_table(long_ctx_8b, long_ctx_4b):
    print("=" * 130)
    print("EXP 5: LONG-CONTEXT SCALING")
    print("=" * 130)
    for label, data in [("Qwen3.5-0.8B", long_ctx_8b), ("Qwen3.5-4B", long_ctx_4b)]:
        print(f"\n  [{label}]  n_ctx={data['n_ctx']}")
        print(f"  {'Ratio':>6} {'Tokens':>8} {'Cold(ms)':>10} {'Save(ms)':>10} {'Load(ms)':>10} {'State(MB)':>10} {'NZ%':>6} {'Speedup':>8} {'Winner':>10}")
        print(f"  {'-'*86}")
        for s in data.get("scales", []):
            winner = "Disk" if s["disk_wins"] else "Cold"
            print(
                f"  {s['ctx_ratio']:>6.0%} {s['n_tokens']:>8} {s['cold_ms']:>10.2f} "
                f"{s['save_ms']:>10.2f} {s['load_ms']:>10.2f} {s['state_mb']:>10.2f} "
                f"{s['nonzero_pct']:>5.1f}% {s['speedup_vs_cold']:>7.2f}x {winner:>10}"
            )
        crossover = None
        scales = data.get("scales", [])
        for i in range(len(scales) - 1):
            if not scales[i]["disk_wins"] and scales[i + 1]["disk_wins"]:
                crossover = scales[i + 1]["ctx_ratio"]
                break
        if crossover:
            print(f"  → Crossover at ~{crossover:.0%} context utilization")
        elif scales and all(s["disk_wins"] for s in scales):
            print(f"  → DiskKVCache wins at all tested ratios")
        elif scales and not any(s["disk_wins"] for s in scales):
            print(f"  → Cold start wins at all tested ratios (state I/O > token eval)")


def print_breakeven_table(breakeven_results):
    print("\n" + "=" * 130)
    print("CROSS-PROCESS BREAK-EVEN ANALYSIS")
    print("=" * 130)
    print()
    print("  Formula: C_save + (N-1)*C_load < N * T_cold")
    print("  Solved:  N > (C_save - C_load) / (T_cold - C_load)")
    print()
    print(f"  {'Model':<25} {'T_cold (ms)':>12} {'C_save (ms)':>12} {'C_load (ms)':>12} {'N_breakeven':>12} {'N_ceiled':>10}")
    print(f"  {'-'*83}")
    for r in breakeven_results:
        n_str = f"{r['n_breakeven']:.1f}" if r['n_breakeven'] != float('inf') else "∞"
        nc_str = str(r['n_ceiled']) if r['n_ceiled'] != float('inf') else "∞"
        print(
            f"  {r['model']:<25} {r['t_cold']:>12.2f} {r['c_save']:>12.2f} "
            f"{r['c_load']:>12.2f} {n_str:>12} {nc_str:>10}"
        )

    print(f"\n  Per-session cost breakdown for N = 1..10:")
    print(f"  {'Model':<25} {'N':>4} {'Cold_Total (ms)':>16} {'Disk_Total (ms)':>17} {'Δ (ms)':>10} {'Winner':>14}")
    print(f"  {'-'*86}")
    for r in breakeven_results:
        t_cold = r['t_cold']
        c_save = r['c_save']
        c_load = r['c_load']
        for n in range(1, 11):
            cold_total = n * t_cold
            disk_total = c_save + (n - 1) * c_load
            diff = disk_total - cold_total
            winner = "DiskKVCache" if diff < 0 else "Cold Start" if diff > 0 else "Tie"
            print(
                f"  {r['model']:<25} {n:>4} {cold_total:>16.2f} {disk_total:>17.2f} "
                f"{diff:>+10.2f} {winner:>14}"
            )
        print()


def main():
    data_8b = load_json("kvcache_persist_Qwen3.5-0.8B-Q4_K_M.json")
    data_4b = load_json("kvcache_persist_Qwen3.5-4B-Q4_K_M.json")

    # ---- Exp 1: Real per-session paired t-tests ----
    has_per_session = "cold_start_per_session_ms" in data_8b.get("exp1_latency", {})

    if has_per_session:
        def extract_exp1(d):
            lat = d["exp1_latency"]
            return (
                lat["cold_start_per_session_ms"],
                lat.get("pc_restore_per_session_ms", [lat.get("in_memory_pc_avg_ms", 0)] * lat.get("n_sessions", 10)),
                lat["disk_save_per_session_ms"],
                lat["disk_load_per_session_ms"],
            )

        cold_8b, pc_8b, save_8b, load_8b = extract_exp1(data_8b)
        cold_4b, pc_4b, save_4b, load_4b = extract_exp1(data_4b)

        comparisons_8b = [
            paired_test_report("Cold Start vs DiskKVCache Save", cold_8b, save_8b),
            paired_test_report("Cold Start vs DiskKVCache Load", cold_8b, load_8b),
            paired_test_report("Cold Start vs In-Memory PC", cold_8b, pc_8b),
            paired_test_report("DiskKVCache Save vs Load", save_8b, load_8b),
        ]

        comparisons_4b = [
            paired_test_report("Cold Start vs DiskKVCache Save", cold_4b, save_4b),
            paired_test_report("Cold Start vs DiskKVCache Load", cold_4b, load_4b),
            paired_test_report("Cold Start vs In-Memory PC", cold_4b, pc_4b),
            paired_test_report("DiskKVCache Save vs Load", save_4b, load_4b),
        ]
    else:
        print("⚠ No per-session data found in JSON. Run exp_kvcache_persist.py to generate real data.")
        return

    print_exp1_table(comparisons_8b, comparisons_4b)

    # ---- Exp 3: State decomposition ----
    if "exp3_decomposition" in data_8b:
        print_exp3_table(data_8b["exp3_decomposition"], data_4b["exp3_decomposition"])

    # ---- Exp 4: Compression with zero-byte analysis ----
    def build_compression(d):
        c = d["exp4_compression"]
        if "compression_ratio_raw" in c:
            return {
                "model": d["model"].replace(".gguf", ""),
                "uncompressed": c["uncompressed_bytes"],
                "compressed": c["compressed_bytes"],
                "nonzero_bytes": c.get("nonzero_bytes", c["uncompressed_bytes"]),
                "zero_padding_pct": c.get("zero_padding_pct", 0),
                "ratio_raw": c["compression_ratio_raw"],
                "ratio_effective": c.get("compression_ratio_effective", c["compression_ratio_raw"]),
                "save_uncomp": c["save_uncompressed_ms"],
                "save_comp": c["save_compressed_ms"],
                "load_uncomp": c["load_uncompressed_ms"],
                "load_comp": c["load_compressed_ms"],
            }
        else:
            return {
                "model": d["model"].replace(".gguf", ""),
                "uncompressed": c["uncompressed_bytes"],
                "compressed": c["compressed_bytes"],
                "nonzero_bytes": c["uncompressed_bytes"],
                "zero_padding_pct": 0,
                "ratio_raw": c["compression_ratio"],
                "ratio_effective": c["compression_ratio"],
                "save_uncomp": c["save_uncompressed_ms"],
                "save_comp": c["save_compressed_ms"],
                "load_uncomp": c["load_uncompressed_ms"],
                "load_comp": c["load_compressed_ms"],
            }

    compression_results = [build_compression(data_8b), build_compression(data_4b)]
    print_exp4_table(compression_results)

    # ---- Exp 5: Long-context scaling ----
    if "exp5_long_context" in data_8b:
        print_exp5_table(data_8b["exp5_long_context"], data_4b["exp5_long_context"])

    # ---- Break-even from Exp 1 real data ----
    print()
    print("=" * 130)
    print("BREAK-EVEN ANALYSIS (from real Exp 1 data)")
    print("=" * 130)

    breakeven_results = []
    for label, data in [("Qwen3.5-0.8B", data_8b), ("Qwen3.5-4B", data_4b)]:
        lat = data["exp1_latency"]
        if "disk_save_per_session_ms" in lat:
            t_cold = lat["cold_start_avg_ms"]
            c_save = lat["disk_save_avg_ms"]
            c_load = lat["disk_load_avg_ms"]

            if t_cold > c_load:
                n_breakeven = (c_save - c_load) / (t_cold - c_load)
            else:
                n_breakeven = float('inf')

            n_ceiled = math.ceil(n_breakeven) if n_breakeven != float('inf') and n_breakeven > 0 else (1 if n_breakeven > 0 else float('inf'))

            breakeven_results.append({
                "model": label,
                "t_cold": t_cold,
                "c_save": c_save,
                "c_load": c_load,
                "n_breakeven": n_breakeven,
                "n_ceiled": n_ceiled,
            })

    if breakeven_results:
        print_breakeven_table(breakeven_results)

    # ---- Notes ----
    print("=" * 130)
    print("NOTES")
    print("=" * 130)
    if has_per_session:
        n_sess = data_8b["exp1_latency"].get("n_sessions", "?")
        print(f"  - Per-session data is real (N={n_sess}), not synthesized.")
        print(f"  - Paired t-tests use the same sessions as matched pairs.")
    else:
        print("  - ⚠ Data uses aggregated means only. Re-run exp_kvcache_persist.py for real per-session data.")
    print(f"  - Cohen's d: 0.2=small, 0.5=medium, 0.8=large effect size.")
    print(f"  - CI uses t-critical for df=N-1 (adaptive to sample size).")
    print(f"  - Break-even formula: C_save + (N-1)*C_load < N * T_cold")


if __name__ == "__main__":
    main()
