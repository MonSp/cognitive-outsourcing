#!/usr/bin/env python3
"""Paper 6 Major Revision — CUDA-optimized End-to-End Migration PoC.

回应评审意见 #1: 原 Exp5 使用 CPU numpy 做 extract, 结果过于悲观。
本脚本用 Exp1 GPU 实测数据重构 CUDA 优化的 PoC 全链路, 包含:

  Table 9b-revised: CUDA PoC vs CPU PoC vs Local Prefill (多带宽)
  Jitter sensitivity: CUDA 优化 extract 下的 Monte Carlo 抖动分析
  Multi-tenant break-even: 10Gbps 共享下的设备数上限
  Feasibility domain: (token_count × bandwidth) 组合中 HYBRID 胜出域

数据来源:
  Exp1 GPU: gpu_reshape, gpu_fp16_total, gpu_d2h, actual_kv (RTX 4070 SUPER)
  Exp4 local prefill: authoritative prefill times
  Exp5 CPU PoC: CPU numpy PoC 全链路

用法:
  conda activate sig_bench
  python exp_paper6_cuda_poc.py
"""

import json
import os
import time

import numpy as np

TOKEN_COUNTS = [128, 256, 512, 1024, 2048, 4096]
BANDWIDTHS_GBPS = [0.5, 1.0, 3.0, 10.0]

GPU_DATA = {
    128:  {"gpu_reshape_ms": 14.44, "gpu_fp16_total_ms": 14.44, "gpu_d2h_ms": 3.01,  "actual_kv_mb": 10},
    256:  {"gpu_reshape_ms": 0.0,   "gpu_fp16_total_ms": 0.0,   "gpu_d2h_ms": 7.14,  "actual_kv_mb": 20},
    512:  {"gpu_reshape_ms": 1.51,  "gpu_fp16_total_ms": 1.51,  "gpu_d2h_ms": 7.76,  "actual_kv_mb": 40},
    1024: {"gpu_reshape_ms": 2.71,  "gpu_fp16_total_ms": 2.71,  "gpu_d2h_ms": 9.60,  "actual_kv_mb": 80},
    2048: {"gpu_reshape_ms": 8.35,  "gpu_fp16_total_ms": 8.35,  "gpu_d2h_ms": 19.36, "actual_kv_mb": 160},
    4096: {"gpu_reshape_ms": 25.68, "gpu_fp16_total_ms": 28.19, "gpu_d2h_ms": 28.56, "actual_kv_mb": 320},
}

CPU_POC_DATA = {
    128:  {"extract_ms": 63.6,  "translate_ms": 5.0,   "transfer_ms_3g": 28.0,  "translate_back_ms": 3.5,  "restore_ms": 2.0,  "total_ms": 102.1},
    256:  {"extract_ms": 134.2, "translate_ms": 7.0,   "transfer_ms_3g": 55.9,  "translate_back_ms": 7.0,  "restore_ms": 4.0,  "total_ms": 208.2},
    512:  {"extract_ms": 270.4, "translate_ms": 14.2,  "transfer_ms_3g": 111.9, "translate_back_ms": 14.1, "restore_ms": 8.1,  "total_ms": 418.7},
    1024: {"extract_ms": 539.8, "translate_ms": 33.6,  "transfer_ms_3g": 223.7, "translate_back_ms": 32.5, "restore_ms": 18.5, "total_ms": 848.1},
    2048: {"extract_ms": 1088.4,"translate_ms": 249.9, "transfer_ms_3g": 447.4, "translate_back_ms": 63.2, "restore_ms": 27.5, "total_ms": 1876.3},
    4096: {"extract_ms": 2166.9,"translate_ms": 1125.1,"transfer_ms_3g": 894.8, "translate_back_ms": 131.3,"restore_ms": 52.2, "total_ms": 4370.3},
}

LOCAL_PREFILL_TIMES_S = {
    128: 0.019,
    256: 0.0329,
    512: 0.061,
    1024: 0.1603,
    2048: 0.3091,
    4096: 0.651,
}

CUDA_RESTORE_MS = 0.5


def transfer_ms(kv_mb, bw_gbps):
    return (kv_mb * 8) / (bw_gbps * 1e3) * 1000


def build_cuda_poc(n_tokens, bw_gbps):
    g = GPU_DATA[n_tokens]
    kv_mb = g["actual_kv_mb"]

    cuda_extract_ms = g["gpu_fp16_total_ms"] + g["gpu_d2h_ms"]
    cuda_translate_ms = 0.0
    cuda_transfer_ms = transfer_ms(kv_mb, bw_gbps)
    cuda_translate_back_ms = 0.0
    cuda_restore_ms = CUDA_RESTORE_MS
    cuda_total_ms = cuda_extract_ms + cuda_translate_ms + cuda_transfer_ms + \
                    cuda_translate_back_ms + cuda_restore_ms

    local_prefill_ms = LOCAL_PREFILL_TIMES_S[n_tokens] * 1000
    verdict = "HYBRID" if cuda_total_ms < local_prefill_ms else "EDGE"

    return {
        "n_tokens": n_tokens,
        "bandwidth_gbps": bw_gbps,
        "kv_size_mb": kv_mb,
        "cuda_extract_ms": round(cuda_extract_ms, 2),
        "cuda_gpu_reshape_ms": g["gpu_reshape_ms"],
        "cuda_gpu_d2h_ms": g["gpu_d2h_ms"],
        "cuda_translate_ms": round(cuda_translate_ms, 2),
        "cuda_transfer_ms": round(cuda_transfer_ms, 2),
        "cuda_translate_back_ms": round(cuda_translate_back_ms, 2),
        "cuda_restore_ms": round(cuda_restore_ms, 2),
        "cuda_total_ms": round(cuda_total_ms, 2),
        "local_prefill_ms": round(local_prefill_ms, 2),
        "verdict": verdict,
    }


def build_cpu_poc(n_tokens, bw_gbps):
    c = CPU_POC_DATA[n_tokens]
    kv_mb = GPU_DATA[n_tokens]["actual_kv_mb"]
    cpu_extract_ms = c["extract_ms"]
    cpu_translate_ms = c["translate_ms"]
    cpu_transfer_ms = transfer_ms(kv_mb, bw_gbps)
    cpu_translate_back_ms = c["translate_back_ms"]
    cpu_restore_ms = c["restore_ms"]
    cpu_total_ms = cpu_extract_ms + cpu_translate_ms + cpu_transfer_ms + \
                   cpu_translate_back_ms + cpu_restore_ms

    local_prefill_ms = LOCAL_PREFILL_TIMES_S[n_tokens] * 1000
    verdict = "HYBRID" if cpu_total_ms < local_prefill_ms else "EDGE"

    return {
        "n_tokens": n_tokens,
        "bandwidth_gbps": bw_gbps,
        "kv_size_mb": kv_mb,
        "cpu_extract_ms": round(cpu_extract_ms, 2),
        "cpu_translate_ms": round(cpu_translate_ms, 2),
        "cpu_transfer_ms": round(cpu_transfer_ms, 2),
        "cpu_translate_back_ms": round(cpu_translate_back_ms, 2),
        "cpu_restore_ms": round(cpu_restore_ms, 2),
        "cpu_total_ms": round(cpu_total_ms, 2),
        "local_prefill_ms": round(local_prefill_ms, 2),
        "verdict": verdict,
    }


def compute_table9b_revised():
    print("=" * 90)
    print("  Table 9b-revised: CUDA-optimized vs CPU PoC vs Local Prefill")
    print("=" * 90)

    table = {}
    for bw in BANDWIDTHS_GBPS:
        print(f"\n  --- 带宽 {bw} Gbps ---")
        print(f"  {'Tokens':<8} {'CUDA(ms)':<12} {'CPU(ms)':<12} {'LocalPF(ms)':<14} "
              f"{'CUDA/HYBRID':<14} {'CPU/HYBRID':<14} {'Speedup':<10}")
        print(f"  {'-'*8} {'-'*12} {'-'*12} {'-'*14} {'-'*14} {'-'*14} {'-'*10}")

        bw_key = str(bw)
        table[bw_key] = {}
        for n in TOKEN_COUNTS:
            cuda = build_cuda_poc(n, bw)
            cpu = build_cpu_poc(n, bw)
            speedup = cpu["cpu_total_ms"] / cuda["cuda_total_ms"] if cuda["cuda_total_ms"] > 0 else 0

            table[bw_key][str(n)] = {
                "cuda_total_ms": cuda["cuda_total_ms"],
                "cuda_verdict": cuda["verdict"],
                "cpu_total_ms": cpu["cpu_total_ms"],
                "cpu_verdict": cpu["verdict"],
                "local_prefill_ms": cuda["local_prefill_ms"],
                "speedup_over_cpu": round(speedup, 1),
                "cuda_breakdown": {
                    "extract_ms": cuda["cuda_extract_ms"],
                    "transfer_ms": cuda["cuda_transfer_ms"],
                    "restore_ms": cuda["cuda_restore_ms"],
                },
                "cpu_breakdown": {
                    "extract_ms": cpu["cpu_extract_ms"],
                    "translate_ms": cpu["cpu_translate_ms"],
                    "transfer_ms": cpu["cpu_transfer_ms"],
                    "translate_back_ms": cpu["cpu_translate_back_ms"],
                    "restore_ms": cpu["cpu_restore_ms"],
                },
            }

            print(f"  {n:<8} {cuda['cuda_total_ms']:<12.2f} {cpu['cpu_total_ms']:<12.2f} "
                  f"{cuda['local_prefill_ms']:<14.2f} {cuda['verdict']:<14} "
                  f"{cpu['verdict']:<14} {speedup:<10.1f}x")

    return table


def compute_jitter_sensitivity():
    print(f"\n{'=' * 90}")
    print("  Jitter Sensitivity: CUDA-optimized Extract + Monte Carlo Transfer")
    print("=" * 90)

    n_samples = 10000
    rng = np.random.default_rng(42)

    results_by_bw = {}

    for bw_mean in BANDWIDTHS_GBPS:
        print(f"\n  --- Mean bandwidth = {bw_mean} Gbps ---")
        print(f"  {'Tokens':<8} {'σ=0.1':<10} {'σ=0.3':<10} {'σ=0.5':<10} "
              f"{'σ=1.0':<10} {'σ=2.0':<10}")
        print(f"  {'-'*8} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")

        sigmas = [0.1, 0.3, 0.5, 1.0, 2.0]
        bw_key = str(bw_mean)
        results_by_bw[bw_key] = {}

        for n in TOKEN_COUNTS:
            g = GPU_DATA[n]
            kv_mb = g["actual_kv_mb"]
            cuda_non_transfer_ms = g["gpu_fp16_total_ms"] + g["gpu_d2h_ms"] + CUDA_RESTORE_MS
            local_prefill_ms = LOCAL_PREFILL_TIMES_S[n] * 1000

            row = {"n_tokens": n, "cuda_non_transfer_ms": round(cuda_non_transfer_ms, 2),
                   "local_prefill_ms": round(local_prefill_ms, 2), "by_sigma": {}}

            row_str = f"  {n:<8}"
            for sigma in sigmas:
                bw_samples = rng.normal(bw_mean, sigma, n_samples)
                bw_samples = np.maximum(bw_samples, 0.01)
                transfer_samples = (kv_mb * 8) / (bw_samples * 1e3) * 1000
                total_samples = cuda_non_transfer_ms + transfer_samples
                p_hybrid = float(np.mean(total_samples < local_prefill_ms))
                mean_total = float(np.mean(total_samples))
                p99_total = float(np.percentile(total_samples, 99))

                row["by_sigma"][str(sigma)] = {
                    "p_hybrid_wins": round(p_hybrid, 4),
                    "mean_total_ms": round(mean_total, 2),
                    "p99_total_ms": round(p99_total, 2),
                }
                row_str += f" {p_hybrid:<10.1%}"

            results_by_bw[bw_key][str(n)] = row
            print(row_str)

    return results_by_bw


def compute_multitenant_breakeven():
    print(f"\n{'=' * 90}")
    print("  Multi-tenant Break-even: Devices Sharing 10 Gbps (CUDA-optimized)")
    print("=" * 90)

    total_bw = 10.0
    n_device_values = [1, 2, 3, 4, 5, 8, 10, 15, 20, 50]

    print(f"\n  总带宽: {total_bw} Gbps")
    print(f"  {'N_dev':<8} {'PerDev(Gbps)':<14} {'Tokens':<8} {'CUDA_T(ms)':<14} "
          f"{'LocalPF(ms)':<14} {'Verdict':<10}")
    print(f"  {'-'*8} {'-'*14} {'-'*8} {'-'*14} {'-'*14} {'-'*10}")

    results = {}

    for n_dev in n_device_values:
        per_bw = total_bw / n_dev
        dev_key = str(n_dev)
        results[dev_key] = {"per_device_bw_gbps": round(per_bw, 3), "by_tokens": {}}

        for n in TOKEN_COUNTS:
            cuda = build_cuda_poc(n, per_bw)
            results[dev_key]["by_tokens"][str(n)] = {
                "cuda_total_ms": cuda["cuda_total_ms"],
                "local_prefill_ms": cuda["local_prefill_ms"],
                "verdict": cuda["verdict"],
            }
            print(f"  {n_dev:<8} {per_bw:<14.3f} {n:<8} {cuda['cuda_total_ms']:<14.2f} "
                  f"{cuda['local_prefill_ms']:<14.2f} {cuda['verdict']:<10}")

    print(f"\n  --- Break-even 设备数 (CUDA-optimized, per token count) ---")
    break_even = {}
    for n in TOKEN_COUNTS:
        be_n = None
        for nd in range(1, 200):
            pbw = total_bw / nd
            c = build_cuda_poc(n, pbw)
            if c["verdict"] == "EDGE":
                be_n = nd
                break
        if be_n is None:
            be_n = ">200"
            be_bw = "<0.05"
        else:
            be_bw = f"{total_bw / be_n:.3f}"
        break_even[str(n)] = {"max_hybrid_devices": be_n if isinstance(be_n, str) else be_n - 1,
                              "break_even_per_device_bw_gbps": be_bw}
        print(f"    {n} tokens: max {be_n if isinstance(be_n, str) else be_n - 1} devices "
              f"(break-even at {be_bw} Gbps/dev)")

    return {"multi_tenant_table": results, "break_even": break_even}


def compute_feasibility_domain():
    print(f"\n{'=' * 90}")
    print("  Feasibility Domain: CUDA-optimized HYBRID Wins Over Local Prefill")
    print("=" * 90)

    fine_bw = [0.5, 1.0, 2.0, 3.0, 5.0, 7.5, 10.0, 20.0, 50.0, 100.0]

    print(f"\n  {'BW(Gbps)':<12}", end="")
    for n in TOKEN_COUNTS:
        print(f" {n:<10}", end="")
    print()
    print(f"  {'-'*12}", end="")
    for _ in TOKEN_COUNTS:
        print(f" {'-'*10}", end="")
    print()

    domain = {}
    for bw in fine_bw:
        row = {}
        print(f"  {bw:<12}", end="")
        for n in TOKEN_COUNTS:
            c = build_cuda_poc(n, bw)
            marker = "HYBRID" if c["verdict"] == "HYBRID" else "EDGE"
            row[str(n)] = {
                "verdict": marker,
                "cuda_total_ms": c["cuda_total_ms"],
                "local_prefill_ms": c["local_prefill_ms"],
                "margin_ms": round(c["local_prefill_ms"] - c["cuda_total_ms"], 2),
            }
            print(f" {marker:<10}", end="")
        print()
        domain[str(bw)] = row

    print(f"\n  --- 理论带宽阈值 (CUDA_total = local_prefill) ---")
    thresholds = {}
    for n in TOKEN_COUNTS:
        g = GPU_DATA[n]
        kv_mb = g["actual_kv_mb"]
        cuda_non_transfer_ms = g["gpu_fp16_total_ms"] + g["gpu_d2h_ms"] + CUDA_RESTORE_MS
        local_prefill_ms = LOCAL_PREFILL_TIMES_S[n] * 1000
        margin_ms = local_prefill_ms - cuda_non_transfer_ms
        if margin_ms > 0:
            threshold_bw = (kv_mb * 8) / margin_ms
            thresholds[str(n)] = {
                "cuda_non_transfer_ms": round(cuda_non_transfer_ms, 2),
                "local_prefill_ms": round(local_prefill_ms, 2),
                "budget_for_transfer_ms": round(margin_ms, 2),
                "threshold_gbps": round(threshold_bw, 2),
            }
            print(f"    {n} tokens: extract={cuda_non_transfer_ms:.2f}ms, "
                  f"localPF={local_prefill_ms:.2f}ms, "
                  f"transfer budget={margin_ms:.2f}ms, "
                  f"threshold={threshold_bw:.2f} Gbps")
        else:
            thresholds[str(n)] = {
                "cuda_non_transfer_ms": round(cuda_non_transfer_ms, 2),
                "local_prefill_ms": round(local_prefill_ms, 2),
                "budget_for_transfer_ms": round(margin_ms, 2),
                "threshold_gbps": None,
                "note": "CUDA overhead alone exceeds local prefill",
            }
            print(f"    {n} tokens: extract={cuda_non_transfer_ms:.2f}ms > "
                  f"localPF={local_prefill_ms:.2f}ms → IMPOSSIBLE")

    return {"feasibility_domain": domain, "bandwidth_thresholds": thresholds}


def compute_improvement_summary():
    print(f"\n{'=' * 90}")
    print("  Summary: CPU PoC → CUDA PoC Improvement")
    print("=" * 90)

    for bw in BANDWIDTHS_GBPS:
        print(f"\n  --- {bw} Gbps ---")
        print(f"  {'Tokens':<8} {'CPU_PoC':<12} {'CUDA_PoC':<12} {'Speedup':<10} "
              f"{'CPU_V':<8} {'CUDA_V':<8} {'LocalPF':<10}")
        print(f"  {'-'*8} {'-'*12} {'-'*12} {'-'*10} {'-'*8} {'-'*8} {'-'*10}")
        for n in TOKEN_COUNTS:
            cuda = build_cuda_poc(n, bw)
            cpu = build_cpu_poc(n, bw)
            sp = cpu["cpu_total_ms"] / cuda["cuda_total_ms"] if cuda["cuda_total_ms"] > 0 else 0
            print(f"  {n:<8} {cpu['cpu_total_ms']:<12.2f} {cuda['cuda_total_ms']:<12.2f} "
                  f"{sp:<10.1f}x {cpu['verdict']:<8} {cuda['verdict']:<8} "
                  f"{cuda['local_prefill_ms']:<10.2f}")


def main():
    print("Paper 6 — CUDA-optimized End-to-End Migration PoC Analysis")
    print(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"GPU: RTX 4070 SUPER, Model: Qwen3.5-4B-Q4_K_M")
    print(f"Restore estimate: {CUDA_RESTORE_MS} ms (CPU→GPU memcpy)")

    table9b = compute_table9b_revised()
    jitter = compute_jitter_sensitivity()
    multi_tenant = compute_multitenant_breakeven()
    feasibility = compute_feasibility_domain()
    compute_improvement_summary()

    all_results = {
        "model": "Qwen3.5-4B-Q4_K_M",
        "gpu": "RTX 4070 SUPER",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "cuda_restore_estimate_ms": CUDA_RESTORE_MS,
        "note": ("CUDA-optimized PoC replaces CPU numpy extract (50-2100ms) with "
                 "GPU reshape + D2H (17-57ms). Network transfer remains the dominant bottleneck."),
        "table_9b_revised": table9b,
        "jitter_sensitivity": jitter,
        "multi_tenant_breakeven": multi_tenant,
        "feasibility_domain": feasibility,
    }

    os.makedirs("data/paper6_review", exist_ok=True)
    out_path = "data/paper6_review/paper6_review_cuda_optimized.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n结果已保存: {out_path}")


if __name__ == "__main__":
    main()
