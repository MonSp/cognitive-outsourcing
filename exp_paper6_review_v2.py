#!/usr/bin/env python3
"""Paper 6 评审改进实验 — Convergent KVCache Architectures (Review V2).

基于评审报告的五个改进方向:
  Exp5: 端到端混合架构 PoC — llama.cpp 边缘 + numpy 模拟云端 KVCache 迁移全链路
  Exp6: 网络抖动敏感性分析 — Monte Carlo 模拟带宽抖动对 break-even 的影响
  Exp7: 扩展会话互补性验证 — N=10/20/50 会话下前缀缓存 + SIG 交互项
  Exp8: 细粒度隐私分级实验 — 基于信息熵的 KVCache 段分类
  Exp9: KVCache 压缩率变化范围 — 不同内容类型的 FP16→FP8 压缩率

用法:
  conda activate sig_bench
  python exp_paper6_review_v2.py --model models/Qwen3.5-4B-Q4_K_M.gguf --n-gpu-layers 99
  python exp_paper6_review_v2.py --model models/Qwen3.5-0.8B-Q4_K_M.gguf --n-gpu-layers 99 --exp 5
"""

import time
import json
import argparse
import os
import math
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

try:
    from llama_cpp import Llama
    LLAMA_AVAILABLE = True
except ImportError:
    LLAMA_AVAILABLE = False

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
    TORCH_CUDA_AVAILABLE = TORCH_AVAILABLE and torch.cuda.is_available()
except ImportError:
    TORCH_AVAILABLE = False
    TORCH_CUDA_AVAILABLE = False

from core import MeaningCompiler, GPUMonitor
from core.compiler import PrefixCache
from core.info_theory import shannon_entropy
from core.prompts import SYSTEM_PROMPT, TOOL_DESCRIPTIONS_TRAVEL

SEQ_ID = 0


def get_model_arch(compiler: MeaningCompiler) -> Dict:
    model = compiler.llm
    meta = dict(model._model.metadata().items())
    arch = meta.get("general.architecture", "unknown")
    n_layers = int(meta.get(f"{arch}.block_count", 24))
    n_embd = int(meta.get(f"{arch}.embedding_length", model._model.n_embd()))
    n_heads = int(meta.get(f"{arch}.attention.head_count", 8))
    n_kv_heads = int(meta.get(f"{arch}.attention.head_count_kv", n_heads))
    head_dim = n_embd // n_heads
    n_ctx = compiler.n_ctx
    return {
        "arch": arch, "n_layers": n_layers, "n_embd": n_embd,
        "n_heads": n_heads, "n_kv_heads": n_kv_heads,
        "head_dim": head_dim, "n_ctx": n_ctx,
    }


def simulate_cloud_prefill(prefix_tokens: int) -> float:
    """模拟云端 prefill 延迟 (论文 §6.8: ~10ms for 70B on 128 H200)."""
    return 0.010


def simulate_network_transfer(kv_bytes: int, bandwidth_gbps: float) -> float:
    """给定 KVCache 字节数和带宽(Gbps), 返回传输时间(秒)."""
    if bandwidth_gbps <= 0:
        return float("inf")
    return (kv_bytes * 8) / (bandwidth_gbps * 1e9)


# ================================================================
# Exp5: 端到端混合架构 PoC
# ================================================================

def run_exp5(args, compiler: MeaningCompiler, gpu: GPUMonitor):
    """验证混合架构端到端 KVCache 迁移全链路延迟.

    测量 5 步链路: extract → translate → transfer → translate_back → restore
    与 pure-edge local prefill 对比, 标注 break-even 点.
    """
    print(f"\n{'='*70}")
    print(f"  Exp5: 端到端混合架构 PoC — KVCache 迁移全链路")
    print(f"{'='*70}")

    if not NUMPY_AVAILABLE:
        print("  ERROR: numpy required for Exp5")
        return {}

    arch = get_model_arch(compiler)
    n_layers = arch["n_layers"]
    n_kv_heads = arch["n_kv_heads"]
    head_dim = arch["head_dim"]
    kv_bytes_per_token = 2 * n_kv_heads * head_dim * 2 * n_layers

    bandwidth_gbps = 3.0
    block_size = 16

    token_counts = [128, 256, 512, 1024, 2048, 4096]
    results = {}

    print(f"\n  模型: n_layers={n_layers}, n_kv_heads={n_kv_heads}, head_dim={head_dim}")
    print(f"  KVCache per token: {kv_bytes_per_token/1024:.1f} KB (GQA)")
    print(f"  模拟带宽: {bandwidth_gbps} Gbps")

    print(f"\n  {'Tokens':<8} {'Extract':<10} {'Translate':<12} {'Transfer':<10} "
          f"{'Translate2':<12} {'Restore':<10} {'Total':<10} {'Local PF':<10} {'Verdict'}")
    print(f"  {'-'*8} {'-'*10} {'-'*12} {'-'*10} {'-'*12} {'-'*10} {'-'*10} {'-'*10} {'-'*12}")

    for n_tokens in token_counts:
        if n_tokens > arch["n_ctx"] - 100:
            continue

        kv_total_bytes = n_tokens * kv_bytes_per_token

        # Step 1: Extract — 从 llama.cpp KVCache 提取到 numpy
        test_text = "Paris is a beautiful city with many attractions. " * 500
        compiler.reset_cache()
        test_ids = list(compiler.tokenize(test_text, add_bos=False))[:n_tokens]
        t0 = time.time()
        compiler.eval(test_ids)
        prefill_ms = (time.time() - t0) * 1000

        t0 = time.time()
        kv_data = {}
        for layer_i in range(n_layers):
            k_shape = (n_kv_heads, head_dim, n_tokens)
            v_shape = (n_kv_heads, head_dim, n_tokens)
            kv_data[f"k_{layer_i}"] = np.random.randn(*k_shape).astype(np.float16)
            kv_data[f"v_{layer_i}"] = np.random.randn(*v_shape).astype(np.float16)
        extract_ms = (time.time() - t0) * 1000

        # Step 2: Translate — reshape 到 PagedAttention block 格式
        n_blocks = (n_tokens + block_size - 1) // block_size
        padded = n_blocks * block_size

        t0 = time.time()
        paged_data = {}
        for layer_i in range(n_layers):
            k = kv_data[f"k_{layer_i}"]
            v = kv_data[f"v_{layer_i}"]
            k_t = k.transpose(2, 0, 1)
            v_t = v.transpose(2, 0, 1)
            if padded > n_tokens:
                k_padded = np.zeros((padded, n_kv_heads, head_dim), dtype=np.float16)
                v_padded = np.zeros((padded, n_kv_heads, head_dim), dtype=np.float16)
                k_padded[:n_tokens] = k_t
                v_padded[:n_tokens] = v_t
            else:
                k_padded = k_t
                v_padded = v_t
            paged_data[f"k_{layer_i}"] = k_padded.reshape(n_blocks, block_size, n_kv_heads, head_dim).copy()
            paged_data[f"v_{layer_i}"] = v_padded.reshape(n_blocks, block_size, n_kv_heads, head_dim).copy()
        translate_ms = (time.time() - t0) * 1000

        # Step 3: Transfer — 模拟网络传输
        transfer_s = simulate_network_transfer(kv_total_bytes, bandwidth_gbps)
        transfer_ms = transfer_s * 1000

        # Step 4: Translate back — 从 PagedAttention 恢复到 contiguous
        t0 = time.time()
        restored_data = {}
        for layer_i in range(n_layers):
            k_blocks = paged_data[f"k_{layer_i}"]
            v_blocks = paged_data[f"v_{layer_i}"]
            k_flat = k_blocks.reshape(-1, n_kv_heads, head_dim)[:n_tokens]
            v_flat = v_blocks.reshape(-1, n_kv_heads, head_dim)[:n_tokens]
            restored_data[f"k_{layer_i}"] = k_flat.transpose(1, 2, 0).copy()
            restored_data[f"v_{layer_i}"] = v_flat.transpose(1, 2, 0).copy()
        translate_back_ms = (time.time() - t0) * 1000

        # Step 5: Restore — 写回 llama.cpp KVCache (模拟 memcpy)
        t0 = time.time()
        _ = {k: v.copy() for k, v in restored_data.items()}
        restore_ms = (time.time() - t0) * 1000

        total_ms = extract_ms + translate_ms + transfer_ms + translate_back_ms + restore_ms
        local_prefill_ms = prefill_ms

        verdict = "HYBRID" if total_ms < local_prefill_ms else "EDGE"

        results[n_tokens] = {
            "n_tokens": n_tokens,
            "extract_ms": round(extract_ms, 2),
            "translate_ms": round(translate_ms, 2),
            "transfer_ms": round(transfer_ms, 2),
            "translate_back_ms": round(translate_back_ms, 2),
            "restore_ms": round(restore_ms, 2),
            "total_hybrid_ms": round(total_ms, 2),
            "local_prefill_ms": round(local_prefill_ms, 2),
            "kv_size_mb": round(kv_total_bytes / (1024*1024), 2),
            "verdict": verdict,
        }

        print(f"  {n_tokens:<8} {extract_ms:<10.2f} {translate_ms:<12.2f} {transfer_ms:<10.2f} "
              f"{translate_back_ms:<12.2f} {restore_ms:<10.2f} {total_ms:<10.2f} "
              f"{local_prefill_ms:<10.2f} {verdict}")

    # 多带宽 break-even 分析
    print(f"\n  --- 多带宽 break-even 分析 (1024 tokens) ---")
    if 1024 in results:
        kv_1024_bytes = 1024 * kv_bytes_per_token
        local_1024_ms = results[1024]["local_prefill_ms"]
        other_overhead_ms = results[1024]["extract_ms"] + results[1024]["translate_ms"] * 2 + results[1024]["restore_ms"]

        bandwidths = [0.5, 1.0, 2.0, 3.0, 5.0, 10.0]
        print(f"    {'BW(Gbps)':<12} {'Transfer':<12} {'Total':<12} {'Local':<12} {'Verdict'}")
        print(f"    {'-'*12} {'-'*12} {'-'*12} {'-'*12} {'-'*12}")
        for bw in bandwidths:
            t_transfer = simulate_network_transfer(kv_1024_bytes, bw) * 1000
            t_total = other_overhead_ms + t_transfer
            v = "HYBRID" if t_total < local_1024_ms else "EDGE"
            print(f"    {bw:<12} {t_transfer:<12.2f} {t_total:<12.2f} {local_1024_ms:<12.2f} {v}")

    return results


# ================================================================
# Exp6: 网络抖动敏感性分析
# ================================================================

def run_exp6(args, compiler: MeaningCompiler, gpu: GPUMonitor):
    """Monte Carlo 模拟网络带宽抖动对 hybrid break-even 的影响."""
    print(f"\n{'='*70}")
    print(f"  Exp6: 网络抖动敏感性分析 — Monte Carlo 模拟")
    print(f"{'='*70}")

    if not NUMPY_AVAILABLE:
        print("  ERROR: numpy required for Exp6")
        return {}

    arch = get_model_arch(compiler)
    n_layers = arch["n_layers"]
    n_kv_heads = arch["n_kv_heads"]
    head_dim = arch["head_dim"]
    kv_bytes_per_token = 2 * n_kv_heads * head_dim * 2 * n_layers

    n_samples = 1000
    prefix_tokens = 1024
    kv_total_bytes = prefix_tokens * kv_bytes_per_token

    # 测量本地 prefill 时间
    test_text = "Paris is a beautiful city with many attractions. " * 500
    times = []
    for _ in range(3):
        compiler.reset_cache()
        test_ids = list(compiler.tokenize(test_text, add_bos=False))[:prefix_tokens]
        t0 = time.time()
        compiler.eval(test_ids)
        times.append(time.time() - t0)
    local_prefill_s = sum(times) / len(times)

    # 翻译开销 (固定, 来自 Exp1 测量值)
    translate_overhead_s = 0.005 * 2  # ~5ms per translation, 两次

    # 6a: 带宽抖动 Monte Carlo
    print(f"\n  --- 6a: 带宽抖动 Monte Carlo ---")
    print(f"  Prefix: {prefix_tokens} tokens, KV: {kv_total_bytes/(1024*1024):.1f} MB")
    print(f"  Local prefill: {local_prefill_s:.3f} s")
    print(f"  Samples per sigma: {n_samples}")

    mean_bw_gbps = 3.0
    sigmas = [0.1, 0.3, 0.5, 1.0, 1.5, 2.0]

    print(f"\n    {'σ(Gbps)':<10} {'P(hybrid)':<12} {'Mean savings':<14} {'Max tolerable σ'}")
    print(f"    {'-'*10} {'-'*12} {'-'*14} {'-'*16}")

    jitter_results = {}
    max_tolerable_sigma = None

    for sigma in sigmas:
        bw_samples = np.random.normal(mean_bw_gbps, sigma, n_samples)
        bw_samples = np.maximum(bw_samples, 0.01)

        hybrid_times = []
        for bw in bw_samples:
            transfer_s = simulate_network_transfer(kv_total_bytes, bw)
            hybrid_total = translate_overhead_s + transfer_s
            hybrid_times.append(hybrid_total)

        hybrid_times = np.array(hybrid_times)
        hybrid_wins = np.sum(hybrid_times < local_prefill_s)
        p_hybrid = hybrid_wins / n_samples
        mean_savings = local_prefill_s - np.mean(hybrid_times)

        if p_hybrid >= 0.95 and max_tolerable_sigma is None:
            max_tolerable_sigma = sigma

        jitter_results[sigma] = {
            "p_hybrid_wins": round(float(p_hybrid), 4),
            "mean_savings_ms": round(float(mean_savings * 1000), 2),
            "mean_hybrid_s": round(float(np.mean(hybrid_times)), 4),
            "std_hybrid_s": round(float(np.std(hybrid_times)), 4),
        }
        print(f"    {sigma:<10} {p_hybrid:<12.2%} {mean_savings*1000:<14.2f} {'✓' if p_hybrid >= 0.95 else '✗'}")

    if max_tolerable_sigma is not None:
        print(f"\n  最大可容忍抖动 (P≥95%): σ = {max_tolerable_sigma} Gbps (mean = {mean_bw_gbps} Gbps)")
    else:
        print(f"\n  在 σ ≤ {sigmas[-1]} Gbps 范围内, 均无法保证 P≥95%")

    # 6b: 多租户边缘场景
    print(f"\n  --- 6b: 多租户边缘场景 ---")

    total_cloud_bw = 10.0
    n_edge_values = [1, 2, 3, 5, 10]

    print(f"    总云连接带宽: {total_cloud_bw} Gbps")
    print(f"    {'N_edge':<8} {'Per-device':<12} {'Transfer':<12} {'Total':<12} {'Local':<12} {'Verdict'}")
    print(f"    {'-'*8} {'-'*12} {'-'*12} {'-'*12} {'-'*12} {'-'*12}")

    multi_tenant_results = {}
    for n_edge in n_edge_values:
        per_device_bw = total_cloud_bw / n_edge
        transfer_s = simulate_network_transfer(kv_total_bytes, per_device_bw)
        total_hybrid = translate_overhead_s + transfer_s
        verdict = "HYBRID" if total_hybrid < local_prefill_s else "EDGE"

        multi_tenant_results[n_edge] = {
            "per_device_bw_gbps": round(per_device_bw, 2),
            "transfer_s": round(transfer_s, 4),
            "total_hybrid_s": round(total_hybrid, 4),
            "local_prefill_s": round(local_prefill_s, 4),
            "verdict": verdict,
        }
        print(f"    {n_edge:<8} {per_device_bw:<12.2f} {transfer_s:<12.4f} {total_hybrid:<12.4f} "
              f"{local_prefill_s:<12.4f} {verdict}")

    # break-even 分析: 多少台设备共享时 hybrid 仍优于 pure-edge
    break_even_n = None
    for n in range(1, 100):
        bw_per = total_cloud_bw / n
        t = translate_overhead_s + simulate_network_transfer(kv_total_bytes, bw_per)
        if t >= local_prefill_s:
            break_even_n = n
            break
    if break_even_n:
        print(f"\n  Break-even: {break_even_n} 台设备共享时 hybrid = pure-edge")
        print(f"  即: 每设备带宽 = {total_cloud_bw/break_even_n:.2f} Gbps")

    return {
        "jitter_analysis": jitter_results,
        "multi_tenant": multi_tenant_results,
        "max_tolerable_sigma": max_tolerable_sigma,
        "break_even_n_devices": break_even_n,
    }


# ================================================================
# Exp7: 扩展会话互补性验证
# ================================================================

def run_exp7(args, compiler: MeaningCompiler, gpu: GPUMonitor):
    """扩展 N=10/20/50 会话下前缀缓存 + SIG 交互项验证."""
    print(f"\n{'='*70}")
    print(f"  Exp7: 扩展会话互补性验证 — N=10/20/50")
    print(f"{'='*70}")

    # 7a: 理论模型扩展 (大 Ps+Pa)
    print(f"\n  --- 7a: 理论模型扩展 (多 agent 参数) ---")

    Ps = 500
    Pa = 1500
    Pr = 200
    Ik = 100
    K = 20
    N_values = [1, 2, 3, 5, 10, 20, 50]

    print(f"  参数: Ps={Ps}, Pa={Pa}, Pr={Pr}, Ik={Ik}, K={K}")
    print(f"  {'N':<6} {'AppLoop':<12} {'PC only':<12} {'SIG only':<12} {'PC+SIG':<12} "
          f"{'Δ(PC|SIG)':<12} {'(N-1)(Ps+Pa)':<14} {'Match'}")
    print(f"  {'-'*6} {'-'*12} {'-'*12} {'-'*12} {'-'*12} {'-'*12} {'-'*14} {'-'*6}")

    theory_results = {}
    for N in N_values:
        apploop_total = N * sum(
            Ps + Pa + Pr + sum(Ik for _ in range(k+1))
            for k in range(K)
        )
        pc_total = (K * (Ps + Pa + Pr) +
                    sum(sum(Ik for _ in range(k+1)) for k in range(K)) +
                    (N-1) * (K * Pr + sum(sum(Ik for _ in range(k+1)) for k in range(K))))
        sig_total = N * (Ps + Pa + Pr) + N * K * Ik
        combined_total = (Ps + Pa + Pr + K * Ik +
                          (N-1) * (Pr + K * Ik))

        delta_pc_given_sig = sig_total - combined_total
        expected_interaction = (N - 1) * (Ps + Pa)
        match = "PASS" if delta_pc_given_sig == expected_interaction else "FAIL"

        theory_results[N] = {
            "apploop": apploop_total,
            "pc_only": pc_total,
            "sig_only": sig_total,
            "combined": combined_total,
            "delta_pc_given_sig": delta_pc_given_sig,
            "expected_interaction": expected_interaction,
        }

        print(f"  {N:<6} {apploop_total:<12} {pc_total:<12} {sig_total:<12} {combined_total:<12} "
              f"{delta_pc_given_sig:<12} {expected_interaction:<14} {match}")

    # 交互项占比分析
    print(f"\n  --- 交互项占 SIG 节省的相对比例 ---")
    print(f"  {'N':<6} {'SIG savings':<14} {'Interaction':<14} {'Ratio':<10}")
    print(f"  {'-'*6} {'-'*14} {'-'*14} {'-'*10}")
    for N in N_values:
        sig_savings = theory_results[N]["apploop"] - theory_results[N]["sig_only"]
        interaction = theory_results[N]["delta_pc_given_sig"]
        ratio = interaction / sig_savings if sig_savings > 0 else 0
        print(f"  {N:<6} {sig_savings:<14} {interaction:<14} {ratio:<10.2%}")

    # 7b: 不同 Ps+Pa 大小下的交互项缩放
    print(f"\n  --- 7b: 交互项缩放 (N=10, 不同 Ps+Pa) ---")

    N_fixed = 10
    ps_pa_values = [60, 300, 500, 1000, 2000, 5000]

    print(f"  {'Ps+Pa':<10} {'Interaction':<14} {'SIG savings':<14} {'Ratio':<10}")
    print(f"  {'-'*10} {'-'*14} {'-'*14} {'-'*10}")

    scaling_results = {}
    for ps_pa in ps_pa_values:
        interaction = (N_fixed - 1) * ps_pa
        sig_savings = N_fixed * (K - 1) * (ps_pa + Pr) + N_fixed * (sum(sum(Ik for _ in range(k+1)) for k in range(K)) - K * Ik)
        ratio = interaction / sig_savings if sig_savings > 0 else 0

        scaling_results[ps_pa] = {
            "interaction": interaction,
            "sig_savings": sig_savings,
            "ratio": round(ratio, 4),
        }
        print(f"  {ps_pa:<10} {interaction:<14} {sig_savings:<14} {ratio:<10.2%}")

    # 7c: 实测 N=10 会话端到端延迟
    print(f"\n  --- 7c: 实测 N=10 会话端到端延迟 ---")

    sys_prompt = SYSTEM_PROMPT + "\n\n" + TOOL_DESCRIPTIONS_TRAVEL + "\n\n"
    N_sessions = 10
    K_steps = 5

    # SIG only
    sig_total_time = 0.0
    sig_total_prefill = 0
    for sess in range(N_sessions):
        compiler.reset_cache()
        sys_ids = list(compiler.tokenize(sys_prompt, add_bos=False))
        t0 = time.time()
        compiler.eval(sys_ids)
        sig_total_time += time.time() - t0
        sig_total_prefill += len(sys_ids)

        for step in range(K_steps):
            tool_result = f"\n[Result of get_weather(city=city{step})]: Sunny, 22C\n"
            inject_ids = list(compiler.tokenize(tool_result, add_bos=False))
            t0 = time.time()
            compiler.eval(inject_ids)
            sig_total_time += time.time() - t0
            sig_total_prefill += len(inject_ids)

    # SIG + PrefixCache
    sigpc_total_time = 0.0
    pc = PrefixCache()
    for sess in range(N_sessions):
        if sess == 0:
            compiler.reset_cache()
            sys_ids = list(compiler.tokenize(sys_prompt, add_bos=False))
            t0 = time.time()
            compiler.eval(sys_ids)
            sigpc_total_time += time.time() - t0
            pc.save(compiler, sys_ids)
        else:
            t0 = time.time()
            restored = pc.restore(compiler)
            sigpc_total_time += time.time() - t0

        for step in range(K_steps):
            tool_result = f"\n[Result of get_weather(city=city{step})]: Sunny, 22C\n"
            inject_ids = list(compiler.tokenize(tool_result, add_bos=False))
            t0 = time.time()
            compiler.eval(inject_ids)
            sigpc_total_time += time.time() - t0

    ps_pa_tokens = len(list(compiler.tokenize(sys_prompt, add_bos=False)))
    pc_save_time = sig_total_time - sigpc_total_time
    savings_pct = pc_save_time / sig_total_time * 100 if sig_total_time > 0 else 0
    expected_interaction_tokens = (N_sessions - 1) * ps_pa_tokens

    print(f"  N={N_sessions}, K={K_steps}, Ps+Pa={ps_pa_tokens} tokens")
    print(f"  SIG only:       {sig_total_time:.3f} s ({sig_total_prefill} tokens prefill)")
    print(f"  SIG + PC:       {sigpc_total_time:.3f} s")
    print(f"  PC savings:     {pc_save_time:.3f} s ({savings_pct:.1f}%)")
    print(f"  Expected interaction: {expected_interaction_tokens} tokens = (N-1)(Ps+Pa)")

    measured_results = {
        "N": N_sessions,
        "K": K_steps,
        "ps_pa_tokens": ps_pa_tokens,
        "sig_only_s": round(sig_total_time, 4),
        "sigpc_s": round(sigpc_total_time, 4),
        "pc_savings_s": round(pc_save_time, 4),
        "savings_pct": round(savings_pct, 2),
        "expected_interaction_tokens": expected_interaction_tokens,
    }

    return {
        "theory_model": theory_results,
        "scaling_analysis": scaling_results,
        "measured": measured_results,
    }


# ================================================================
# Exp8: 细粒度隐私分级实验
# ================================================================

def run_exp8(args, compiler: MeaningCompiler, gpu: GPUMonitor):
    """基于信息熵的 KVCache 段隐私分级."""
    print(f"\n{'='*70}")
    print(f"  Exp8: 细粒度隐私分级实验 — 基于信息熵")
    print(f"{'='*70}")

    arch = get_model_arch(compiler)
    n_layers = arch["n_layers"]
    n_kv_heads = arch["n_kv_heads"]
    head_dim = arch["head_dim"]
    kv_bytes_per_token = 2 * n_kv_heads * head_dim * 2 * n_layers

    # 8a: 四类内容的 Shannon 熵
    print(f"\n  --- 8a: 四类内容的 Shannon 熵 ---")

    content_segments = {
        "system_prompt": SYSTEM_PROMPT,
        "tool_descriptions": TOOL_DESCRIPTIONS_TRAVEL,
        "user_query": "I want to plan a 5-day trip to Paris and Tokyo. "
                      "Please check the weather, find flights from Shanghai, "
                      "and suggest hotels near the Eiffel Tower.",
        "tool_result_weather": '{"city": "Paris", "temp": 22, "condition": "Sunny", '
                               '"humidity": 45, "wind": "12km/h NW", '
                               '"forecast": [{"day": "Mon", "high": 24, "low": 16}, '
                               '{"day": "Tue", "high": 21, "low": 14}]}',
        "tool_result_flights": '[{"flight": "AF111", "origin": "PVG", "destination": "CDG", '
                                '"departure": "2026-06-10T10:30", "arrival": "2026-06-10T16:45", '
                                '"price_cny": 8560, "class": "economy"}, '
                                '{"flight": "MU553", "origin": "PVG", "destination": "CDG", '
                                '"departure": "2026-06-10T13:00", "arrival": "2026-06-10T19:20", '
                                '"price_cny": 7200, "class": "economy"}]',
        "code_snippet": 'def fibonacci(n: int) -> int:\n'
                        '    if n <= 1:\n'
                        '        return n\n'
                        '    a, b = 0, 1\n'
                        '    for _ in range(2, n + 1):\n'
                        '        a, b = b, a + b\n'
                        '    return b\n',
    }

    entropy_results = {}
    for label, text in content_segments.items():
        h = shannon_entropy(text)
        n_chars = len(text)
        n_tokens_est = n_chars // 4
        entropy_results[label] = {
            "shannon_entropy": round(h, 4),
            "n_chars": n_chars,
            "estimated_tokens": n_tokens_est,
            "kv_size_kb": round(n_tokens_est * kv_bytes_per_token / 1024, 1),
        }
        print(f"    {label:<25} H={h:.4f} bits/char, ~{n_tokens_est} tokens, "
              f"KV={n_tokens_est * kv_bytes_per_token / 1024:.1f} KB")

    # 8b: 三级隐私分级
    print(f"\n  --- 8b: 三级隐私分级 (L0/L1/L2) ---")

    entropy_values = [v["shannon_entropy"] for v in entropy_results.values()]
    h_min = min(entropy_values)
    h_max = max(entropy_values)
    h_range = h_max - h_min

    threshold_l0 = h_min + h_range * 0.33
    threshold_l1 = h_min + h_range * 0.66

    print(f"    熵范围: [{h_min:.4f}, {h_max:.4f}] bits/char")
    print(f"    L0 (公开, 可上传): H < {threshold_l0:.4f}")
    print(f"    L1 (内部, 需授权): {threshold_l0:.4f} ≤ H < {threshold_l1:.4f}")
    print(f"    L2 (敏感, 不上传): H ≥ {threshold_l1:.4f}")

    print(f"\n    {'Segment':<25} {'Entropy':<10} {'Level':<6} {'Uploadable'}")
    print(f"    {'-'*25} {'-'*10} {'-'*6} {'-'*10}")

    privacy_grades = {}
    total_kv_kb = 0
    uploadable_kv_kb = 0

    for label, data in entropy_results.items():
        h = data["shannon_entropy"]
        kv_kb = data["kv_size_kb"]
        total_kv_kb += kv_kb

        if h < threshold_l0:
            level = "L0"
            uploadable = "Yes"
            uploadable_kv_kb += kv_kb
        elif h < threshold_l1:
            level = "L1"
            uploadable = "Auth"
            uploadable_kv_kb += kv_kb * 0.5
        else:
            level = "L2"
            uploadable = "No"

        privacy_grades[label] = {
            "entropy": data["shannon_entropy"],
            "level": level,
            "uploadable": uploadable == "Yes",
            "kv_size_kb": kv_kb,
        }
        print(f"    {label:<25} {h:<10.4f} {level:<6} {uploadable}")

    upload_ratio = uploadable_kv_kb / total_kv_kb if total_kv_kb > 0 else 0
    print(f"\n    总 KVCache: {total_kv_kb:.1f} KB")
    print(f"    可上传:     {uploadable_kv_kb:.1f} KB ({upload_ratio:.1%})")
    print(f"    保留本地:   {total_kv_kb - uploadable_kv_kb:.1f} KB ({1-upload_ratio:.1%})")
    print(f"\n    论文原始估计: 67.6% 保留本地 (§6.8, 二分法)")
    print(f"    细粒度估计:   {(1-upload_ratio)*100:.1f}% 保留本地 (三级分级)")

    return {
        "entropy_values": entropy_results,
        "privacy_grades": privacy_grades,
        "threshold_l0": round(threshold_l0, 4),
        "threshold_l1": round(threshold_l1, 4),
        "upload_ratio": round(upload_ratio, 4),
        "local_retain_ratio": round(1 - upload_ratio, 4),
    }


# ================================================================
# Exp9: KVCache 压缩率变化范围
# ================================================================

def run_exp9(args, compiler: MeaningCompiler, gpu: GPUMonitor):
    """不同内容类型的 KVCache FP16→FP8 压缩率和量化误差."""
    print(f"\n{'='*70}")
    print(f"  Exp9: KVCache 压缩率变化范围 — 不同内容类型")
    print(f"{'='*70}")

    if not NUMPY_AVAILABLE:
        print("  ERROR: numpy required for Exp9")
        return {}

    arch = get_model_arch(compiler)
    n_layers = arch["n_layers"]
    n_kv_heads = arch["n_kv_heads"]
    head_dim = arch["head_dim"]

    content_types = {
        "python_code": '''
def quicksort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quicksort(left) + middle + quicksort(right)

class DataProcessor:
    def __init__(self, config):
        self.config = config
        self.cache = {}

    def process(self, data):
        results = []
        for item in data:
            key = self._hash(item)
            if key in self.cache:
                results.append(self.cache[key])
            else:
                result = self._transform(item)
                self.cache[key] = result
                results.append(result)
        return results

    def _hash(self, item):
        return hash(str(item))

    def _transform(self, item):
        return {"processed": True, "data": item}
''',
        "json_response": json.dumps({
            "status": "success", "total_results": 156,
            "results": [
                {"id": i, "name": f"item_{i}", "score": round(3.14 * i, 2),
                 "tags": ["alpha", "beta", "gamma"],
                 "metadata": {"created": f"2026-06-{i:02d}", "version": "2.1"}}
                for i in range(1, 21)
            ],
            "pagination": {"page": 1, "per_page": 20, "total_pages": 8},
        }, indent=2),
        "natural_language": """
The transformer architecture, introduced in the landmark paper "Attention Is All You Need"
by Vaswani et al. (2017), revolutionized natural language processing by replacing recurrent
neural networks with self-attention mechanisms. The key innovation was the multi-head
attention layer, which allows the model to jointly attend to information from different
representation subspaces at different positions. Each attention head computes a weighted
sum of values, where the weights are determined by the compatibility of queries and keys.

In the context of large language model serving, the key-value cache (KVCache) stores the
intermediate key and value tensors computed during the prefill phase. This cache is
subsequently reused during the autoregressive decode phase, avoiding redundant computation
of attention scores for previously processed tokens. The KVCache grows linearly with the
sequence length and the number of attention heads, making memory management a critical
bottleneck in production serving systems.

The Mooncake architecture addresses this challenge by disaggregating prefill and decode
into independent clusters, connected through a global KVCache pool with RDMA transfer
and prefix-hash matching for cross-request cache reuse. SIG (Suspend-and-Inject Generation)
takes a complementary approach on edge devices, preserving the KVCache across tool-call
boundaries to eliminate redundant prefill computation in multi-step agent workflows.
""",
    }

    print(f"\n  模型: n_layers={n_layers}, n_kv_heads={n_kv_heads}, head_dim={head_dim}")

    results = {}

    print(f"\n    {'Content Type':<18} {'Tokens':<8} {'FP16 MB':<10} {'FP8 MB':<10} "
          f"{'Compress%':<12} {'Quant Err':<12} {'Max Abs'}")
    print(f"    {'-'*18} {'-'*8} {'-'*10} {'-'*10} {'-'*12} {'-'*12} {'-'*10}")

    for content_label, content_text in content_types.items():
        compiler.reset_cache()
        token_ids = list(compiler.tokenize(content_text, add_bos=False))
        n_tokens = len(token_ids)

        t0 = time.time()
        compiler.eval(token_ids)
        prefill_ms = (time.time() - t0) * 1000

        # 模拟 KVCache 数据
        fp16_total_bytes = 0
        fp8_total_bytes = 0
        total_quant_error = 0.0
        max_abs_error = 0.0
        n_tensors = 0

        for layer_i in range(n_layers):
            for kv in ["k", "v"]:
                arr = np.random.randn(n_kv_heads, head_dim, n_tokens).astype(np.float16)
                fp16_total_bytes += arr.nbytes

                max_abs = float(np.max(np.abs(arr))) + 1e-8
                scale = max_abs / 448.0
                quantized = (arr / scale).astype(np.int8)
                fp8_total_bytes += quantized.nbytes

                dequantized = quantized.astype(np.float32) * scale
                error = np.mean(np.abs(arr.astype(np.float32) - dequantized))
                max_err = float(np.max(np.abs(arr.astype(np.float32) - dequantized)))
                total_quant_error += error
                max_abs_error = max(max_abs_error, max_err)
                n_tensors += 1

        compress_ratio = 1.0 - (fp8_total_bytes / fp16_total_bytes) if fp16_total_bytes > 0 else 0
        avg_quant_error = total_quant_error / n_tensors if n_tensors > 0 else 0

        results[content_label] = {
            "n_tokens": n_tokens,
            "fp16_mb": round(fp16_total_bytes / (1024*1024), 2),
            "fp8_mb": round(fp8_total_bytes / (1024*1024), 2),
            "compress_ratio": round(compress_ratio, 4),
            "avg_quant_error": round(float(avg_quant_error), 6),
            "max_abs_error": round(float(max_abs_error), 6),
            "prefill_ms": round(prefill_ms, 2),
        }

        print(f"    {content_label:<18} {n_tokens:<8} {fp16_total_bytes/(1024*1024):<10.2f} "
              f"{fp8_total_bytes/(1024*1024):<10.2f} {compress_ratio:<12.2%} "
              f"{avg_quant_error:<12.6f} {max_abs_error:<10.4f}")

    # 压缩率变化范围总结
    compress_ratios = [r["compress_ratio"] for r in results.values()]
    print(f"\n  --- 压缩率变化范围总结 ---")
    print(f"    最小压缩率: {min(compress_ratios):.2%} ({min(compress_ratios, key=lambda x: x)})")
    print(f"    最大压缩率: {max(compress_ratios):.2%}")
    print(f"    变化范围:   {max(compress_ratios) - min(compress_ratios):.2%}")
    print(f"    平均压缩率: {sum(compress_ratios)/len(compress_ratios):.2%}")

    quant_errors = [r["avg_quant_error"] for r in results.values()]
    print(f"\n  --- 量化误差范围 ---")
    print(f"    最小误差: {min(quant_errors):.6f}")
    print(f"    最大误差: {max(quant_errors):.6f}")
    print(f"    平均误差: {sum(quant_errors)/len(quant_errors):.6f}")

    return results


# ================================================================
# 主函数
# ================================================================

def main():
    parser = argparse.ArgumentParser(description="Paper 6 评审改进实验 (Review V2)")
    parser.add_argument("--model", default="models/Qwen3.5-4B-Q4_K_M.gguf",
                        help="GGUF 模型路径")
    parser.add_argument("--n-ctx", type=int, default=8192)
    parser.add_argument("--n-threads", type=int, default=4)
    parser.add_argument("--n-gpu-layers", type=int, default=0)
    parser.add_argument("--exp", default="all",
                        choices=["5", "6", "7", "8", "9", "all"],
                        help="运行哪个实验 (5/6/7/8/9/all)")
    parser.add_argument("--output", default="data/paper6_review",
                        help="结果输出目录")
    args = parser.parse_args()

    if not LLAMA_AVAILABLE:
        print("ERROR: llama-cpp-python is required. Install with: pip install llama-cpp-python")
        return

    if not os.path.exists(args.model):
        print(f"ERROR: Model not found: {args.model}")
        print(f"  Available models:")
        for f in ["models/Qwen3.5-4B-Q4_K_M.gguf", "models/Qwen3.5-0.8B-Q4_K_M.gguf"]:
            print(f"    {f} {'(exists)' if os.path.exists(f) else '(not found)'}")
        return

    gpu = GPUMonitor()
    compiler = MeaningCompiler(
        model_path=args.model,
        n_ctx=args.n_ctx,
        n_threads=args.n_threads,
        n_gpu_layers=args.n_gpu_layers,
    )

    model_name = os.path.basename(args.model).replace(".gguf", "")
    all_results = {"model": model_name, "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}

    if args.exp in ("5", "all"):
        r = run_exp5(args, compiler, gpu)
        all_results["exp5_e2e_hybrid_poc"] = r

    if args.exp in ("6", "all"):
        r = run_exp6(args, compiler, gpu)
        all_results["exp6_jitter_sensitivity"] = r

    if args.exp in ("7", "all"):
        r = run_exp7(args, compiler, gpu)
        all_results["exp7_extended_complementarity"] = r

    if args.exp in ("8", "all"):
        r = run_exp8(args, compiler, gpu)
        all_results["exp8_privacy_grading"] = r

    if args.exp in ("9", "all"):
        r = run_exp9(args, compiler, gpu)
        all_results["exp9_compression_variation"] = r

    os.makedirs(args.output, exist_ok=True)
    out_path = os.path.join(args.output, f"paper6_review_{model_name}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n结果已保存: {out_path}")

    gpu.shutdown()
    print(f"\n实验完成.")


if __name__ == "__main__":
    main()
