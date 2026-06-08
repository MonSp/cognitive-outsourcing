#!/usr/bin/env python3
"""KV-Cache Disk Persistence Benchmark — H1.1 Validation.

Validates the cross-session prefix reuse benefit of disk-backed KV-Cache
persistence (Horizon 1.1 from Paper 6 roadmap).

Measures:
  1. Save/Load latency (state serialization vs token replay)
  2. Multi-session cold-start elimination (disk save + load overhead vs re-prefill)
  3. State size decomposition (KV-Cache tensor vs full LlamaState)
  4. Compression impact with zero-byte analysis
  5. Long-context prefix scaling

Usage:
  conda activate sig_bench
  python exp_kvcache_persist.py --model models/Qwen3.5-0.8B-Q4_K_M.gguf --n-gpu-layers 99
  python exp_kvcache_persist.py --model models/Qwen3.5-4B-Q4_K_M.gguf --n-gpu-layers 99
"""

import os
import time
import json
import struct
import shutil
import argparse
import tempfile
from pathlib import Path
from typing import List, Dict

from core import MeaningCompiler, GPUMonitor
from core.compiler import PrefixCache
from core.kvcache_persist import DiskKVCache
from core.prompts import SYSTEM_PROMPT, TOOL_DESCRIPTIONS_TRAVEL


def bench_save_load(
    compiler: MeaningCompiler,
    prefix_ids: List[int],
    n_sessions: int = 10,
    compress: bool = False,
) -> Dict:
    """Benchmark 1: Save/Load latency comparison with real per-session data.

    Compares four strategies, recording per-session measurements:
      - Cold start:   rebuild_cache from tokens (full re-prefill)
      - In-memory PC: PrefixCache save/restore (sequence copy)
      - Disk fast:    DiskKVCache load via load_state() (SSD deserialization)
      - Disk save:    DiskKVCache save via save_state() (SSD serialization)
    """
    tmpdir = tempfile.mkdtemp(prefix="kvcache_bench_")
    disk_cache = DiskKVCache(
        os.path.join(tmpdir, "kv_store"), max_entries=32, compress=compress
    )

    cold_times = []
    pc_save_times = []
    pc_restore_times = []
    disk_save_times = []
    disk_load_times = []

    pc = PrefixCache()

    for sess in range(n_sessions):
        compiler.reset_cache()
        t0 = time.time()
        compiler.eval(prefix_ids)
        cold_times.append((time.time() - t0) * 1000)

        t0 = time.time()
        pc.save(compiler, prefix_ids)
        pc_save_times.append((time.time() - t0) * 1000)

        t0 = time.time()
        disk_cache.save(compiler, prefix_ids, tag=f"session_{sess}")
        disk_save_times.append((time.time() - t0) * 1000)

    for sess in range(n_sessions):
        compiler.reset_cache()
        t0 = time.time()
        pc.restore(compiler)
        pc_restore_times.append((time.time() - t0) * 1000)

    for sess in range(n_sessions):
        compiler.reset_cache()
        t0 = time.time()
        disk_cache.load(compiler, tag=f"session_{sess}")
        disk_load_times.append((time.time() - t0) * 1000)

    avg = lambda lst: sum(lst) / len(lst) if lst else 0.0
    median = lambda lst: sorted(lst)[len(lst) // 2] if lst else 0.0

    results = {
        "n_sessions": n_sessions,
        "n_prefix_tokens": len(prefix_ids),
        "cold_start_per_session_ms": [round(t, 2) for t in cold_times],
        "cold_start_avg_ms": round(avg(cold_times), 2),
        "cold_start_median_ms": round(median(cold_times), 2),
        "pc_restore_per_session_ms": [round(t, 2) for t in pc_restore_times],
        "pc_restore_avg_ms": round(avg(pc_restore_times), 2),
        "disk_save_per_session_ms": [round(t, 2) for t in disk_save_times],
        "disk_save_avg_ms": round(avg(disk_save_times), 2),
        "disk_save_median_ms": round(median(disk_save_times), 2),
        "disk_load_per_session_ms": [round(t, 2) for t in disk_load_times],
        "disk_load_avg_ms": round(avg(disk_load_times), 2),
        "disk_load_median_ms": round(median(disk_load_times), 2),
    }

    disk_usage = disk_cache.get_disk_usage()
    stats = disk_cache.get_stats()
    results["disk_usage_bytes"] = disk_usage["bytes"]
    results["disk_usage_mb"] = disk_usage["mb"]
    results["cache_stats"] = stats

    shutil.rmtree(tmpdir, ignore_errors=True)
    return results


def bench_multi_session(
    compiler: MeaningCompiler,
    prefix_ids: List[int],
    n_sessions: int = 5,
    compress: bool = False,
) -> Dict:
    """Benchmark 2: Multi-session cold-start elimination.

    Compares the cumulative cost of N independent cold starts versus:
      1 x cold_start + 1 x disk_save + (N-1) x disk_load

    This isolates the prefix prefill component and measures whether
    DiskKVCache actually reduces cumulative latency at the session boundary.
    """
    tmpdir = tempfile.mkdtemp(prefix="kvcache_multisess_")
    disk_cache = DiskKVCache(
        os.path.join(tmpdir, "kv_store"), max_entries=32, compress=compress
    )

    cold_times = []
    for sess in range(n_sessions):
        compiler.reset_cache()
        t0 = time.time()
        compiler.eval(prefix_ids)
        cold_times.append((time.time() - t0) * 1000)

    compiler.reset_cache()
    t0 = time.time()
    compiler.eval(prefix_ids)
    first_cold_ms = (time.time() - t0) * 1000

    t0 = time.time()
    disk_cache.save(compiler, prefix_ids, tag="shared_prefix")
    disk_save_ms = (time.time() - t0) * 1000

    disk_load_times = []
    for sess in range(1, n_sessions):
        compiler.reset_cache()
        t0 = time.time()
        disk_cache.load(compiler, tag="shared_prefix")
        disk_load_times.append((time.time() - t0) * 1000)

    compiler.reset_cache()
    fresh_cache = DiskKVCache(
        os.path.join(tmpdir, "kv_store"), max_entries=32, compress=compress
    )
    t0 = time.time()
    fresh_cache.load(compiler, tag="shared_prefix")
    warm_reload_ms = (time.time() - t0) * 1000

    avg = lambda lst: sum(lst) / len(lst) if lst else 0.0
    total_cold = sum(cold_times)
    total_disk = first_cold_ms + disk_save_ms + sum(disk_load_times)

    return {
        "n_sessions": n_sessions,
        "n_prefix_tokens": len(prefix_ids),
        "cold_per_session_ms": [round(t, 2) for t in cold_times],
        "cold_avg_ms": round(avg(cold_times), 2),
        "cold_total_ms": round(total_cold, 2),
        "first_cold_ms": round(first_cold_ms, 2),
        "disk_save_ms": round(disk_save_ms, 2),
        "disk_load_per_session_ms": [round(t, 2) for t in disk_load_times],
        "disk_load_avg_ms": round(avg(disk_load_times), 2) if disk_load_times else 0,
        "disk_total_ms": round(total_disk, 2),
        "warm_reload_ms": round(warm_reload_ms, 2),
        "savings_ms": round(total_cold - total_disk, 2),
        "savings_pct": round(
            (1 - total_disk / total_cold) * 100, 1
        ) if total_cold > 0 else 0.0,
        "breakeven_sessions": round(
            (disk_save_ms - avg(disk_load_times)) / (avg(cold_times) - avg(disk_load_times)), 1
        ) if avg(cold_times) > avg(disk_load_times) else float('inf'),
    }


def bench_state_decomposition(
    compiler: MeaningCompiler,
) -> Dict:
    """Benchmark 3: State size decomposition — KV-Cache vs full LlamaState.

    save_state() serializes the full llama.cpp context (including unused
    KV slots), not just the populated KV-Cache tensors. This benchmark
    measures:
      - Total state size at different token counts
      - Incremental cost per additional token
      - Non-zero byte ratio (revealing zero-padding dominance)
      - State overhead ratio (state_size / n_tokens)
    """
    tmpdir = tempfile.mkdtemp(prefix="kvcache_match_")
    disk_cache = DiskKVCache(
        os.path.join(tmpdir, "kv_store"), max_entries=64
    )

    prefixes = {
        "system": compiler.tokenize(SYSTEM_PROMPT, add_bos=True),
        "tools": compiler.tokenize(TOOL_DESCRIPTIONS_TRAVEL, add_bos=False),
        "combined": compiler.tokenize(
            SYSTEM_PROMPT + "\n" + TOOL_DESCRIPTIONS_TRAVEL, add_bos=True
        ),
    }

    sizes = {}
    for name, ids in prefixes.items():
        compiler.reset_cache()
        compiler.eval(ids)
        cache_id = disk_cache.save(compiler, ids, tag=name)

        state_path = disk_cache._state_path(cache_id)
        raw_bytes = state_path.read_bytes()
        n_total = len(raw_bytes)
        n_nonzero = sum(1 for b in raw_bytes if b != 0)
        zero_ratio = round((1 - n_nonzero / n_total) * 100, 1) if n_total > 0 else 0

        import zlib
        nonzero_only = bytes(b for b in raw_bytes if b != 0)
        compressed_raw = len(zlib.compress(raw_bytes, 6))
        compressed_nonzero = len(zlib.compress(nonzero_only, 6)) if nonzero_only else 0

        sizes[name] = {
            "n_tokens": len(ids),
            "state_bytes": n_total,
            "state_mb": round(n_total / (1024 * 1024), 2),
            "nonzero_bytes": n_nonzero,
            "nonzero_ratio_pct": round(n_nonzero / n_total * 100, 1) if n_total > 0 else 0,
            "zero_padding_pct": zero_ratio,
            "compressed_raw_bytes": compressed_raw,
            "compressed_nonzero_bytes": compressed_nonzero,
            "bytes_per_token": round(n_total / len(ids), 0) if ids else 0,
        }

    if "system" in sizes and "tools" in sizes and "combined" in sizes:
        sys_b = sizes["system"]["state_bytes"]
        tools_b = sizes["tools"]["state_bytes"]
        comb_b = sizes["combined"]["state_bytes"]
        incremental = comb_b - max(sys_b, tools_b)
        sizes["incremental_tools_after_system"] = {
            "bytes": incremental,
            "mb": round(incremental / (1024 * 1024), 2),
        }

    stats = disk_cache.get_stats()

    shutil.rmtree(tmpdir, ignore_errors=True)
    return {"prefixes": sizes, "stats": stats}


def bench_compression(
    compiler: MeaningCompiler,
    prefix_ids: List[int],
) -> Dict:
    """Benchmark 4: Compression impact with zero-byte analysis.

    The naive compression ratio can be misleading when LlamaState contains
    large zero-padded regions. This benchmark reports:
      - Raw compression ratio (uncompressed / compressed bytes)
      - Non-zero byte ratio (revealing zero-padding dominance)
      - Effective compression ratio (nonzero-only bytes / compressed bytes)
      - Save/Load latency for both modes
    """
    tmpdir = tempfile.mkdtemp(prefix="kvcache_compress_")
    uncompressed_dir = os.path.join(tmpdir, "uncompressed")
    compressed_dir = os.path.join(tmpdir, "compressed")

    cache_plain = DiskKVCache(uncompressed_dir, max_entries=32, compress=False)
    cache_zlib = DiskKVCache(compressed_dir, max_entries=32, compress=True, compress_level=6)

    compiler.reset_cache()
    compiler.eval(prefix_ids)

    t0 = time.time()
    cache_plain.save(compiler, prefix_ids, tag="bench")
    save_uncompressed_ms = (time.time() - t0) * 1000

    compiler.reset_cache()
    t0 = time.time()
    cache_zlib.save(compiler, prefix_ids, tag="bench")
    save_compressed_ms = (time.time() - t0) * 1000

    compiler.reset_cache()
    t0 = time.time()
    cache_plain.load(compiler, tag="bench")
    load_uncompressed_ms = (time.time() - t0) * 1000

    compiler.reset_cache()
    t0 = time.time()
    cache_zlib.load(compiler, tag="bench")
    load_compressed_ms = (time.time() - t0) * 1000

    plain_usage = cache_plain.get_disk_usage()
    zlib_usage = cache_zlib.get_disk_usage()
    uncompressed_bytes = plain_usage["bytes"]
    compressed_bytes = zlib_usage["bytes"]

    state_files = list(
        Path(uncompressed_dir).glob("*.state")
    )
    nonzero_count = 0
    total_count = 0
    for sf in state_files:
        raw = sf.read_bytes()
        total_count += len(raw)
        nonzero_count += sum(1 for b in raw if b != 0)

    nonzero_ratio = nonzero_count / total_count if total_count > 0 else 0
    zero_padding_pct = round((1 - nonzero_ratio) * 100, 1)

    nonzero_bytes = nonzero_count
    effective_ratio = nonzero_bytes / compressed_bytes if compressed_bytes > 0 else 0

    shutil.rmtree(tmpdir, ignore_errors=True)
    return {
        "uncompressed_bytes": uncompressed_bytes,
        "compressed_bytes": compressed_bytes,
        "nonzero_bytes": nonzero_bytes,
        "zero_padding_pct": zero_padding_pct,
        "compression_ratio_raw": round(
            uncompressed_bytes / compressed_bytes, 2
        ) if compressed_bytes > 0 else 0,
        "compression_ratio_effective": round(effective_ratio, 2),
        "save_uncompressed_ms": round(save_uncompressed_ms, 2),
        "save_compressed_ms": round(save_compressed_ms, 2),
        "load_uncompressed_ms": round(load_uncompressed_ms, 2),
        "load_compressed_ms": round(load_compressed_ms, 2),
    }


def bench_long_context(
    compiler: MeaningCompiler,
    prefix_ids: List[int],
    n_ctx: int,
) -> Dict:
    """Benchmark 5: Long-context prefix scaling.

    Measures how DiskKVCache behaves as prefix length grows toward
    the context window limit. For short prefixes, the state file is
    dominated by zero-padding and cold-start may be faster. For long
    prefixes, the KV-Cache tensors dominate and DiskKVCache should win.
    """
    filler_template = "The quick brown fox jumps over the lazy dog. "
    filler_ids = compiler.tokenize(filler_template, add_bos=False)

    ratios = [0.01, 0.05, 0.10, 0.25, 0.50]
    results = []

    for ratio in ratios:
        target_tokens = int(n_ctx * ratio)
        if target_tokens < len(prefix_ids):
            target_tokens = len(prefix_ids)

        extra_needed = target_tokens - len(prefix_ids)
        if extra_needed <= 0:
            long_prefix = prefix_ids[:target_tokens]
        else:
            repeats = (extra_needed // len(filler_ids)) + 1
            extended = prefix_ids + (filler_ids * repeats)[:extra_needed]
            long_prefix = extended

        if len(long_prefix) > n_ctx:
            long_prefix = long_prefix[:n_ctx]

        n_actual = len(long_prefix)

        compiler.reset_cache()
        t0 = time.time()
        compiler.eval(long_prefix)
        cold_ms = (time.time() - t0) * 1000

        tmpdir = tempfile.mkdtemp(prefix="kvcache_long_")
        disk_cache = DiskKVCache(
            os.path.join(tmpdir, "kv_store"), max_entries=8
        )

        t0 = time.time()
        disk_cache.save(compiler, long_prefix, tag="long")
        save_ms = (time.time() - t0) * 1000

        state_files = list(Path(os.path.join(tmpdir, "kv_store")).glob("*.state"))
        state_bytes = state_files[0].stat().st_size if state_files else 0
        raw_data = state_files[0].read_bytes() if state_files else b""
        nonzero = sum(1 for b in raw_data if b != 0)
        nonzero_pct = round(nonzero / len(raw_data) * 100, 1) if raw_data else 0

        compiler.reset_cache()
        t0 = time.time()
        disk_cache.load(compiler, tag="long")
        load_ms = (time.time() - t0) * 1000

        results.append({
            "ctx_ratio": ratio,
            "n_tokens": n_actual,
            "cold_ms": round(cold_ms, 2),
            "save_ms": round(save_ms, 2),
            "load_ms": round(load_ms, 2),
            "state_mb": round(state_bytes / (1024 * 1024), 2),
            "nonzero_pct": nonzero_pct,
            "speedup_vs_cold": round(cold_ms / load_ms, 2) if load_ms > 0 else 0,
            "disk_wins": load_ms < cold_ms,
        })

        shutil.rmtree(tmpdir, ignore_errors=True)

    return {
        "n_ctx": n_ctx,
        "n_prefix_tokens": len(prefix_ids),
        "scales": results,
    }


def run_benchmark(args):
    print(f"{'='*70}")
    print(f"  KV-Cache Disk Persistence Benchmark (H1.1)")
    print(f"  Model: {args.model}")
    print(f"  GPU Layers: {args.n_gpu_layers}")
    print(f"  Context: {args.n_ctx}")
    print(f"{'='*70}")

    gpu = GPUMonitor()
    compiler = MeaningCompiler(
        model_path=args.model,
        n_ctx=args.n_ctx,
        n_threads=args.n_threads,
        n_gpu_layers=args.n_gpu_layers,
    )

    system_ids = compiler.tokenize(SYSTEM_PROMPT, add_bos=True)
    tools_ids = compiler.tokenize(TOOL_DESCRIPTIONS_TRAVEL, add_bos=False)
    prefix_ids = system_ids + tools_ids

    result = {
        "model": os.path.basename(args.model),
        "n_ctx": args.n_ctx,
        "n_gpu_layers": args.n_gpu_layers,
        "n_prefix_tokens": len(prefix_ids),
    }

    # ---- Exp 1: Save/Load Latency ----
    print(f"\n{'='*70}")
    print(f"  Exp 1: Save/Load Latency (N={args.n_sessions} sessions)")
    print(f"{'='*70}")

    latency = bench_save_load(compiler, prefix_ids, n_sessions=args.n_sessions, compress=args.compress)
    result["exp1_latency"] = latency

    print(f"    Prefix tokens:      {latency['n_prefix_tokens']}")
    print(f"    Cold start:         avg={latency['cold_start_avg_ms']:.2f} ms  median={latency['cold_start_median_ms']:.2f} ms")
    print(f"    Disk save:          avg={latency['disk_save_avg_ms']:.2f} ms  median={latency['disk_save_median_ms']:.2f} ms")
    print(f"    Disk load:          avg={latency['disk_load_avg_ms']:.2f} ms  median={latency['disk_load_median_ms']:.2f} ms")
    print(f"    PC restore:         avg={latency['pc_restore_avg_ms']:.2f} ms")
    print(f"    Disk usage:         {latency['disk_usage_mb']:.2f} MB")

    cold = latency["cold_start_avg_ms"]
    disk_load = latency["disk_load_avg_ms"]
    if cold > 0 and disk_load > 0:
        speedup = cold / disk_load
        print(f"    Speedup (cold→load): {speedup:.2f}x")
        if speedup < 1:
            print(f"    ⚠ Disk load SLOWER than cold start (state I/O > token eval)")

    # ---- Exp 2: Multi-Session Cold-Start Elimination ----
    print(f"\n{'='*70}")
    print(f"  Exp 2: Multi-Session Cold-Start Elimination (N=5)")
    print(f"{'='*70}")

    multi = bench_multi_session(
        compiler, prefix_ids,
        n_sessions=5, compress=args.compress,
    )
    result["exp2_multi_session"] = multi

    print(f"    Cold start total:   {multi['cold_total_ms']:.2f} ms")
    print(f"    Disk total:         {multi['disk_total_ms']:.2f} ms")
    print(f"      first cold:       {multi['first_cold_ms']:.2f} ms")
    print(f"      save:             {multi['disk_save_ms']:.2f} ms")
    print(f"      load avg:         {multi['disk_load_avg_ms']:.2f} ms")
    print(f"    Savings:            {multi['savings_ms']:.2f} ms ({multi['savings_pct']:.1f}%)")
    print(f"    Break-even N:       {multi['breakeven_sessions']}")
    print(f"    Warm reload:        {multi['warm_reload_ms']:.2f} ms")

    if multi['savings_pct'] < 0:
        print(f"    ⚠ DiskKVCache SLOWER than repeated cold starts for this prefix size")
        print(f"    Reason: state I/O ({multi['disk_save_ms']:.1f}+{multi['disk_load_avg_ms']:.1f} ms) > token eval ({multi['cold_avg_ms']:.1f} ms)")

    # ---- Exp 3: State Decomposition ----
    print(f"\n{'='*70}")
    print(f"  Exp 3: State Size Decomposition")
    print(f"{'='*70}")

    decomposition = bench_state_decomposition(compiler)
    result["exp3_decomposition"] = decomposition

    for name, info in decomposition["prefixes"].items():
        if isinstance(info, dict) and "state_mb" in info:
            print(f"    {name:<12}: {info['n_tokens']:>6} tokens → "
                  f"{info['state_mb']:>8.2f} MB total, "
                  f"{info['nonzero_ratio_pct']:>5.1f}% nonzero, "
                  f"{info['zero_padding_pct']:>5.1f}% zero-padding")
            print(f"    {'':<12}  compressed: raw={info['compressed_raw_bytes']:>10,} B, "
                  f"nonzero-only={info['compressed_nonzero_bytes']:>10,} B")
        elif isinstance(info, dict) and "bytes" in info:
            print(f"    {name:<12}: {info['mb']:>8.2f} MB incremental")

    # ---- Exp 4: Compression Impact ----
    print(f"\n{'='*70}")
    print(f"  Exp 4: Compression Impact (with zero-byte analysis)")
    print(f"{'='*70}")

    compression = bench_compression(compiler, prefix_ids)
    result["exp4_compression"] = compression

    print(f"    Uncompressed:       {compression['uncompressed_bytes']:,} bytes")
    print(f"    Compressed:         {compression['compressed_bytes']:,} bytes")
    print(f"    Non-zero bytes:     {compression['nonzero_bytes']:,} bytes")
    print(f"    Zero-padding:       {compression['zero_padding_pct']:.1f}%")
    print(f"    Ratio (raw):        {compression['compression_ratio_raw']:.1f}x")
    print(f"    Ratio (effective):  {compression['compression_ratio_effective']:.1f}x")
    print(f"    Save (plain):       {compression['save_uncompressed_ms']:.2f} ms")
    print(f"    Save (zlib):        {compression['save_compressed_ms']:.2f} ms")
    print(f"    Load (plain):       {compression['load_uncompressed_ms']:.2f} ms")
    print(f"    Load (zlib):        {compression['load_compressed_ms']:.2f} ms")

    if compression['zero_padding_pct'] > 50:
        print(f"    NOTE: >50% zero-padding — raw compression ratio is dominated by")
        print(f"    zero-value bytes in unused KV-Cache slots, not by actual data compression.")

    # ---- Exp 5: Long-Context Scaling ----
    print(f"\n{'='*70}")
    print(f"  Exp 5: Long-Context Prefix Scaling")
    print(f"{'='*70}")

    long_ctx = bench_long_context(compiler, prefix_ids, args.n_ctx)
    result["exp5_long_context"] = long_ctx

    print(f"    {'Ratio':>6} {'Tokens':>8} {'Cold(ms)':>10} {'Save(ms)':>10} {'Load(ms)':>10} {'State(MB)':>10} {'NZ%':>6} {'Speedup':>8} {'Winner':>10}")
    print(f"    {'-'*80}")
    for s in long_ctx["scales"]:
        winner = "Disk" if s["disk_wins"] else "Cold"
        print(f"    {s['ctx_ratio']:>6.0%} {s['n_tokens']:>8} {s['cold_ms']:>10.2f} {s['save_ms']:>10.2f} {s['load_ms']:>10.2f} {s['state_mb']:>10.2f} {s['nonzero_pct']:>5.1f}% {s['speedup_vs_cold']:>7.2f}x {winner:>10}")

    crossover = None
    for i in range(len(long_ctx["scales"]) - 1):
        s0 = long_ctx["scales"][i]
        s1 = long_ctx["scales"][i + 1]
        if not s0["disk_wins"] and s1["disk_wins"]:
            crossover = s1["ctx_ratio"]
            break

    if crossover:
        print(f"    DiskKVCache becomes faster at ~{crossover:.0%} context utilization")
    elif long_ctx["scales"] and all(s["disk_wins"] for s in long_ctx["scales"]):
        print(f"    DiskKVCache wins at all tested context ratios")
    elif long_ctx["scales"] and not any(s["disk_wins"] for s in long_ctx["scales"]):
        print(f"    DiskKVCache slower at all tested ratios (state I/O dominates)")

    # ---- Summary ----
    print(f"\n{'='*70}")
    print(f"  Summary")
    print(f"{'='*70}")

    gpu_stats = gpu.snapshot() if hasattr(gpu, "snapshot") else {}
    result["gpu_stats"] = gpu_stats

    out_path = os.path.join(
        "data",
        f"kvcache_persist_{os.path.basename(args.model).replace('.gguf', '')}.json",
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"    Results saved to: {out_path}")

    return result


def main():
    parser = argparse.ArgumentParser(
        description="KV-Cache Disk Persistence Benchmark (H1.1)"
    )
    parser.add_argument(
        "--model",
        default="models/Qwen3.5-0.8B-Q4_K_M.gguf",
        help="Path to GGUF model",
    )
    parser.add_argument("--n-ctx", type=int, default=8192)
    parser.add_argument("--n-threads", type=int, default=4)
    parser.add_argument("--n-gpu-layers", type=int, default=0)
    parser.add_argument("--n-sessions", type=int, default=10, help="Number of sessions for Exp 1")
    parser.add_argument("--compress", action="store_true", help="Enable zlib compression for state files")
    args = parser.parse_args()
    run_benchmark(args)


if __name__ == "__main__":
    main()
