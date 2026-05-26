# EdgeAgent-Kitchen Benchmark Results

> **Date**: 2026-05-26 (revised, n=3 clean batch)
> **Hardware**: NVIDIA GeForce RTX 4070 SUPER (12282 MB VRAM), CPU i7-class
> **Models**: Qwen3.5 0.8B / 4B (Q4_K_M quantization, dense, February 2026 release)
> **Framework**: llama-cpp-python (CUDA/CPU, n_ctx=16384)
> **Methodology**: n=3 independent subprocess runs per configuration, mean ± std
> **Code fixes**: kv_cache_clear, PrefixCache removal, evict_range rebuild_cache

---

## Executive Summary

```
┌─────────────────────┬────────────────────────────────────────────┐
│ Baseline            │ 4B GPU (n=3)           0.8B GPU (n=3)      │
├─────────────────────┼────────────────────────────────────────────┤
│ SIG                 │ 6.2 ± 0.0s             2.3 ± 0.1s          │
│ AppLoop             │ 15.7 ± 0.1s            2.3 ± 0.1s          │
│ AppLoop-PC          │ 23.5 ± 0.1s            8.6 ± 0.1s          │
│ Sliding             │ 15.8 ± 0.1s            2.2 ± 0.1s          │
├─────────────────────┼────────────────────────────────────────────┤
│ SIG/AppLoop         │ 2.54×                  1.00×                │
└─────────────────────┴────────────────────────────────────────────┘
```

### Per-Turn Generation Token Analysis (4B, 32 steps, diagnostic)

| Metric | SIG | AppLoop | Ratio |
|--------|-----|---------|-------|
| Total gen tokens | 475 | 921 | 1.94× |
| Total gen time | 4.4s | 9.0s | 2.05× |
| Per-token rate | 108 tok/s | 103 tok/s | 1.05× |

**Key finding**: Gen time ratio (2.05×) ≈ token count ratio (1.94×). Per-token rates differ by only 5%. AppLoop generates longer responses because the full re-encoding prompt template induces more verbose output.

### Speedup Decomposition (4B)

| Component | SIG | AppLoop | Δ | Attribution |
|-----------|-----|---------|---|-------------|
| Prefill | 0.1s | 5.5s | +5.4s | KV-cache persistence (57%) |
| Generation | 4.7s | 9.0s | +4.3s | Output verbosity (43%) |
| Total | 6.2s | 15.7s | +9.5s | 2.54× |

---

## 1. Kitchen Main Benchmark

### 4B GPU (n=3, --kitchen-steps 30, 32 actual steps)

| Baseline | Wall-Clock | Gen | Prefill | vs SIG |
|----------|-----------|-----|---------|--------|
| SIG | 6.2 ± 0.0s | 4.7s | 0.1s | 1.00× |
| AppLoop | 15.7 ± 0.1s | 9.0 ± 0.0s | 5.5 ± 0.1s | 2.54× |
| AppLoop-PC | 23.5 ± 0.1s | 15.4 ± 0.1s | 0.0s | 3.79× |
| Sliding | 15.8 ± 0.1s | 9.1 ± 0.1s | 5.5 ± 0.0s | 2.55× |

### 0.8B GPU (n=3, --kitchen-steps 30, 32 actual steps)

| Baseline | Wall-Clock | Gen | Prefill | vs SIG |
|----------|-----------|-----|---------|--------|
| SIG | 2.3 ± 0.1s | 1.6s | 0.1s | 1.00× |
| AppLoop | 2.3 ± 0.1s | 1.0s | 1.0s | 1.00× |
| AppLoop-PC | 8.6 ± 0.1s | 6.1s | 0.0s | 3.81× |
| Sliding | 2.2 ± 0.1s | 1.0s | 1.0s | 0.99× |

---

## 2. R15: Hybrid Scheduling (4B GPU, n=3)

Kitchen contains only continuous chains → Hybrid converges to 100% SIG (degenerate case). Pure SIG: 6.2s, Pure AppLoop: 15.9s. Hybrid scheduling requires genuinely mixed workloads to demonstrate value.

---

## 3. R16: Multi-Sequence Concurrency (4B GPU, n=3)

| Metric | Value |
|--------|-------|
| Total wall-clock | 18.2 ± 0.1s |
| Avg switch latency | 246.0 ± 0.2ms |
| Steps completed | 45/45 |
| Multi-seq API (theoretical) | <1ms (246×) |

VRAM savings: 59% (3 independent instances ~7.8 GB vs 1 with 3 KV sequences ~3.2 GB).

---

## 4. R17: Context Compression (4B GPU, n=3, 41 steps)

| Strategy | Wall-Clock | Cache Tokens |
|----------|-----------|-------------|
| None | 6.6 ± 0.0s | 2336 |
| Drop-25% | 6.9 ± 0.0s | 1006 (2.3×) |
| Drop-50% | 6.7 ± 0.1s | 386 (6.1×) |
| Recent-30 | 9.4 ± 0.0s | 1602 (1.5×) |

> F1 omitted: word-overlap metric is too conservative for 0.8B–4B models.

---

## 5. R18: Pipeline Separation (4B GPU, n=3, 25 steps)

| Metric | Value |
|--------|-------|
| SIG/AppLoop speedup | 4.0× |
| Prefill overlap potential | 3.3 ± 0.1% |

---

## 6. R19: Fragment Routing (4B GPU)

KV fragments 200–400× larger than text. Full Wi-Fi transmission uncompetitive (1,011–20,211ms vs 2ms local re-encode).

---

## Fix Details

| File | Change |
|------|--------|
| `core/compiler.py` | `reset_cache()`: `kv_cache_seq_rm` → `kv_cache_clear()` |
| `core/injection.py` | `evict_range()`: `rebuild_cache()` for correct compaction |
| `edge_agent_bench.py` | Removed PrefixCache overhead from AppLoop-PC |
| `run_multi_bench.py` | CPU_ONLY_TASKS → set(); `--no-gpu` via CLI |

**Root cause of original OOM**: `kv_cache_seq_rm(0, -1, -1)` marks slots free without defragmenting. After multiple reset→eval cycles the allocator can't find contiguous blocks. `kv_cache_clear()` fully resets the cache and allocator. All baselines now complete 100% of steps.
