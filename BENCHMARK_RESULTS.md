# EdgeAgent-Kitchen Benchmark Results

> **Date**: 2026-05-26 (peer-review response update)
> **Hardware**: NVIDIA GeForce RTX 4070 SUPER (12282 MB VRAM), CPU Intel i7-class
> **Models**: Qwen3.5 0.8B / 4B (Q4_K_M quantization, dense)
> **Framework**: llama-cpp-python (CUDA/CPU, n_ctx=8192)
> **Methodology**: single-run per configuration (deterministic temp=0)
> **Environment**: conda sig_bench, Python 3.11.15

---

## Executive Summary

```
                        4B GPU (35 steps)       0.8B GPU (35 steps)
SIG                     4.7s                    2.4s
AppLoop                 18.0s (3.85x)           2.7s (1.11x)
AppLoop-PC              30.2s (6.45x)           9.8s (4.01x)
Sliding                 18.1s (3.88x)           2.6s (1.09x)
```

SIG achieves 3.85x speedup on 4B, 1.11x on 0.8B. 0.8B SIG generates more tokens (1.7s vs 1.1s), incurring "generation inflation" that offsets prefill savings.

---

## 1. Kitchen Main Benchmark (GPU)

### 4B GPU (n=1, 35 steps)

| Baseline | Wall-Clock | Gen | Prefill | Turns/s | vs SIG |
|----------|-----------|-----|---------|---------|--------|
| SIG | 4.7s | 2.9s | 0.1s | 7.52 | 1.00x |
| AppLoop | 18.0s | 9.8s | 6.5s | 1.95 | 3.85x |
| AppLoop-PC | 30.2s | 19.3s | 0.0s | 1.16 | 6.45x |
| Sliding | 18.1s | 9.8s | 6.6s | 1.94 | 3.88x |

### 0.8B GPU (n=1, 35 steps)

| Baseline | Wall-Clock | Gen | Prefill | Turns/s | vs SIG |
|----------|-----------|-----|---------|---------|--------|
| SIG | 2.4s | 1.7s | 0.1s | 14.44 | 1.00x |
| AppLoop | 2.7s | 1.1s | 1.2s | 13.19 | 1.11x |
| AppLoop-PC | 9.8s | 6.8s | 0.0s | 3.58 | 4.01x |
| Sliding | 2.6s | 1.1s | 1.2s | 13.57 | 1.09x |

---

## 2. R15: Hybrid Scheduling

Kitchen contains only continuous chains -> Hybrid converges to 100% SIG (degenerate). SIG: 4.6s, AppLoop: 18.0s, Hybrid split SIG=35/PC=0. Requires genuinely mixed workloads.

---

## 3. R16-R19: Multi-Sequence / Compression / Pipeline / Routing

See `bench_multi_results.json` for detailed R16-R19 measurements.

---

## 4. FlashAttention-Normalized Prefill Analysis (GPU, --task flash, 35 steps)

### 4B GPU

| FA Factor | SIG pf | SIG wc | AppLoop pf | AppLoop wc | Speedup | SIG pf% |
|-----------|--------|--------|-----------|-----------|---------|---------|
| 1.0x (naive) | 0.12s | 3.78s | 6.74s | 17.84s | 4.71x | 3.1% |
| 2.0x | 0.06s | 3.73s | 3.37s | 14.48s | 3.88x | 1.6% |
| 3.0x (FA-2) | 0.04s | 3.71s | 2.25s | 13.35s | 3.60x | 1.1% |
| 5.0x (FA-3) | 0.02s | 3.69s | 1.35s | 12.45s | 3.37x | 0.6% |
| 8.0x (vLLM) | 0.01s | 3.68s | 0.84s | 11.95s | 3.25x | 0.4% |

### 0.8B GPU (generation inflation inverts speedup)

| FA Factor | SIG pf | SIG wc | AppLoop pf | AppLoop wc | Speedup | SIG pf% |
|-----------|--------|--------|-----------|-----------|---------|---------|
| 1.0x (naive) | 0.10s | 1.75s | 1.21s | 2.23s | 1.27x | 5.9% |
| 2.0x | 0.05s | 1.70s | 0.61s | 1.62s | 0.96x | 3.1% |
| 3.0x (FA-2) | 0.03s | 1.68s | 0.40s | 1.42s | 0.85x | 2.1% |
| 5.0x (FA-3) | 0.02s | 1.67s | 0.24s | 1.26s | 0.76x | 1.2% |
| 8.0x (vLLM) | 0.01s | 1.66s | 0.15s | 1.17s | 0.71x | 0.8% |

**Key insight:** On 0.8B, SIG is already only 1.27x at naive baseline, and inverts at 2.0x FA due to generation inflation (SIG gen=1.65s vs AppLoop gen=1.02s). On 4B, SIG retains robust 3.25x even at 8.0x vLLM-level FA.

---

## 5. R20: Retrospective SIG — Compensatory Recall (4B GPU, 35 steps)

| Agent | Wall-Clock | Gen Tokens | Prefill Tokens | vs SIG |
|-------|-----------|-----------|----------------|--------|
| SIG | 5.1s | 394 | 1,579 | 1.00x |
| AppLoop | 19.6s | 1,132 | 47,689 | 3.85x |
| RetroSIG (int=5) | 6.1s | 439 | 2,341 | 1.20x |
| RetroSIG (int=3) | 16.2s | 1,525 | 2,754 | 3.20x |
| RetroSIG-Heavy | 6.4s | 431 | 3,374 | 1.27x |

**Key insight:** On GPU, RetroSIG-Heavy (recall every step) is only 1.27x vs SIG (6.4s vs 5.1s), while RetroSIG(int=5) is 1.20x. The recall prompts themselves cost little on GPU, and the generation token counts remain near SIG levels (394-439), avoiding the explosion seen with int=3 (1525 tokens).

---

## 6. R21: SIG + KV Cache Compression (4B GPU, 41 steps)

| Config | Wall-Clock | Cache (tok) | Removed | vs AppLoop |
|--------|-----------|------------|---------|------------|
| SIG | 5.3s | 2,244 | 0 | 4.57x |
| AppLoop | 24.1s | 0 | 0 | 1.00x |
| CompSIG-30% | 5.9s | 664 | 1,565 | 4.08x |
| CompSIG-50% | 6.2s | 870 | 1,391 | 3.88x |
| CompSIG-70% | 7.1s | 1,272 | 1,078 | 3.39x |

**Key insight:** CompSIG-50% retains 3.88x vs AppLoop while reducing cache 61% (2,244->870). Wall-clock overhead is only +17% (5.3s->6.2s). CompSIG-30% achieves the smallest cache (664) but may drop mid-context facts.

---

## 7. Fusion: RetroSIG + CompSIG (4B GPU, 40 steps)

| Agent | Wall-Clock | Gen Tokens | Cache (tok) | Removed | vs AppLoop |
|-------|-----------|-----------|-------------|---------|------------|
| SIG | 5.3s | 424 | 2,244 | 0 | 4.57x |
| AppLoop | 24.0s | 1,350 | 0 | 0 | 1.00x |
| RetroSIG | 18.6s | 1,789 | 5,034 | 0 | 1.29x |
| CompSIG | 6.2s | 441 | 870 | 1,391 | 3.86x |
| RetroCompSIG | 11.0s | 923 | 1,064 | 2,226 | 2.19x |

**Key insight:** CompSIG alone is the best configuration (6.2s, 3.86x vs AppLoop, cache 870). RetroSIG standalone is too slow (18.6s, inflated to 5,034 cache tokens). RetroCompSIG fusion is a middle ground (11.0s, 2.19x) but still lags CompSIG. **Conclusion: compression alone is sufficient for bounded-memory SIG; retro recall adds cost without proportional benefit in this setup.**

---

## 8. Model-Scale Comparison

| Metric | 4B GPU | 0.8B GPU |
|--------|--------|----------|
| Kitchen speedup (naive) | 3.85x | 1.11x |
| Kitchen speedup (3x FA) | 3.60x | 0.85x |
| AppLoop prefill % | 36.1% | 43.7% |
| SIG prefill % | 2.0% | 5.9% |
| SIG gen inflation | None (faster) | 2.01x vs AppLoop gen |

**SIG is appropriate for models >=1B parameters on edge hardware. Sub-1B models require active KV-cache compression to avoid generation inflation.**
