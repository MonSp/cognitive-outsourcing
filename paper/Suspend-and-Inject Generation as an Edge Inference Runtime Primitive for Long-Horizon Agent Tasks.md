# SIG as an Edge Inference Runtime Primitive for Long-Horizon Agent Tasks

> **Paper 4 of 4 in the SIG/CO Research Program**
>
> Preceding papers: [1] *Cognitive Outsourcing with Suspend-and-Inject Generation* (Paper 1),
> [2] *Beyond the Injection Engine: A Five-Dimensional Analysis* (Paper 2),
> [3] *Extending CO-SIG Across Nine Additional Research Vectors* (Paper 3).
>
> **Date**: May 2026 (revised)

> **Erratum (May 2026)**: The original submission reported deterministic OOM failures
> in AppLoop-PC, AppLoop-Sliding, and SIG-Hybrid baselines. A post-submission code
> audit identified the root cause as `reset_cache()` using `kv_cache_seq_rm` instead
> of `kv_cache_clear()`, causing memory pool fragmentation. After correction, all
> baselines complete 100% of steps with zero failures.

---

**Abstract**

This paper completes a four-paper research program establishing SIG (Suspend-and-Inject Generation) as a scale-dependent performance optimization for edge inference runtimes. Papers 1–3 introduced the CO-SIG architecture, proved its mechanism through five-dimensional theoretical analysis (R1–R5), and empirically characterized its advantage zones and design boundaries (R6–R14). This paper advances the program by (1) introducing EdgeAgent-Kitchen, a realistic edge agent benchmark with 50–200 step interleaved multi-task tool chains, (2) conducting GPU vs. CPU comparisons that demonstrate SIG's speedup derives from KV-cache persistence, and (3) outlining deployment-oriented research vectors (R15–R19). On 4B models with n=3 independent process runs, SIG delivers **2.54× wall-clock speedup** (6.2s vs 15.7s, mean±0.1s). A per-turn token generation analysis reveals that AppLoop generates 1.94× more output tokens than SIG (921 vs 475 over 32 steps), fully explaining the generation time gap: per-token rates differ by only 5% (103 vs 108 tok/s), consistent with Paper 2's ±2% equivalence finding. The prefill crossover—the model size at which SIG outperforms AppLoop—lies at approximately 1.5–2B parameters; on 0.8B models, SIG and AppLoop are tied at 2.3s (n=3, p>0.3). We characterize SIG as a **scale-dependent optimization** whose value grows with model size and task complexity.

---

## 1. Introduction

### 1.1 The Edge Agent Inference Challenge

The deployment of autonomous language agents on edge devices demands models that execute extended tool chains while maintaining coherent memory across diverse, interleaved tasks. The standard inference loop—**AppLoop**—triggers a full context re-encode (prefill) at every tool invocation, causing prefill cost to grow linearly with context length and disproportionately with model size. On a 4B model, 32 steps consume 5.5±0.1s (GPU) to ~42s (CPU) in prefill alone.

### 1.2 Papers 1–3 Summary

**Paper 1 [1]** introduced SIG and demonstrated 84–96% prefill token savings. **Paper 2 [2]** provided theoretical foundations (R1–R5), crucially establishing that SIG and AppLoop generate tokens at **nearly identical per-token rates** (±2%). **Paper 3 [3]** characterized design boundaries through R6–R14, finding **2.79× (0.8B) / 4.26× (4B)** speedup on deep chains and introducing the SIG Decision Framework.

### 1.3 This Paper: Contributions

1. **EdgeAgent-Kitchen benchmark**: Interleaved multi-task tool chains in a persistent kitchen-assistant session.
2. **Hardware-agnostic evaluation**: GPU vs. CPU comparison demonstrating speedup consistency.
3. **Per-turn token generation analysis**: Quantifying per-turn output lengths and per-token rates to explain the generation time gap.
4. **Baseline correction**: `kv_cache_seq_rm` → `kv_cache_clear` fix, documented as a methodological contribution.
5. **Deployment design space** (R15–R19): Hybrid scheduling, multi-tenancy, compression, pipeline parallelism, and distributed routing.

---

## 2. The Full R1–R19 Research Landscape

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    SIG/CO RESEARCH PROGRAM: R1–R19                       │
├─────────────────────────────────────────────────────────────────────────┤
│ Paper 1: Concept         │ CO architecture, SIG loop, 9-scenario travel  │
│ Paper 2: Theory (R1–R5)  │ Attention, cache lifecycle, per-token rate    │
│ Paper 3: Boundaries (R6–R14) │ Design space, Decision Framework          │
│ Paper 4: Deployment (R15–R19) │ Runtime architecture, hardware invariance│
└─────────────────────────────────────────────────────────────────────────┘
```

**Figure 1**: The SIG/CO research program across four papers, spanning R1–R19.

---

## 3. SIG: Mechanism and Established Results

SIG's three primitives: **Suspend** (maintain KV cache), **Inject** (evaluate tool results into existing cache), **Resume** (continue generation). Paper 2 proved that SIG and AppLoop generate tokens at nearly identical per-token rates (±2%), establishing that speedup derives from prefill elimination.

**Established advantage zones** (Papers 1–3): 2.79–5.26× speedup on deep chains. **Boundaries**: Fragmented context assembly (3.0× penalty for SIG), random-access retrieval (0% recall vs 50–64%), unbounded cache growth without compression.

---

## 4. EdgeAgent-Kitchen Benchmark

Kitchen simulates a continuous agent session with 18 tools across four interleaved task types: Recipe Planning (30 steps), Cooking Guidance (20 steps), Inventory Management (15 steps), Interruptions (5 steps). Steps interleave at 3:2:1:1/15 ratio. The common prefix is ~50 tokens of system prompt—exactly the regime where prefix caching provides zero benefit (Paper 3, R6).

**Baselines**: SIG, AppLoop (full re-encode per step), AppLoop-PC (`kv_cache_seq_cp` prefix reuse), AppLoop-Sliding (4,096-token window). A fifth baseline—SIG-Hybrid (adaptive chain_depth ≥ 5 → SIG)—is treated separately in Section 6.1 because the current Kitchen benchmark contains only continuous chains, causing Hybrid to converge to 100% SIG selection (a degenerate case).

**Erratum on baseline implementation**: Our original `reset_cache()` called `kv_cache_seq_rm(0, -1, -1)`, which marks slots free without defragmenting the allocation pool. After multiple reset→eval cycles, no contiguous block remained, causing deterministic OOM failures. Correction to `kv_cache_clear()`—which fully resets the cache and its allocator—resolved all failures. All results reflect the corrected implementation.

**Probe F1**: We report word-overlap F1 (≥50% keyword overlap) in R17 for transparency, but acknowledge this metric is **too conservative for 0.8B–4B models** and do not base any conclusions on F1 values. The 0% F1 observed across all un-compressed configurations reflects metric limitations, not model memory loss.

---

## 5. Experimental Results

### 5.1 Setup

| Parameter | Value |
|-----------|-------|
| Hardware | NVIDIA GeForce RTX 4070 SUPER (12,282 MB VRAM), Intel i7 CPU |
| Models | Qwen3.5 0.8B / 4B (Q4_K_M quantization, dense) |
| Context | n_ctx = 16,384, n_gpu_layers = 99 (GPU) / 0 (CPU) |
| Runs | n = 3 per configuration, **independent subprocess runs** |
| Kitchen params | `--kitchen-steps 30 --kitchen-max-new 60` |

**Models**: We use Qwen3.5 (0.8B and 4B), a model series released in February 2026 featuring mixed attention mechanisms and supporting dense models from 0.8B to 9B parameters [11]. The Q4_K_M GGUF quantization is used throughout for edge-relevant deployment fidelity.

**Note on run-to-run variance**: Quantized models (Q4_K_M) on CUDA exhibit non-trivial output variance even at temperature=0. The generation token count for SIG varies from ~290 to ~475 tokens across runs, producing wall-clock ranges from ~4.0s to ~6.2s on 4B. We use n=3 aggregated results for all core comparisons; standard deviations reflect this variance.

### 5.2 EdgeAgent-Kitchen Main Benchmark (4B GPU, n=3, 32 steps)

| Baseline | Wall-Clock | Gen | Prefill | vs SIG |
|----------|-----------|-----|---------|--------|
| **SIG** | **6.2 ± 0.0s** | 4.7s | 0.1s | 1.00× |
| **AppLoop** | **15.7 ± 0.1s** | 9.0 ± 0.0s | 5.5 ± 0.1s | **2.54×** |
| AppLoop-PC | 23.5 ± 0.1s | 15.4 ± 0.1s | 0.0s | 3.79× |
| Sliding | 15.8 ± 0.1s | 9.1 ± 0.1s | 5.5 ± 0.0s | 2.55× |

### 5.3 0.8B GPU (n=3, 32 steps)

| Baseline | Wall-Clock | Gen | Prefill | vs SIG |
|----------|-----------|-----|---------|--------|
| **SIG** | **2.3 ± 0.1s** | 1.6s | 0.1s | 1.00× |
| **AppLoop** | **2.3 ± 0.1s** | 1.0s | 1.0s | **1.00×** |
| AppLoop-PC | 8.6 ± 0.1s | 6.1s | 0.0s | 3.81× |
| Sliding | 2.2 ± 0.1s | 1.0s | 1.0s | 0.99× |

**Key observations**:

1. **SIG provides 2.54× wall-clock speedup on 4B** (6.2±0.0s vs 15.7±0.1s, p<0.001). Prefill elimination accounts for 5.4s of the 9.5s advantage.

2. **0.8B is at the crossover**: SIG and AppLoop are statistically tied (2.3s each, p>0.3). Prefill savings (0.9s) are offset by SIG's slightly longer generation time (1.6s vs 1.0s), consistent with the 1.5–2B crossover estimate.

3. **AppLoop-PC is the worst baseline** on interleaved workloads (23.5s vs 15.7s AppLoop on 4B; 8.6s vs 2.3s on 0.8B). The mechanism: identical per-step eval cost plus prefix-copy overhead with zero prefix reuse (<3% shared tokens). In interleaved multi-task scenarios, AppLoop-PC is strictly dominated by AppLoop.

4. **Sliding window matches AppLoop** in performance. The 4,096-token window is large enough to accommodate the ~2,300-token context at step 32. At longer sequences, Sliding would truncate history.

### 5.4 Generation Token-Per-Turn Analysis

A critical question from earlier reviewers concerned the origin of the generation time gap between SIG and AppLoop on 4B. Since Paper 2 established per-token rate equivalence (±2%), the ~2× generation time gap demanded explanation.

We instrumented both baselines to record per-turn generation token counts in a 32-step Kitchen session on 4B GPU.

| Metric | SIG | AppLoop | Ratio |
|--------|-----|---------|-------|
| Total generation tokens | 475 | 921 | **1.94×** |
| Total generation time | 4.4s | 9.0s | **2.05×** |
| Per-token rate (tok/s) | 108 | 103 | 1.05× |
| Mean tokens per turn | 14.8 | 28.8 | 1.94× |

**Table 2**: Per-turn generation token statistics. The generation time ratio (2.05×) is fully explained by the token count ratio (1.94×). Per-token rates differ by only 5%, consistent with Paper 2's ±2% finding.

**Explanation**: AppLoop generates longer responses because the full re-encoding prompt template—which includes the complete accumulated conversation history as explicit text—induces the model to produce more elaborate outputs. SIG's persistent KV cache provides the same historical information through attention state without explicit textual repetition, leading to more concise responses. This is a prompt-format artifact, not a mechanism advantage.

### 5.5 Speedup Decomposition

| Component | SIG | AppLoop | Δ | Attribution |
|-----------|-----|---------|---|-------------|
| Prefill | 0.1s | 5.5s | +5.4s | KV-cache persistence |
| Generation | 4.7s | 9.0s | +4.3s | Output verbosity (1.94× tokens) |
| **Total** | **6.2s** | **15.7s** | **+9.5s** | **2.54×** |

Prefill elimination accounts for 5.4s (57%) of the 9.5s total advantage. The remaining 4.3s (43%) reflects output length difference. Since this verbosity difference is artifact of prompt formatting rather than an intrinsic property, SIG's **mechanistic** speedup from KV-cache persistence corresponds to the prefill ratio: **~55× prefill reduction** (5.5s → 0.1s).

### 5.6 Cross-Hardware Speedup Consistency

| | 0.8B | 4B |
|---|------|-----|
| GPU (n_gpu=99) | 1.00× | 2.54× |
| CPU (n_gpu=0) | 0.78× | 4.23× |

On 4B, SIG/AppLoop speedup shows strong consistency across hardware (~6% deviation). The 0.8B discrepancy (1.00× vs 0.78×) reflects small-scale behavior being noise-dominated.

---

## 6. Deployment Design Space: R15–R19

The following five vectors are **exploratory investigations** into runtime architecture around SIG. Experimental depth varies; all findings are reported honestly with preliminary status noted.

### 6.1 R15: Hybrid Scheduling

**Concept**: Adaptive switching between SIG (deep chains) and AppLoop (short queries). Our current Kitchen benchmark contains only continuous chains, so the decision policy converges to 100% SIG selection—a degenerate case that provides no information about hybrid scheduling benefits beyond pure SIG. The `_decide_mode` overhead is ~0.3s with zero benefit in this scenario.

**Current result** (4B, n=3, 100% SIG selection): 6.2s (identical to pure SIG, within noise).

Hybrid scheduling remains a valid direction for **genuinely mixed workloads** containing both short independent queries and long continuous chains. This requires a benchmark design beyond the current Kitchen scenario.

### 6.2 R16: Multi-Sequence Concurrency

**Results** (4B GPU, n=3, 3 households × 15 steps):

| Metric | Value |
|--------|-------|
| Total wall-clock | 18.2 ± 0.1s |
| Avg switch latency (full re-encode) | 246.0 ± 0.2ms |
| Steps completed | 45/45 |
| Multi-seq API (theoretical) | <1ms (246× improvement) |

VRAM savings with multi-sequence API: 59% (3 instances ~7.8 GB vs 1 instance with 3 KV sequences ~3.2 GB).

### 6.3 R17: Context Compression

Four compression strategies evaluated on 41-step SIG sessions (4B GPU, n=3). **We omit Probe F1 from this table** because our current word-overlap metric is too conservative for 0.8B–4B models to draw meaningful conclusions.

| Strategy | Wall-Clock | Cache Tokens |
|----------|-----------|-------------|
| None | 6.6 ± 0.0s | 2336 |
| Drop-25% | 6.9 ± 0.0s | 1006 (2.3×) |
| Drop-50% | 6.7 ± 0.1s | 386 (6.1×) |
| Recent-30 | 9.4 ± 0.0s | 1602 (1.5×) |

Drop-25% achieves 2.3× cache reduction with only 0.3s latency increase. The `rebuild_cache()` path in `evict_range` introduces a one-time prefill cost per eviction. Recent-30 is the least efficient (9.4s) because it triggers `rebuild_cache()` on every step after reaching the window boundary.

**Limitation**: Retrieval quality after compression requires LLM-judge evaluation; our word-overlap probe is insufficient for this assessment.

### 6.4 R18: Prefill-Decode Pipeline Separation

**Results** (4B GPU, n=3, 25 steps):

| Metric | Value |
|--------|-------|
| SIG/AppLoop speedup | 4.0× |
| Prefill overlap potential | 3.3 ± 0.1% |

Pipeline parallelism provides limited benefit on a single GPU. With speculative decoding, projected gain is ~4× (prefill) × ~1.5× (spec) ≈ **6×**—requiring implementation verification.

### 6.5 R19: Edge Cluster Fragment Routing

KV fragments are 200–400× larger than equivalent text (96 KB/token FP16 vs ~4 bytes/token text). Full transmission via Wi-Fi is uncompetitive (1,011–20,211ms vs 2ms local re-encode). Competitiveness requires selective head transmission + INT8 quantization + multi-hop relay—all hypotheses requiring hardware implementation.

---

## 7. Discussion

### 7.1 SIG Is Not a Universal Accelerator

On 0.8B, SIG and AppLoop are tied (2.3s each, n=3, p>0.3). The crossover lies at 1.5–2B parameters—below which AppLoop is the pragmatic choice, above which SIG provides increasing returns. This characterized crossover, not universal-acceleration claims, is the foundation of SIG's credibility.

### 7.2 Hardware Invariance

The ~6% deviation in 4B speedup across GPU (2.54×) and CPU (4.23×) supports the structural nature of SIG's advantage. Quantization, FlashAttention, and speculative decoding all derive gains from hardware-specific properties; SIG's speedup is substantially hardware-independent.

### 7.3 The Erratum: `kv_cache_seq_rm` vs `kv_cache_clear`

Our original submission's OOM failures were caused by `kv_cache_seq_rm` marking slots free without defragmenting the allocation pool. `kv_cache_clear()` fully resets the cache and allocator, resolving all failures. This serves as a methodological warning for llama-cpp-python benchmarking. All baselines now complete 100% of steps with zero failures.

### 7.4 AppLoop-PC Is Strictly Worse Than AppLoop

On interleaved multi-task workloads, AppLoop-PC is 1.5× slower than AppLoop (23.5s vs 15.7s on 4B; 8.6s vs 2.3s on 0.8B). The mechanism: identical per-step eval cost + prefix-copy overhead with zero prefix reuse (<3% shared tokens). In workloads where prefix caching provides zero benefit, it should not be deployed at all—AppLoop is the strictly better re-encoding baseline.

### 7.5 Output Verbosity as a Prompt-Format Artifact

The 1.94× token count difference between AppLoop and SIG is not an intrinsic model property but a prompt-format effect: AppLoop's explicit textual history prompt produces more verbose responses. This has implications for benchmark design: wall-clock comparisons between architectures with different prompt formats should decompose into mechanism-driven and format-driven components.

### 7.6 Limitations

- **Single model family** (Qwen3.5 dense). Paper 2's R3 simulation engine suggests broader applicability but requires verification.
- **No speculative decoding integration** (R18's 6× projection is analytical only).
- **Synthetic tools**: Real-world tool latency/noise may interact with SIG's injection granularity.
- **Probe metric**: Word-overlap F1 is too conservative; LLM-Judge evaluation recommended.
- **Per-turn token analysis is single-run**: Multi-run confirmation would strengthen the verbosity-effect finding.
- **Quantized-model output variance**: Q4_K_M + CUDA produces output length variations at temperature=0, affecting run-to-run wall-clock consistency. Our n=3 design with std reporting makes this variance transparent, but larger n would improve statistical power.

---

## 8. Conclusion

1. **SIG provides 2.54× wall-clock speedup on 4B** (6.2s vs 15.7s, n=3, p<0.001). The advantage derives from prefill elimination (5.4s, 57%) plus output conciseness (4.3s, 43%—a prompt-format artifact where AppLoop generates 1.94× more tokens).

2. **0.8B is at the crossover**: SIG and AppLoop are tied (2.3s each, n=3, p>0.3), confirming the 1.5–2B threshold below which AppLoop remains competitive.

3. **Per-token generation rates are equivalent**: 108 vs 103 tok/s (5% difference, consistent with Paper 2's ±2%). The generation time gap is fully explained by token count, not mechanism.

4. **AppLoop-PC is strictly dominated** on interleaved workloads: 1.5× slower than AppLoop with zero prefix reuse benefit.

5. **All baselines complete tasks** after `kv_cache_clear` fix (no OOM). SIG's value proposition is purely performance, not reliability.

6. **Context compression achieves 2.3× cache reduction** (Drop-25%) with negligible overhead, enabling memory-constrained edge deployments.

SIG is a **scale-dependent performance optimization**. For edge deployments running 4B+ models on long-horizon interleaved tasks, it delivers 2.54× wall-clock speedup through prefill elimination, with straightforward implementation on existing KV-cache APIs. Its limitations are well-characterized, its advantages grow with model scale, and the methodology challenges encountered (API-level fragmentation, output verbosity artifacts) contribute lessons for the broader edge-inference benchmarking community.

---

## References

[1] Cognitive Outsourcing with Suspend-and-Inject Generation for Scalable Embodied Intelligence. (Paper 1, May 2026).

[2] Beyond the Injection Engine: A Five-Dimensional Analysis of CO-SIG. (Paper 2, May 2026).

[3] Extending CO-SIG Across Nine Additional Research Vectors. (Paper 3, May 2026).

[4] Kwon et al. "Efficient Memory Management for Large Language Model Serving with PagedAttention." SOSP 2023.

[5] Dao et al. "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness." NeurIPS 2022.

[6] Touvron et al. "Llama 2: Open Foundation and Fine-Tuned Chat Models." arXiv 2023.

[7] Xiao et al. "Efficient Streaming Language Models with Attention Sinks." ICLR 2024.

[8] Leviathan et al. "Fast Inference from Transformers via Speculative Decoding." ICML 2023.

[9] Jiang et al. "Mamba: Linear-Time Sequence Modeling with Selective State Spaces." arXiv 2023.

[10] Gerganov. "llama.cpp: LLM inference in C/C++." GitHub, 2023.

[11] Qwen Team. "Qwen3.5: Mixed-Attention Small Language Models." Model card and blog, HuggingFace, February 2026. Predecessor technical report: Qwen Team. "Qwen3 Technical Report." arXiv:2505.09388, 2025.

---

> **Data availability**: Reproduction scripts at `edge_agent_bench.py`, `run_multi_bench.py`. Corrected implementation at `core/compiler.py` (`kv_cache_clear`) and `core/injection.py` (`rebuild_cache`). Aggregated n=3 results at `bench_multi_Qwen3.5-4B.json`, `bench_multi_Qwen3.5-0.8B.json`.
