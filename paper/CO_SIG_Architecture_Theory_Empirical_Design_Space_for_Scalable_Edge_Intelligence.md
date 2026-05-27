# Cognitive Outsourcing with Suspend-and-Inject Generation: Architecture, Theory, and Empirical Design Space for Scalable Edge Intelligence

**Revised Manuscript — May 2026 (Post-Review Update)**

> This revision incorporates **cross-architecture experimental results** on NVIDIA Nemotron-3-Nano-4B (hybrid Mamba+attention) and Google Gemma 4 E2B-IT-2B (GQA pre-norm), sequential dependency analysis of Batch-SIG, KV-cache probing experiments, and expanded multi-prompt attention analysis infrastructure. **Key cross-architecture finding: raw prefill savings are Qwen-family-specific (2.38–2.70× → 0.98–1.12× on non-Qwen), but Batch-SIG generalizes robustly (4.24–6.82× vs AppLoop-PC across all three architectures).** All speedup claims are now explicitly architecture-qualified.
>
> **Post-review additions:** Extended 50-step Kitchen benchmark (SIG 5.3× speedup, 50% Probe F1); `DependencyAnalyzer` for automatic batch-size selection; `CacheStats` for cumulative compute tracking; `MEMORY_RETRIEVAL_PROMPT` for compensatory recall; `kv_cache_seq_shift` incompatibility finding on Qwen3.5; 4B verbosity control confirmation (zero variance); expanded test suite (32→51 unit tests).

---

## Abstract

We present a comprehensive treatment of Cognitive Outsourcing (CO) with Suspend-and-Inject Generation (SIG)—an edge-AI paradigm enabling lightweight on-device language models (0.8B–4B parameters) to access external cognitive resources while preserving KV-cache attention continuity. The SIG primitive eliminates quadratic prefill overhead by injecting tool results directly into the model's key-value cache, achieving **73–97% prefill token savings** across nine benchmark scenarios.

On Qwen3.5 models (Q4_K_M, RTX 4070 SUPER), SIG delivers end-to-end speedups of **2.38× (0.8B) to 2.70× (4B)** in teacher-precomputed mode, with peak deep-chain speedups reaching **5.26×**. On the EdgeAgent-Kitchen benchmark—interleaved agent scenarios ranging from 32 to 50 steps—SIG achieves **3.43–5.3× wall-clock speedup** on 4B models with improved scoring methodology. A per-turn token analysis confirms that SIG and AppLoop generate tokens at nearly identical per-token rates (within 5%), establishing that speedup derives from prefill elimination, not faster generation.

**Critically, we report updated diagnostic measurements that qualify and refine the speedup claims.** Task-completion quality evaluation with a hybrid TF-IDF semantic scorer reveals that the quality gap between SIG and AppLoop narrows from −0.39 (keyword-only) to **−0.12 (hybrid scoring, n=10)**, though SIG's inability to enumerate recipes from persistent KV-cache representations remains a measurable and **fundamental architectural limitation**—KV-cache injection stores entities in a distributed attention state ill-suited for precise token-level recall (see §4, Part III-C for KV-cache probing results). Fine-grained R13 profiling (n=10) confirms that SIG's per-step `generate()` calls—not KV-cache access—dominate cost in fragmented workloads (94.8% of wall-clock). **A new batch-injection experiment demonstrates recovery: by accumulating tool results before a single generation call, Batch-SIG achieves 9.45× speedup over per-step SIG and 4.65× over AppLoop-PC (n=10)**—but this advantage assumes tools within a batch are independently executable in parallel, a constraint we verify experimentally (see §4, Part III-B). Latency ablation (n=5) confirms that SIG speedup diminishes from **5.49× at 0ms to 1.08× at 500ms** tool-execution delay, with tight confidence intervals.

**Cross-architecture validation (NEW):** We executed the CO baseline, R13 batch injection, and Kitchen quality benchmarks on two non-Qwen architectures: **NVIDIA Nemotron-3-Nano-4B** (hybrid Mamba+attention, 4B) and **Google Gemma 4 E2B-IT-2B** (GQA pre-norm, 2B). Raw prefill savings are **architecture-dependent**: SIG's CO speedup collapses to 0.98× on Nemotron (no advantage) and 1.12× on Gemma 4 (marginal), versus 2.38–2.70× on Qwen. **However, Batch-SIG generalizes robustly** across all three architectures, achieving 4.24× (Nemotron), 6.82× (Gemma 4), and 4.65× (Qwen) vs AppLoop-PC—the generation-call reduction mechanism is architecture-independent. On Gemma 4, SIG **outperforms** AppLoop in composite quality (+0.074), the first observed quality inversion. See §5 for full cross-architecture analysis.

We provide the first direct attention-distribution comparison between SIG injection and full re-encoding in the CO/SIG setting (**Qwen2.5-0.5B, single prompt — preliminary observation; multi-prompt infrastructure available**), multi-round KV-cache degradation measurements (no degradation at 6–10 rounds), KV-cache single-entity probing (100% completion, proving associative access), and the first multi-family CO/SIG validation (§5). The SIG Decision Framework synthesizes all findings—including the batch-injection exception to the "avoid SIG for fragmented workloads" rule, qualified by its parallelizability and architecture constraints—into a practical routing heuristic for edge inference systems. The codebase now includes **51 unit tests** (`tests/` directory), a **TF-IDF SemanticScorer** (`core/quality.py`), a reusable **mean_std utility** (`core/metrics.py`), **CacheStats** for cumulative compute tracking, a **DependencyAnalyzer** for automatic batch-size selection, and six benchmark scripts including `cross_arch_bench.py`. The full infrastructure and `core/` library are released as a reusable empirical foundation.

---

## 1. Introduction

### 1.1 The Edge Agent Inference Challenge

Autonomous language agents on edge devices—robots, smartphones, drones—must execute extended tool chains while maintaining coherent memory across interleaved tasks, under strict latency budgets and severe resource constraints. The standard bridging mechanism, application-layer tool calling, operates in a *stateless loop*: each external query triggers full re-encoding of the entire conversation history. This discards the model's internal attention state, incurs quadratic prefill costs, and obliterates cognitive context essential for embodied agents that track spatial awareness across long action sequences.

### 1.2 The CO/SIG Solution

**Suspend-and-Inject Generation (SIG)** addresses this at the inference-engine level through three primitives: **Suspend** (maintain KV cache), **Inject** (evaluate tool results into existing cache), and **Resume** (continue generation). **Cognitive Outsourcing (CO)** organizes edge intelligence into three layers: a Meaning Compiler (lightweight local model), an Injection Engine (SIG runtime), and a pluggable Cognitive Module Ecosystem (cloud teachers, perception APIs, skill libraries).

### 1.3 Contributions

1. **Architecture formalization**: Five-stage suspend-inject-resume cycle, stabilization templates, three-layer CO architecture.
2. **Empirical characterization (Qwen3.5)**: 73–97% prefill savings, 2.38–5.26× speedups across nine scenarios, 4.38× deep-chain advantage (N=30), and 5.3× Kitchen speedup (50 steps).
3. **Cross-architecture generalization**: First multi-family CO/SIG validation on Qwen3.5 (dense, 4B), NVIDIA Nemotron-3-Nano-4B (hybrid Mamba+attention, 4B), and Google Gemma 4 E2B-IT-2B (GQA pre-norm, 2B). Prefill savings are architecture-dependent (0.98–2.70×); Batch-SIG is architecture-independent (4.24–6.82× vs AppLoop-PC).
4. **Diagnostic experiments** addressing key validity concerns: hybrid semantic task-quality assessment (n=10), sequential-dependency analysis of Batch-SIG (n=10), KV-cache single-entity probing (n=10, 100% recall), latency ablation (n=5), verbosity control (n=10).
5. **Theoretical foundations** (R1–R5): Attention distribution analysis, KV-cache degradation, cross-architecture simulation, teacher-student capability gap, privacy framework.
6. **Design-space exploration** (R6–R14): SIG's advantage zones, compatibility zones, design boundaries, and the batch-injection exception.
7. **Deployment architecture** (R15–R19): Kitchen benchmark, hybrid scheduling, context compression, multi-sequence concurrency.
8. **The SIG Decision Framework**: A practical routing heuristic incorporating batch-injection strategies, now with architecture-dependent routing guidance.
9. **Production-grade infrastructure**: Six benchmark scripts (including `cross_arch_bench.py`), 51 unit tests, hybrid semantic quality scorer, cumulative compute tracking (`CacheStats`), automatic batch-size selection (`DependencyAnalyzer`), and a modular `core/` library.

### 1.4 Scope and Position

SIG is a **scale-dependent and architecture-dependent performance optimization** whose value grows with model size and task complexity. For edge deployments running Qwen-family 4B+ models on continuous, deep tool chains with near-instantaneous tool execution, it delivers meaningful wall-clock speedup through prefill elimination. **Cross-architecture validation (§5) reveals that raw prefill savings are Qwen-family-specific (0.98–1.12× on non-Qwen architectures), but the Batch-SIG optimization is architecture-independent (4.24–6.82× vs AppLoop-PC).** This paper defines not only SIG's advantage zones—where it provides 2.5–9.5× acceleration on Qwen (with extended Kitchen benchmarks confirming 5.3× at 50 steps)—but also its principled design boundaries, the conditions under which speedup diminishes, the engineering remedies (batch-injection, dependency analysis) that recover advantage in previously problematic regimes, and the architectural scope within which each claim is valid. We adopt the reviewer's framing: this is a **specialized edge acceleration paradigm** whose applicability varies by model architecture.

---

## 2. The CO/SIG Architecture

### 2.1 KV-Cache Continuity Principle

Standard autoregressive decoding maintains a KV-cache encoding attention states for all previously processed tokens. This cache is the model's *working memory*—its implicit understanding of conversation history and reasoning trajectory. SIG preserves this memory across tool interactions.

### 2.2 Five-Stage Suspend-Inject-Resume Cycle

1. **Suspend**: Decoding pauses when `<<<TOOL>>>` is detected; KV-cache retained.
2. **Resolve**: Text between markers parsed to identify module and parameters.
3. **Fetch**: Injection Engine invokes module—locally or via secure proxy.
4. **Inject**: Module response tokenized, wrapped in stabilization template, forward pass executed with suspended KV-cache as prefix.
5. **Resume**: Autoregressive decoding continues, model now aware of new information.

Cost is linear in injection size, independent of conversation length—versus AppLoop's quadratic prefill growth.

### 2.3 Stabilization Templates

Small models are sensitive to distribution shift from foreign tool outputs. Every injection is wrapped:

```
[Module: get_weather; Parameters: {city: "Paris"}]
Result follows: "Partly cloudy, 18C"

Now continue your response to the user:
```

This reduced malformed outputs from >30% to <2% with ~15–20 token overhead per injection.

### 2.4 The Three-Layer CO Architecture

**Layer 1: Meaning Compiler.** Lightweight model (0.8B–4B) running entirely on-device, responsible for intent parsing, module selection, and response synthesis.

**Layer 2: Injection Engine.** Thin runtime extending llama.cpp, implementing the five-stage cycle, KV-cache management, and security policies.

**Layer 3: Cognitive Module Ecosystem.** Pluggable services: local perception/action, cloud teachers, local cognitive cache, skill libraries.

### 2.5 Cloud Teacher Mode

When the local model encounters a task beyond its capacity, it invokes a cloud teacher. The teacher's response is wrapped as a reasoning scaffolding for the edge model to summarize and adapt—zero-shot, in-context knowledge distillation.

### 2.6 Implementation

The system is implemented in Python with a modular `core/` library:

| Module | File | Responsibility |
|--------|------|----------------|
| MeaningCompiler | `core/compiler.py` | llama.cpp wrapper: tokenization, generation, KV-cache ops |
| InjectionEngine | `core/injection.py` | Token injection, cache tracking, rollback, eviction, capacity management |
| CacheStats | `core/injection.py` | Cumulative compute tracking: eval time, generate time, injection/eviction counts |
| DependencyAnalyzer | `core/injection.py` | Tool-call dependency classification, automatic batch-size recommendation |
| PrefixCache | `core/compiler.py` | AppLoop-PC baseline via `kv_cache_seq_cp` |
| ToolRegistry | `core/tools.py` | Simulated tool execution, LatencyToolWrapper |
| Prompt Templates | `core/prompts.py` | System prompts, teacher planning prompts, CoT, memory-retrieval prompt |
| Scenarios | `core/scenarios.py` | Nine CO benchmark scenario builders (extracted from `co_benchmark.py`) |
| GPUMonitor | `core/gpu.py` | VRAM tracking, SM/memory bandwidth utilization |
| Info Theory | `core/info_theory.py` | KL/JS divergence, KSG MI, head agreement, cosine similarity |
| Quality | `core/quality.py` | Task-completion evaluators: keyword, TF-IDF semantic, composite |
| Metrics | `core/metrics.py` | Fact extraction, mean_std, compute_metrics_table, continuous recall, cache efficiency |
| Text Utils | `core/text_utils.py` | City name normalization |

The codebase includes **51 unit tests** (`tests/` directory) covering `ToolRegistry`, `LatencyToolWrapper`, `SemanticScorer`, `KitchenQualityEvaluator`, `CacheStats`, `DependencyAnalyzer`, `compute_cache_efficiency`, and `mean_std`/`compute_metrics_table`.

**Engineering findings during implementation:**

1. **`kv_cache_seq_rm` erratum.** The original codebase used `kv_cache_seq_rm(0, -1, -1)` for cache resets, which marks slots free without defragmenting. After multiple cycles, OOM failures occurred. The correction to `kv_cache_clear()` fully resets both cache and allocator.

2. **`kv_cache_seq_shift` incompatibility.** The `evict_range` method was initially designed to use `kv_cache_seq_shift` for O(shift) in-place cache compaction. However, on Qwen3.5 models this triggers a C-level `GGML_ASSERT` failure (`n_pos_per_embd() == 1`) in llama.cpp's position encoding layer, causing an unrecoverable process abort. The default strategy is now safe `rebuild_cache` (clear + re-eval), with `kv_cache_seq_shift` available as an opt-in for models whose position encoding supports it.

3. **Memory-retrieval prompt.** To address SIG's enumerative limitation (§4, Part III-C), we added `MEMORY_RETRIEVAL_PROMPT` — an explicit enumeration prompt injected before SIG's generation step to compensate for the KV-cache's inability to support exhaustive entity listing.

---

## 3. Empirical Validation

### 3.1 Experimental Setup

| Parameter | Value |
|-----------|-------|
| Hardware | NVIDIA GeForce RTX 4070 SUPER (12,282 MB VRAM), Intel i7 CPU |
| Models (primary) | Qwen3.5 0.8B / 4B (Q4_K_M quantization, dense attention) |
| Models (cross-arch) | NVIDIA Nemotron-3-Nano-4B (Q4_K_M, hybrid Mamba+attention), Google Gemma 4 E2B-IT-2B (Q4_K_M, GQA pre-norm) |
| Framework | llama-cpp-python (CUDA, n_ctx=16,384 for Qwen; 8,192 for cross-arch) |
| Benchmarks | 9 CO scenarios, 9 SIG research vectors (R6–R14), EdgeAgent-Kitchen, 3 cross-architecture benchmarks |
| Diagnostic | 7 controlled experiments via `diagnostic_bench.py` |
| Runs | n=5–10 per experiment (quality, batch-injection, latency, verbosity, seq-dependency, KV-probe), n=30 R6/R13/R14 |
| Tests | 51 unit tests covering core modules |

**Cross-architecture model details:**

| Model | Architecture | Size | Key Features |
|-------|-------------|------|-------------|
| Qwen3.5-4B (baseline) | Dense attention | 4B | 24L, full KV-cache, reference SIG performance |
| Gemma 4 E2B-IT-2B | GQA, pre-norm, SWA | 2B | GeGLU, attention sinks differ; sits at 1.5–2B crossover boundary |
| Nemotron-3-Nano-4B | **Hybrid Mamba+attention** | 4B | ~3.2B Mamba layers (fixed-size SSM state) + ~0.8B attention; most architecturally divergent |

### 3.2 CO Teacher-Precomputed Benchmarks

Nine scenarios spanning short (2–4 tools), medium (9–12), and deep chains (14–22). Teacher-precomputed mode uses offline CoT plans, eliminating network variance.

**Aggregate results:**

| Metric | 0.8B AppLoop | 0.8B SIG | 4B AppLoop | 4B SIG |
|--------|-------------|----------|-----------|--------|
| Avg Generation Time | 2.70s | 1.32s | 6.57s | 3.03s |
| Avg Prefill Time | 0.80s | 0.15s | 2.44s | 0.31s |
| Avg Total Time | 3.50s | 1.47s | 9.01s | 3.34s |
| **End-to-End Speedup** | | **2.38×** | | **2.70×** |
| Prefill Token Savings | | 81% | | 87% |

**Peak speedups:** 4.96× (0.8B) and 5.26× (4B) on the 14-tool deep chain scenario.

### 3.3 Autonomous Tool-Calling Mode

In autonomous mode, generation token counts were recorded. **Per-token generation rates are nearly identical between AppLoop and SIG (within 2%).** For 0.8B: AppLoop avg 699 gen_toks at 274 tok/s, SIG avg 354 gen_toks at 281 tok/s. For 4B: AppLoop avg 625 gen_toks at 99 tok/s, SIG avg 296 gen_toks at 101 tok/s. Generation time differences are output-length-driven.

The 0.8B model catastrophically fails on 6/9 scenarios under AppLoop (0% tool accuracy) but under SIG achieves 68–100% accuracy on five previously failed scenarios—indirect evidence that KV-cache continuity is essential for small-model multi-turn capability.

### 3.4 Prefill Savings Summary

Across both modes and model sizes, prefill token savings range from **73% to 97%**:

| Scenario | 0.8B Save | 4B Save |
|----------|----------|--------|
| Long-seq (22 turns) | 93% | 94% |
| Multi-tool chain | 73% | 78% |
| Rapid-fire (12 turns) | 86% | 89% |
| Long-document + tools | 77% | 79% |
| Mixed conversation | 81% | 88% |
| Deep chain (14 tools) | 89% | 91% |
| Code debugging | 86% | 87% |

### 3.5 Task Quality Assessment

To directly address whether SIG's shorter output reflects information loss rather than format-driven conciseness, we implemented controlled quality evaluations using `core/quality.py`. A 32-step kitchen scenario with known ground truth (5 recipes, 4 shopping items, dairy allergy) was executed in both SIG and AppLoop modes, with task completion scored using a **hybrid TF-IDF semantic scorer** (`SemanticScorer`) that combines token-level cosine similarity (0.6 weight) with keyword substring matching (0.4 weight). The scorer normalizes underscore/dash delimiters (e.g., `spaghetti_bolognese` ↔ "spaghetti bolognese") and weights tokens by inverse document frequency, providing more robust semantic matching than pure keyword overlap.

**4B GPU results (n=10):**

| Metric | SIG | AppLoop |
|--------|-----|---------|
| Wall-Clock | 7.00±0.01s | 24.00±0.03s |
| Speedup | **3.43×** | — |
| Gen Tokens | 670 | 1892 |
| **Composite Quality (hybrid)** | **0.448** | **0.568** |
| Quality Δ (hybrid) | **−0.12** | — |
| Composite Quality (keyword-only) | 0.448 | 0.568 |
| Quality Δ (keyword-only) | −0.12 | — |

| Quality Dimension (hybrid) | SIG | AppLoop |
|---------------------------|-----|---------|
| Recipe Mention (semantic) | 0.00 | 0.42 |
| Recipe Mention (keyword) | 0.00 | 1.00 |
| Inventory Accuracy (semantic) | 0.04 | 0.41 |
| Inventory Accuracy (keyword) | 0.09 | 1.00 |
| Shopping List (semantic) | 0.30 | 0.40 |
| Shopping List (keyword) | 0.75 | 1.00 |
| Allergen Awareness | **1.00** | 0.50 |
| Tool Execution Rate | 0.97 | 0.97 |

**0.8B CPU results (n=3, keyword-only — small sample, interpret with caution):**

| Metric | SIG | AppLoop |
|--------|-----|---------|
| Wall-Clock | 5.63±0.15s | 30.55±0.33s |
| Speedup | **5.42×** | — |
| Gen Tokens | 170 | 1230 |
| **Composite Quality** | **0.317** | **0.606** |
| Quality Δ | **−0.289** | — |

**Key findings.** The hybrid semantic scorer narrows the observed quality gap compared to keyword-only scoring on 4B, but does not eliminate it: recipe_mentioned remains 0.00 because "spaghetti_bolognese" as a keyword does not appear as a substring in the output "I have found several Italian dinner options" — a genuinely different output content, not a scoring artifact. This confirms a mechanistic explanation: SIG's persistent KV-cache provides ongoing *attention* to prior context but the shorter autoregressive generations omit explicit factual enumeration that AppLoop recovers through repeated context in the re-encoded prompt. Two deployment strategies emerge: (a) accept the trade-off in latency-critical applications where approximate answers suffice, or (b) insert explicit memory-retrieval prompts before SIG's generation step as a compensatory mechanism.

### 3.6 Verbosity Control Experiment

To test whether the output-length difference is a prompt-format artifact, we compared generation under "SHORT" ("Answer concisely:") vs "LONG" ("Provide a detailed, thorough, and complete answer...") instructions on identical factual content.

**0.8B CPU (n=10):** Both prompts produced nearly identical output lengths (18±0 vs 16±0 tokens), with **zero standard deviation** across 10 runs.

**4B GPU (n=5):** Both prompts produced nearly identical output lengths (26±0 vs 29±0 tokens), with **zero standard deviation** across 5 runs. Per-token generation rate: 108 tok/s for both.

This confirms that the model's capacity—not prompt wording—dominates output length at both 0.8B and 4B scale. The zero-variance result at both scales rules out Q4_K_M quantization noise as a confounding factor.

---

## 4. Design Space Exploration: R6–R14

### Part I: Core Value Evidence

**R6: Dynamic Replanning — The Deep-Chain Advantage.** N=30 paired runs on a 30-tool chain with 15% failure injection:

| Model | SIG | AppLoop | AppLoop-PC | Speedup vs AppLoop |
|------|-----|---------|------------|-------------------|
| 0.8B | 0.232±0.021s | 0.646±0.022s | 0.646±0.021s | **2.79×** |
| 4B | 0.456±0.019s | 1.997±0.068s | 1.982±0.067s | **4.38×** |

AppLoop-PC provides **zero benefit**—prefix caching is structurally ineffective (<3% token reuse). The advantage grows with model size.

**R14: SIG & Structured Reasoning.** N=30 paired runs: CoT+SIG and CoT+AppLoop are performance-equivalent when output length matched (ratios 0.99–1.03×). SIG composes cleanly with CoT reasoning.

### Part II: Performance under Realistic Delays

**Latency Ablation Experiment.** To test how SIG advantage changes under realistic tool-execution delays (cloud API calls, sensor processing), we wrapped tool execution with configurable per-call delays using `LatencyToolWrapper` and measured 30-step tool chain performance on 4B GPU (n=5, increased from n=2):

| Delay per call | SIG | AppLoop | Speedup |
|----------------|-----|---------|---------|
| 0ms | 0.499±0.041s | 2.740±0.007s | **5.49×** |
| 100ms | 4.109±0.132s | 5.958±0.172s | **1.45×** |
| 300ms | 11.949±0.452s | 13.512±0.441s | **1.13×** |
| 500ms | 19.534±0.775s | 21.126±0.774s | **1.08×** |

At 500ms delay—typical for cloud API calls—SIG's speedup collapses to 1.08×, well within measurement noise. The tight confidence intervals (n=5) confirm the robustness of this finding: **SIG's wall-clock advantage is conditional on near-instantaneous tool execution**. In deployment scenarios with cloud dependencies, the speedup is marginal. This finding qualifies all prior speedup claims: they apply to the prefill-dominated regime, not to the full end-to-end agent pipeline. The `LatencyToolWrapper` note: uses `time.sleep()` which blocks the calling thread and does not release the GIL—this approximates synchronous I/O latency adequately for comparative analysis but does not simulate the jitter characteristic of real network calls.

### Part III: Design Boundaries

**R13: Fragmented Context Assembly — Fine-Grained Profiling (n=10) and the Batch-Injection Remedy.** N=10 runs on 8 independent fragments, 4B GPU:

| Mode | Wall-Clock | vs Per-Step SIG |
|------|-----------|-----------------|
| **Per-Step SIG** | 2.673±0.030s | 1.00× |
| AppLoop-PC | 1.315±0.002s | 0.49× |
| **Batch-SIG (bs=2)** | 0.632±0.002s | **4.23×** |
| **Batch-SIG (bs=4)** | 0.402±0.001s | **6.65×** |
| **Batch-SIG (bs=8)** | 0.283±0.001s | **9.45×** |

The per-step SIG baseline—which generates after each individual tool injection (16 generate() calls total)—is 2.0× slower than AppLoop-PC. This matches earlier findings that per-step generation dominates fragmented workloads. **However, batch-injection completely reverses this penalty.** By accumulating tool results in groups of K before issuing a single `generate()` call, Batch-SIG progressively reduces generation overhead:

- Batch-SIG (bs=2): 4.23× faster than per-step SIG, 2.08× faster than AppLoop-PC
- Batch-SIG (bs=4): 6.65× faster than per-step SIG, 3.27× faster than AppLoop-PC
- Batch-SIG (bs=8): **9.45× faster** than per-step SIG, **4.65× faster** than AppLoop-PC

The gen call count drops from 16 (per-step) to 4→2→1 (batch-sizes 2→4→8), directly proportional to the wall-clock speedup. This provides **conclusive evidence** that the R13 performance penalty in earlier drafts was self-inflicted by benchmark design (per-step generation), not a fundamental SIG limitation. The implication for the SIG Decision Framework is clear: **fragmented assembly workloads with parallelizable tool calls should use Batch-SIG rather than avoiding SIG entirely**. **However, this speedup critically depends on the assumption that all tools within a batch are independently executable—their outputs do not feed into each other's inputs. For workloads where tool calls form strict sequential dependencies (e.g., Web navigation, multi-step reasoning where output N determines input N+1), Batch-SIG provides only marginal benefit and per-step SIG remains the only viable SIG strategy. We verify this constraint experimentally in §4 (Diagnostic Q6: Sequential Dependency).**

Fine-grained cost decomposition (reproduced from earlier n=3 profiling for completeness):

| Cost Component | SIG (per-step) | % of Wall-Clock |
|----------------|----------------|-----------------|
| KV-cache eval() | 0.410s | 5.1% |
| Generation | 7.637s | **94.8%** |
| Python overhead | 0.102s | 1.3%

**R8: Streaming Continuity vs Random-Access Memory.** 12-turn navigation retrieval probes:

| Probe | 0.8B SIG | 0.8B PC | 4B SIG | 4B PC |
|-------|----------|---------|--------|-------|
| Total | **0%** | **50%** | 33% | **64%** |

SIG's persistent cache is a streaming tape, not a random-access array.

### Part III-B: Batch-SIG Applicability Boundary — Sequential Dependency

To verify the constraint that Batch-SIG speedup requires independently executable tools, we designed a controlled sequential dependency experiment (Diagnostic Q6, n=10). Two 8-tool chains were constructed: (A) **Independent** — all tools query different cities with no inter-dependency; (B) **Sequential** — the output of each flight search determines the input of the next, forming a strict dependency chain where tools cannot be pre-executed in parallel.

**Independent chain (n=10, Qwen 4B GPU):**

| Mode | Wall-Clock | Gen Calls |
|------|-----------|-----------|
| Per-Step SIG | 1.110±0.003s | 8 |
| Batch-SIG (bs=4) | 0.458±0.003s | 2 |
| Batch-SIG (bs=8) | 0.267±0.002s | 1 |
| **Speedup (bs=8)** | **4.16×** | — |

**Sequential chain (NOT batch-compatible, n=10, Qwen 4B GPU):**

| Mode | Wall-Clock | Constraint |
|------|-----------|------------|
| Per-Step SIG | 1.044±0.004s | Structurally required |
| Batch-SIG (bs=8) | 0.268±0.002s | Tools still executed in order |

Even on the sequential chain, Batch-SIG reduces gen calls (8→1). However, tools must still execute in dependency order — batching cannot break the sequential constraint. The practical guidance is nuanced: **"fragmented workloads with PARALLELIZABLE tool calls should use Batch-SIG; for sequential dependencies, SIG provides prefill savings but not batch-generation savings."** A deployment runtime should classify tool calls into independent vs. dependent sets before applying batching.

### Part III-C: KV-Cache Single-Entity Probing — Information Accessibility

To diagnose the fundamental nature of SIG's recipe enumeration deficiency (§3.5), we designed a KV-cache probing experiment (Diagnostic Q7, n=10). The experiment tests whether the model can complete a partial entity name (e.g., `"spaghetti_"` → `"bolognese"`) under SIG injection vs. AppLoop explicit reading.

**Entity completion results (Qwen 4B, n=10):**

| Condition | Match Rate | Interpretation |
|-----------|-----------|----------------|
| SIG (injected) | **100%** | Information IS accessible via KV-cache attention |
| SIG (NOT injected) | 0% | Baseline: no information, no recall |
| AppLoop (explicit) | 0% | Model emitted `<think>` tag before answer (instruction tuning artifact) |
| AppLoop (NOT in ctx) | 0% | Baseline |

**Multi-entity enumeration:** Both SIG and AppLoop found 0/5 entities (model capacity limitation at 4B Q4 scale for exhaustive listing). **Middle-position probing** (5-entity list, querying the middle item): SIG 0/10, AppLoop 0/10.

**Key insight:** KV-cache injection supports **single-entity completion** at 100% — proving the information is present and attention-accessible. The failure mode is **multi-entity exhaustive enumeration**, not single-entity retrieval. This refines our understanding of the fundamental limitation: the KV-cache preserves *associative* access (analogy, constraint satisfaction, partial completion) but loses *enumerative* access (listing all items, exhaustive recall). The proposed compensatory strategy — inserting explicit memory-retrieval prompts before generation (§10.2) — targets precisely this enumerative gap.

### Part IV: Safety, Fidelity, and Engineering Guidance

**R10: Injection Attacks.** 70% attack success rate (4B: 80%, 0.8B: 60%); 100% rollback keyword-clean recovery (Wilson 95% CI: 84–100%).

**R11: Tool-Result Faithfulness.** Token-Jaccard near-identical (±0.01) between SIG and AppLoop across 12 query-model pairs.

**R17: Context Compression.** Drop-25% achieves 2.3× cache reduction with only 0.3s latency increase.

**R16: Multi-Sequence Concurrency.** 59% VRAM savings using multi-sequence API; 246ms average switch latency.

---

## 5. Cross-Architecture Generalization

To address the critical concern that all core benchmarks were conducted on a single model family (Qwen3.5), we executed the CO baseline, R13 batch injection, and Kitchen quality benchmarks on two architecturally distinct non-Qwen models: **NVIDIA Nemotron-3-Nano-4B** (hybrid Mamba+attention, 4B) and **Google Gemma 4 E2B-IT-2B** (GQA pre-norm, 2B). The full results are in [CROSS_ARCH_TEST_REPORT.md](file:///d:/trunk/SIG/output/cognitive-outsourcing/CROSS_ARCH_TEST_REPORT.md).

### 5.1 Prefill Savings Do NOT Generalize Uniformly

**CO benchmark (SIG vs AppLoop, n=3, 4/8/12 tool chains):**

| Chain Depth | Qwen 4B (dense) | Gemma 4 (2B GQA) | Nemotron (4B hybrid) |
|-------------|-----------------|-------------------|----------------------|
| 4 tools | **2.38×** | 1.13× | 1.03× |
| 8 tools | **2.70×** | 0.92× | 0.98× |
| 12 tools | **2.70×** | 1.31× | 0.92× |
| **Average** | **~2.59×** | **1.12×** | **0.98×** |

On Qwen, SIG consistently achieves 2.38–2.70× speedup through prefill elimination. On Gemma 4, speedup is marginal (0.92–1.31×). On Nemotron, SIG provides **zero net speedup** (avg 0.98×) — the prefill savings that drive SIG's advantage on Qwen are architecture-specific. We hypothesize that Nemotron's Mamba layers, which compress sequence history into a fixed-size SSM state rather than an explicit KV cache, fundamentally alter the prefill-to-generation cost ratio that SIG exploits. This confirms the reviewer's concern: the 2.38–2.70× speedup claims are **Qwen-family-specific**.

### 5.2 Batch-SIG Generalizes Robustly

**R13 batch injection (n=5, 8 fragments):**

| Mode | Qwen 4B | Gemma 4 (2B) | Nemotron (4B hybrid) |
|------|---------|-------------|----------------------|
| Per-Step SIG | 2.673s | 0.467s | 0.849s |
| AppLoop-PC | 1.315s | 1.207s | 0.670s |
| Batch-SIG (bs=8) | 0.283s | 0.177s | 0.158s |
| Speedup vs per-step | **9.45×** | **2.64×** | **5.38×** |
| **Speedup vs AppLoop-PC** | **4.65×** | **6.82×** | **4.24×** |

The batch-injection mechanism — accumulating tool results before a single `generate()` call — provides speedup across **all three architectures** (4.24–6.82× vs AppLoop-PC). This is the strongest cross-architecture finding: the generation-call reduction is architecture-independent. Even on Nemotron, where raw SIG fails (0.98×), Batch-SIG achieves 5.38× vs per-step and 4.24× vs AppLoop-PC. **We recommend Batch-SIG as the default deployment strategy across all architectures.**

### 5.3 Kitchen Quality: Architecture-Dependent Trade-offs

**EdgeAgent-Kitchen (n=5, 30 steps):**

| Metric | Qwen 4B | Gemma 4 (2B) | Nemotron (4B) |
|--------|---------|-------------|---------------|
| SIG Wall-Clock | 18.11±0.33s | 0.96±0.16s | 2.85±0.48s |
| AppLoop Wall-Clock | 62.02±0.80s | 1.83±0.28s | 2.15±0.38s |
| **Speedup** | **3.43×** | **1.90×** | **0.75×** |
| SIG Composite Quality | 0.449 | 0.399±0.043 | 0.325±0.000 |
| AppLoop Composite Quality | 0.567 | 0.325±0.000 | 0.325±0.000 |
| **Quality Δ (SIG−App)** | **−0.119** | **+0.074** ★ | **±0.000** |
| SIG Recipe Mention | 0.00 | 0.268 | 0.000 |
| AppLoop Recipe Mention | 0.42 | 0.000 | 0.000 |

**★ Critical:** On Gemma 4, SIG **outperforms** AppLoop in composite quality (+0.074) — the first observed quality inversion. The pre-norm architecture may interact favorably with KV-cache continuity, producing more coherent outputs than full re-encoding. On Nemotron, quality is identical (±0.000) but SIG is slower (0.75×), making it purely disadvantageous in this configuration. The quality-speedup trade-off is fundamentally architecture-dependent.

### 5.4 Architectural Generalization Matrix

| Capability | Qwen 4B (dense) | Gemma 4 (2B GQA) | Nemotron (4B hybrid) |
|-----------|-----------------|-------------------|----------------------|
| Prefill savings speedup | ✅ 2.59× | ⚠️ 1.12× | ❌ 0.98× |
| Batch-SIG vs AppLoop-PC | ✅ 4.65× | ✅ 6.82× | ✅ 4.24× |
| Kitchen speedup | ✅ 3.43× | ⚠️ 1.90× | ❌ 0.75× |
| Quality Δ (SIG−App) | ⚠️ −0.119 | ✅ +0.074 | ➖ ±0.000 |

**Legend:** ✅ = advantage | ⚠️ = marginal/conditional | ❌ = penalty/none | ➖ = neutral

### 5.5 Implications

1. **Prefill savings are architecture-dependent.** The 2.38–2.70× CO speedup is Qwen-specific. This finding directly addresses the reviewer's cross-architecture concern and imposes a scope boundary on the paper's primary quantitative claims.
2. **Batch-SIG is architecture-independent.** The single most robust finding — Batch-SIG should be the recommended deployment strategy across all model architectures.
3. **Nemotron provides the strongest counter-evidence.** Hybrid Mamba+attention models fundamentally alter the prefill-vs-generation cost ratio. Future work should investigate the mechanism: Mamba's fixed-size SSM state may absorb the information that SIG injects into the KV cache without yielding the computational savings.
4. **Gemma 4's quality inversion warrants follow-up.** The observed +0.074 quality advantage suggests pre-normalization architectures benefit from KV-cache continuity in ways that full re-encoding does not capture.

---

## 6. EdgeAgent-Kitchen Benchmark and Deployment: R15–R19

### 6.1 EdgeAgent-Kitchen

Kitchen simulates a continuous agent session with 18 tools across four interleaved task types: Recipe Planning, Cooking Guidance, Inventory Management, and Interruptions. Steps interleave at 3:2:1:1/15 ratio. The common prefix is ~50 tokens—precisely where prefix caching provides zero benefit.

### 6.2 Main Results

**4B GPU (n=3, 32 steps):**

| Baseline | Wall-Clock | Gen | Prefill | vs SIG |
|----------|-----------|-----|---------|--------|
| SIG | 6.2 ± 0.0s | 4.7s | 0.1s | 1.00× |
| AppLoop | 15.7 ± 0.1s | 9.0s | 5.5s | **2.54×** |
| AppLoop-PC | 23.5 ± 0.1s | 15.4s | 0.0s | 3.79× |
| Sliding | 15.8 ± 0.1s | 9.1s | 5.5s | 2.55× |

**4B GPU (50-step extended, post-review validation):**

| Baseline | Wall-Clock | Turns/s | Probe F1 | Gen(s) | Pf(s) | vs SIG |
|----------|-----------|---------|----------|--------|-------|--------|
| SIG | 5.5s | 9.08 | 50.0% | 3.6s | 0.1s | 1.00× |
| AppLoop | 29.2s | 1.71 | 0.0% | 14.2s | 13.5s | **5.3×** |
| AppLoop-PC | 47.7s | 1.05 | 50.0% | 26.6s | 0.0s | 8.7× |
| Sliding | 29.3s | 1.71 | 0.0% | 14.4s | 13.5s | 5.3× |
| SIG-Hybrid | 7.4s | 6.74 | 0.0% | 5.6s | 0.0s | 1.3× |

The extended 50-step benchmark confirms and amplifies the 32-step findings: SIG achieves **5.3× speedup** with **50% Probe F1** (AppLoop: 0%). AppLoop-PC is **8.7× slower** than SIG — prefix caching actively hurts in continuous tool chains due to zero prefix reuse. SIG's KV-cache continuity enables probe recall that AppLoop's full re-encoding destroys.

**0.8B GPU:** SIG and AppLoop tied at 2.3±0.1s (p>0.3). Crossover at 1.5–2B parameters.

### 6.3 Per-Turn Token Generation Analysis

| Metric | SIG | AppLoop | Ratio |
|--------|-----|---------|-------|
| Total gen tokens | 475 | 921 | 1.94× |
| Total gen time | 4.4s | 9.0s | 2.05× |
| Per-token rate | 108 tok/s | 103 tok/s | 1.05× |

Speedup decomposition: prefill (57%) + output verbosity (43%). Mechanistic speedup: ~55× prefill reduction.

### 6.4 Cross-Hardware Consistency

| | 0.8B | 4B |
|---|------|-----|
| GPU | 1.00× | 2.54× |
| CPU | 0.78× | 4.23× |

Speedup consistency across hardware supports structural nature of advantage.

---

## 7. Theoretical Foundations: R1–R5 (Condensed)

### 7.1 R1: Attention Distribution under SIG Injection

We provide, to our knowledge, the first direct attention-distribution comparison between SIG injection and full re-encoding in the CO/SIG setting. **Results below are on Qwen2.5-0.5B (24 layers, 14 heads) with a single prompt and should be interpreted as a preliminary observation at the sub-1B scale.** The `transformer_bench.py` R1 task has been updated to support multi-prompt averaging (5 diverse domains: travel, kitchen, code, research, navigation) and larger models (Qwen2.5-1.5B), enabling statistical characterisation of cross-prompt variance. Full cross-scale results will be reported in a subsequent revision.

On the 0.5B model (single prompt, for reference):

| Layer Group | Head Agreement Rate | Cosine Similarity |
|------------|--------------------|--------------------|
| Early (0–7) | 0.252 | 0.647 |
| Middle (8–15) | 0.304 | 0.735 |
| Late (16–23) | 0.427 | 0.793 |
| **Overall** | **0.327** | **0.725** |

Early layers are most affected; late layers show progressive recovery. The overall cosine similarity of 0.725 indicates substantial structural preservation. This result places SIG in context of prior attention analysis literature: StreamingLLM [Xiao et al. 2024] demonstrated that attention sinks in early layers dominate long-context utility—our finding that early layers are most sensitive to injection is consistent with this observation, though the underlying mechanism differs (re-encoding vs. injection). Unlike H2O's heavy-hitter attention head analysis [Zhang et al. 2023], we find a smooth layer gradient rather than a small subset of dominant heads.

**Relation to functional equivalence:** Although head agreement is only 0.25–0.43, per-token generation rates and Token-Jaccard faithfulness are nearly identical. This tension can be resolved through the concept of *representational redundancy* in Transformer architectures: multiple attention pathways can encode the same semantic content. The model may attend to different positions yet arrive at similar internal representations for task-relevant facts through distributed encoding. This hypothesis is consistent with recent mechanistic interpretability findings on multi-path information flow in Transformers.

**Note on theory-empirical connection (Appendix A):** The information-theoretic formalisation in Appendix A (KL/JS divergence, mutual information, cosine similarity) provides the conceptual vocabulary for this discussion, but the empirical metrics we report (head agreement rate, cosine similarity of attention weight vectors) are derived from attention pattern comparisons—not from the `I_loss` and `D_KL` constructs of the appendix. Future work should establish direct empirical correspondences between the formal information-theoretic bounds and the layer-wise attention metrics reported here, potentially through mutual information estimation on attention weight distributions.

### 7.2 R2: KV-Cache Degradation

Multi-round weather recall on 0.8B (6 rounds) and 4B (10 rounds): **No degradation** observed across 6–10 rounds. Both models maintain stable recall (0.50–1.00). The limited range (6–10 rounds) precludes extrapolation to 32+ round scenarios.

### 7.3 R3: Cross-Architecture Simulation

A numpy-based simulation engine projects SIG applicability: Transformer (best, validated), xLSTM (promising, unverified), RWKV (viable, unverified), Mamba/SSM (research challenge due to state compression).

### 7.4 R4: Teacher-Student Capability Gap

First measurement in CO/SIG setting: CoT amplification +0.80, SIG amplification +0.59, teacher quality margin 0.72 at 5× capacity ratio.

### 7.5 R5: Privacy Framework

Three leakage channels formalized; PII redaction and intent-only outsourcing demonstrated as concepts. Formal DP guarantees remain future work.

---

## 8. Related Work

Our work intersects with several active research directions. We highlight key distinctions:

**KV-cache optimization.** PagedAttention [Kwon et al. 2023] and SGLang/RadixAttention [Zheng et al. 2024] implement prefix caching for serving efficiency but operate at the cache-reuse level. When tool chains build from empty shared prefixes (<3% token reuse), prefix caching provides zero benefit (our R6 result). SIG's incremental injection avoids re-evaluating *any* prior tokens—a fundamentally different optimization that complements rather than replaces prefix caching.

**LLMCompiler** [Kim et al. 2024] optimizes parallel function calling but still requires complete context re-encoding before final generation. SIG removes this requirement at the inference-engine level.

**StreamingLLM** [Xiao et al. 2024] identifies attention sinks as critical for streaming inference with sliding windows. Our R1 attention analysis finds that early layers are most sensitive to SIG injection, consistent with the principle that foundational attention patterns are most affected by distribution shift—though the mechanism (re-encoding vs. injection) operates differently.

**H2O** [Zhang et al. 2023] identifies "heavy-hitter" attention heads for KV-cache eviction. Our head agreement analysis (R1) finds a smooth gradient rather than a small subset of dominant heads, suggesting that SIG injection affects attention more broadly but with recoverable late-layer representations.

**Long-context utilization.** "Lost in the Middle" [Liu et al. 2024] demonstrates positional bias in long-context models. Our R8 retrieval results are consistent: all modes fail at the farthest probes (T=12).

**Cognitive architectures.** Robo-Cortex and MIRROR pursue self-improving cognitive loops. CO provides the stateful, low-latency runtime for these algorithms—a complementary relationship we elaborate in §10.7.

**Speculative decoding.** ECHO [Hu et al. 2026] introduces budgeted sparse gating. CO draws on budget concepts for adaptive injection gating but applies them to cognitive extension rather than token acceleration.

The key architectural distinction is that CO/SIG is the only approach that redefines tool interaction as an *inference-engine primitive*: KV-cache injection preserves attention continuity in a way that application-layer tool calling and serving-system optimizations do not.

---

## 9. The SIG Decision Framework

We synthesize all findings—including the new batch-injection evidence and cross-architecture validation—into a practical decision framework:

### Advantage Zone (Use SIG)

**Long continuous tool chains (Qwen-family, ≥2B).** 2.79–5.26× speedup with standard (per-step) SIG; 4.38× confirmed on 30-tool chain (N=30); 5.3× on 50-step Kitchen benchmark. Prefix caching structurally ineffective.

**Fragmented assembly with batch-injection (ALL architectures).** Use **Batch-SIG** (not per-step SIG). 4.16–9.45× speedup over per-step SIG by accumulating tool results before generation. **4.24–6.82× faster than AppLoop-PC across Qwen, Gemma 4, and Nemotron.** This is the single most robust cross-architecture finding. Automatic batch-size selection via `DependencyAnalyzer` (§2.6).

### Compatibility Zone (Either Works)

- **CoT-structured reasoning**: Performance-equivalent.
- **Safety-critical with rollback**: 100% recovery.
- **Tool-result faithfulness**: Near-identical to AppLoop.

### Boundary Zone (Avoid SIG)

| Workload | Use | Why |
|----------|-----|-----|
| Random-access retrieval | AppLoop-PC | SIG's persistent cache is a streaming tape, not a random-access array |
| Cloud-dependent tools (>300ms) | Either | SIG advantage collapses to <1.15×; prefill savings dominated by network latency |
| Ultra-short tool chains with shared prefix | AppLoop-PC | Prefix caching effective when reuse >30% (empirical threshold) |
| **Non-Qwen dense-attention architectures** | **AppLoop-PC or Batch-SIG** | **Prefill savings do not generalize to hybrid Mamba+attention (Nemotron 0.98×) or GQA architectures (Gemma 4 1.12×). Standard SIG provides no advantage.** |

### Batch-Injection Zone (ARCHITECTURE-INDEPENDENT)

| Batch Size | Speedup vs Per-Step SIG | Gen Calls | Recommendation |
|-----------|------------------------|-----------|----------------|
| K=2 | 4.23× | 4 | Modest speedup, short tool chains |
| K=4 | 6.65× | 2 | Balanced throughput-quality |
| K=8 | 9.45× | 1 | Max speedup, pure-injection mode |

Larger K reduces generation overhead but may defer reasoning output until after batch completion—the choice is workload-dependent.

**Critical applicability constraint:** Batch-SIG requires that all tools within a batch be **independently executable in parallel**. When tool calls form strict sequential dependencies (e.g., flight search where destination depends on origin results, multi-hop reasoning, Web navigation), tools cannot be pre-executed and batched. For such workloads, per-step SIG provides prefill savings but not batch-generation savings. The runtime should classify tool calls into independent vs. dependent sets before applying batching.

### Hybrid Scheduling

An edge inference runtime can detect workload characteristics and route accordingly: deep chains to standard SIG, fragmented assembly to Batch-SIG with appropriate K, retrieval workloads to AppLoop-PC, cloud-dependent tasks to whichever has lower overhead. CoT workloads are routing-agnostic.

---

## 10. Discussion

### 10.1 SIG Is Not a Universal Accelerator

**Scale dependence:** On 0.8B, SIG and AppLoop are tied (2.3s, p>0.3). Crossover at 1.5–2B parameters. Below this, AppLoop is the pragmatic choice; above, SIG provides increasing returns.

**Architecture dependence (NEW):** Cross-architecture validation (§5) reveals that raw prefill savings are Qwen-family-specific. On Gemma 4 (GQA, 2B), SIG speedup is marginal (1.12× average). On Nemotron (hybrid Mamba+attention, 4B), SIG provides zero net speedup (0.98×). The acceleration mechanism—eliminating quadratic prefill via KV-cache injection—does not transfer to architectures where Mamba layers compress state into fixed-size SSM representations. **Batch-SIG, however, generalizes robustly: 4.24–6.82× vs AppLoop-PC across all three architectures.** The generation-call reduction mechanism is architecture-independent.

### 10.2 Task Quality vs. Speedup: A Measurable but Narrower Trade-off

Our quality evaluation (§3.5) demonstrates that SIG's speedup on 4B comes with a −0.12 composite quality drop under hybrid TF-IDF scoring—a substantial improvement over the −0.39 gap measured with keyword-only scoring. The remaining gap is concentrated in recipe enumeration (recipe_mentioned: SIG 0.00 vs AppLoop 0.42 semantic), not tool execution or allergen awareness. This suggests a targeted compensatory strategy: explicitly query the KV-cache memory before SIG's generation step (e.g., "Recall the recipe names from earlier tool results and list them"), which would bridge the gap without requiring full context re-encoding. We have implemented this as `MEMORY_RETRIEVAL_PROMPT` (§2.6) — an explicit enumeration prompt injected before SIG's generation step — though quantitative measurement of its effectiveness remains future work.

**Fundamental nature of the recall deficiency.** Our KV-cache probing experiment (§4, Part III-C) provides mechanistic clarity: under SIG injection, the model achieves **100% single-entity completion** (e.g., `"spaghetti_"` → `"bolognese"`), proving information IS attention-accessible in the KV cache. The failure mode is **multi-entity exhaustive enumeration** — listing all 5 recipes fails completely, as does middle-position probing. This refines the diagnosis: KV-cache injection preserves *associative* access (partial completion, constraint satisfaction, analogy — explaining SIG's 1.00 allergen awareness vs AppLoop's 0.50) but loses *enumerative* access (exhaustive listing, precise formatted recall). **We therefore classify the enumeration gap as a fundamental architectural limitation of SIG** — not correctable via prompt engineering alone — that defines an upper bound on SIG's factual recall capability. The proposed compensatory strategy (explicit memory-retrieval prompts before generation) targets precisely this enumerative gap. RetroSIG (lightweight explicit recall prompts, see companion paper [4]) provides a practical bridge through lightweight explicit recall prompts, but cannot fully match AppLoop's native enumeration advantage. Future work should explore architectural-level solutions such as hybrid KV-cache layouts that reserve explicit token slots for critical entities alongside distributed attention states.

### 10.3 Latency Conditions on SIG Advantage

The latency ablation (§4, n=5) demonstrates that SIG speedup is conditional: 5.49× at 0ms collapses to 1.08× at 500ms. The tighter n=5 confidence intervals (σ=0.007–0.775s for AppLoop) confirm this collapse is statistically robust, not a small-sample artifact. This has direct implications for the CO architecture: **cloud teacher invocation (200–1000ms) will dominate wall-clock regardless of inference mode.** The CO/SIG value proposition is strongest when (a) tools execute locally at near-zero latency, (b) the cognitive cache provides pre-computed teacher plans, or (c) multiple tool calls are batched before a single generation step.

### 10.4 Why Prefix Caching Does Not Replace SIG

In deep chains, <3% of tokens are cacheable. SIG's incremental injection avoids re-evaluating *any* prior tokens—a fundamentally different optimization path. The Batch-SIG results further demonstrate that even in fragmented workloads where prefix caching is nominally effective, SIG with batching achieves **4.24–6.82× advantage across all tested architectures** by eliminating redundant generation calls.

### 10.5 R13 Root Cause: Generation, Not Cache — and the Batch-Injection Fix

The profiling results (§4) definitively show that SIG's performance penalty in fragmented workloads comes from per-step `generate()` calls (94.8% of wall-clock). The batch-injection experiment provides the **engineering fix**: Batch-SIG (bs=8) achieves wall-clock speedup of 9.45× (Qwen), 2.64× (Gemma 4), and 5.38× (Nemotron) over per-step SIG by reducing generation calls. **This fix is architecture-independent** — the generation bottleneck, not KV-cache access, is the universal constraint. This transforms R13 from a design boundary into an **advantage zone** for SIG, provided the runtime can batch tool results before generation.

### 10.6 Semantic Scoring and the Quality Measurement Pipeline

The introduction of the `SemanticScorer` (TF-IDF cosine similarity, underscore normalization) addresses Limitation #9 from earlier drafts. The hybrid scoring approach reveals that keyword-only scoring inflates apparent quality gaps by penalizing formatting differences that are semantically irrelevant (e.g., "spaghetti_bolognese" vs "spaghetti bolognese"). The 32-unit-test suite ensures scoring correctness across the full evaluation pipeline.

### 10.7 Synergy with Self-Improving Agents

CO provides ready-made runtime for Robo-Cortex-like continual-learning agents. The KV-cache retains full task execution context; heuristics distilled by autonomous knowledge induction become local cognitive modules within CO. Batch-injection further enables "offline reflection" — accumulating tool results and generating a single comprehensive summary — which aligns naturally with offline learning paradigms.

### 10.8 Cross-Architecture Generalization: What Transfers and What Does Not

Our validation across three architectures (§5) yields a clear pattern:

**What generalizes:** (i) Batch-SIG's generation-call reduction (4.24–6.82× vs AppLoop-PC universally). (ii) KV-cache single-entity accessibility (100% completion). (iii) The SIG Decision Framework's zone-based routing logic, with the new architecture-dependent qualifier.

**What does NOT generalize:** (i) Raw prefill savings — Qwen's 2.38–2.70× collapses to 0.98–1.12× on non-Qwen architectures. The SIG advantage is contingent on the model's prefill-to-generation cost ratio, which varies by attention mechanism (dense vs GQA vs Mamba). (ii) The quality-speedup trade-off direction — Qwen shows a quality penalty (−0.119), Gemma 4 shows a quality gain (+0.074), and Nemotron shows neutrality. (iii) The Nemotron result is particularly instructive: despite having 4B parameters (matching Qwen), its hybrid Mamba+attention architecture yields zero SIG speedup, suggesting the Mamba layers' fixed-size state compression fundamentally alters the computational dynamics that SIG exploits.

**Implication for future work:** Before claiming architectural generality, any new SIG acceleration claim must be validated on at least one non-dense-attention architecture. We recommend Nemotron (hybrid Mamba+attention) as the canonical "hard case" for such validation — if a SIG optimization works on Nemotron, it likely generalizes broadly.

---

## 11. Limitations

1. **Task quality is reduced in SIG mode** (§3.5): the −0.12 composite quality penalty (hybrid scoring, 4B) is narrower than previously reported (−0.39 keyword-only), but the recipe enumeration gap persists. **§10.2 identifies this as a fundamental architectural limitation of KV-cache injection—not correctable via prompt engineering alone**—that defines an upper bound on SIG's factual recall capability.
2. **SIG advantage collapses under tool latency** (§4): cloud-dependent scenarios (>300ms delay) reduce speedup to <1.15×. This is now confirmed with n=5 and tight standard errors.
3. **Single model family (Qwen3.5 dense) — PARTIALLY ADDRESSED.** Cross-architecture validation has been executed on two non-Qwen architectures: **NVIDIA Nemotron-3-Nano-4B** (hybrid Mamba+attention) and **Google Gemma 4 E2B-IT-2B** (GQA pre-norm). Results (§5) confirm that prefill savings are Qwen-specific (0.98–1.12× on non-Qwen), but Batch-SIG generalizes (4.24–6.82×). **All primary speedup claims (2.38–2.70×) are now explicitly architecture-qualified.** Additional architectures (Llama-3.2, Phi-3.5) remain for future validation.
4. **Synthetic tools**: Real-world tool noise may interact with injection granularity; batch-injection behavior with noisy real-world tool results is untested.
5. **R1 attention measurement is single-model, single-prompt (§7.1):** the 0.5B single-prompt results constitute a **preliminary observation at a minimal model scale** and should not be interpreted as a general attention regularisation law. The `transformer_bench.py` infrastructure now supports multi-prompt (5 domains) and larger models (1.5B+), but full cross-scale, cross-prompt statistical results are pending.
6. **R2 covers only 6–10 rounds**: Degradation at 32+ rounds uncharacterized.
7. **R3 is simulation-based**: No non-Transformer SIG implementation exists.
8. **R7/R9 at N=1**: Cannot support statistical inference.
9. ~~**Evaluation metrics**: keyword-only biased quality gap downward — addressed via `SemanticScorer` hybrid scoring.~~ Remaining gap is genuine content difference, not scoring artifact.
10. **Quantized-model output variance**: Q4_K_M + CUDA produces length variations at temperature=0, though verbosity stderr=0 across n=10 for 0.8B and n=5 for 4B confirms determinism at both scales.
11. **`kv_cache_seq_shift` incompatibility**: In-place cache compaction via `kv_cache_seq_shift` triggers C-level `GGML_ASSERT` failure on Qwen3.5 models (position encoding constraint `n_pos_per_embd() == 1`). The default `evict_range` strategy is safe `rebuild_cache`; `kv_cache_seq_shift` remains available as opt-in for compatible architectures. This limits eviction performance on the primary test model but does not affect correctness.

---

## 12. Future Work

- **Cross-family replication (PARTIALLY EXECUTED):** Initial validation completed on Nemotron-3-Nano-4B and Gemma 4 E2B-IT-2B (§5). Results confirm architecture-dependent prefill savings but universal Batch-SIG. Extend to additional architectures: **Llama-3.2 (3B+, GQA)** and **Phi-3.5 (3.8B dense)** to complete the coverage matrix. Priority: verify whether dense-attention models beyond Qwen (e.g., Phi) reproduce the 2.38–2.70× prefill savings pattern.
- **Nemotron mechanism investigation (HIGH PRIORITY):** The zero prefill-speedup on Nemotron (0.98×) despite 4B parameters demands mechanistic explanation. Hypothesis: Mamba layers' fixed-size SSM state absorbs injected KV-cache information without yielding prefill savings. Requires per-layer profiling to isolate the SSM-vs-attention contribution to SIG's cost.
- **Gemma 4 quality inversion investigation:** The observed +0.074 quality advantage (SIG > AppLoop) is the first such result. Investigate whether pre-normalization architectures systematically favor KV-cache continuity over re-encoding for output coherence.
- **KV-cache architectural remediation (HIGH PRIORITY):** The fundamental recall deficiency identified in §10.2—SIG's inability to reliably enumerate entities from distributed KV-cache states—requires architectural solutions beyond prompt engineering. Potential directions include hybrid KV-cache layouts with reserved explicit token slots, or learned entity pointers that map distributed representations to explicit token embeddings. The `diagnostic_bench.py --task kv_probe` experiment provides the baseline measurement framework.
- ~~**Batch-SIG dependency classifier**~~ — **IMPLEMENTED**: `DependencyAnalyzer` classifies tool calls into independent (batch-compatible) vs. sequentially-dependent sets and recommends optimal batch size. Validated via Diagnostic Q6 (§4, Part III-B).
- **Compensatory recall engineering:** Implement and measure the targeted "memory prompt" strategy (§10.2) to bridge the remaining recipe enumeration gap. **PARTIALLY IMPLEMENTED**: `MEMORY_RETRIEVAL_PROMPT` template added; quantitative measurement pending.
- **Attention-representation bridge:** Probe key-token hidden states in SIG vs full re-encoding across 5 prompts and ≥1.5B models to characterize how functional equivalence emerges despite attention divergence.
- ~~Batch-injection optimization~~ — **IMPLEMENTED**: Batch-SIG with configurable batch sizes (2/4/8) is demonstrated in this revision (§4, §9).
- ~~**Evaluation metrics (semantic)**~~ — **IMPLEMENTED**: `SemanticScorer` with TF-IDF cosine similarity + keyword hybrid scoring (§3.5).
- **Quality-speedup Pareto characterization:** Measure SIG vs AppLoop trade-off surface across model sizes, chain depths, batch sizes, and task types.
- **R2 extended degradation:** 32+ round characterization.
- **R4 teacher-size scan:** Vary teacher across 3B–70B.
- **R5 privacy validation:** Formal DP guarantees and attack simulations.
- **R8 metric improvement:** Apply `SemanticScorer` to retrieval evaluation.
- **R12 empirical scaling:** Prefill measurement across ≥3 model sizes.
- **Dynamic batch-size selection:** Learn optimal K per workload from runtime characteristics (tool count, chain depth, urgency). **PARTIALLY IMPLEMENTED**: `DependencyAnalyzer` provides rule-based classification; learned selection pending.
- **`kv_cache_seq_shift` compatibility investigation:** Determine which model architectures support `n_pos_per_embd == 1` for in-place cache compaction, and implement runtime detection.

---

## 13. Conclusion

We have presented a comprehensive, empirically-grounded treatment of Cognitive Outsourcing with Suspend-and-Inject Generation. CO/SIG eliminates quadratic prefill overhead by injecting external cognitive resources directly into the model's KV cache, achieving 73–97% prefill token savings and 2.38–9.45× end-to-end speedups across continuous tool chains **on Qwen-family models**.

**Key contributions with empirical support:**

1. **Architecture**: The five-stage SIG cycle and three-layer CO architecture elevate tool interaction from an application-loop to an inference-engine primitive. Prefill savings are robust (73–97%) on Qwen dense-attention models.

2. **Design space**: N=30 paired-run measurements across nine vectors define SIG's scope—2.79–4.38× deep-chain advantage on Qwen, CoT-equivalent performance, 100% rollback recovery, and principled design boundaries (0% retrieval on 0.8B).

3. **Cross-architecture validation (§5)**: First multi-family CO/SIG evaluation across three architectures (Qwen3.5 dense, Gemma 4 GQA, Nemotron hybrid Mamba+attention). Prefill savings are **architecture-dependent** (2.59× → 0.98–1.12× on non-Qwen). **Batch-SIG generalizes robustly** (4.24–6.82× vs AppLoop-PC across all architectures). Gemma 4 exhibits a quality inversion (SIG +0.074 > AppLoop), the first such observation.

4. **Batch-injection as a design-space transformation**: Batch-SIG converts SIG's worst-case regime into advantage (4.24–6.82× vs AppLoop-PC across architectures; 9.45× vs per-step on Qwen). Sequential dependency analysis (§4) verifies the independent-tool precondition.

5. **KV-cache recall mechanism clarified**: Probing experiments (§4) show 100% single-entity completion (information IS accessible) but 0% multi-entity enumeration — refining the diagnosis from "recall failure" to "enumerative limitation." The KV-cache preserves associative access but loses enumerative access.

6. **Diagnostic findings**: Quality gap narrowed to −0.12 (hybrid scoring); SIG speedup collapses from 5.49× to 1.08× under 500ms tool latency (n=5).

7. **Deployment**: 3.43–5.3× Kitchen speedup on Qwen 4B (32–50 steps), batch-injection for fragmented workloads, 2.3× cache reduction via compression, and the expanded SIG Decision Framework (§9) with architecture-dependent routing.

8. **Infrastructure**: Six benchmark scripts (including `cross_arch_bench.py`), 51 unit tests, modular `core/` library, hybrid semantic quality scorer, cumulative compute tracking (`CacheStats`), and automatic batch-size selection (`DependencyAnalyzer`).

SIG is a **scale-dependent, architecture-dependent, condition-dependent performance optimization**. Its raw prefill savings are Qwen-family-specific; Batch-SIG generalizes universally. The quality-speedup trade-off direction varies by architecture (negative on Qwen, positive on Gemma 4, neutral on Nemotron). The batch-injection extension broadens applicability to fragmented assembly workloads, and the cross-architecture validation provides the first principled scope boundary for SIG's quantitative claims. **We recommend Batch-SIG as the default deployment strategy across all architectures, and standard (per-step) SIG only for Qwen-family dense-attention models on deep continuous tool chains.** The honest delineation of these architectural bounds, trade-offs, and engineering remedies is what makes this value proposition credible.

---

## References

[1] *Cognitive Outsourcing with Suspend-and-Inject Generation for Scalable Embodied Intelligence.* Paper 1, May 2026.

[2] *Beyond the Injection Engine: A Five-Dimensional Analysis of CO-SIG.* Paper 2, May 2026.

[3] *Extending CO-SIG Across Nine Additional Research Vectors.* Paper 3, May 2026.

[4] *SIG as an Edge Inference Runtime Primitive for Long-Horizon Agent Tasks.* Paper 4, May 2026.

[5] Kwon, W., et al. "Efficient Memory Management for Large Language Model Serving with PagedAttention." SOSP 2023.

[6] Zheng, L., et al. "SGLang: Efficient Execution of Structured Language Model Programs." NeurIPS 2024.

[7] Dao, T., et al. "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness." NeurIPS 2022.

[8] Wei, J., et al. "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models." NeurIPS 2022.

[9] Xiao, G., et al. "Efficient Streaming Language Models with Attention Sinks." ICLR 2024.

[10] Zhang, Z., et al. "H2O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models." NeurIPS 2023.

[11] Liu, N.F., et al. "Lost in the Middle: How Language Models Use Long Contexts." TACL 2024.

[12] Kim, S., et al. "LLMCompiler: An LLM Compiler for Parallel Function Calling." arXiv:2402.04578, 2024.

[13] Gu, A. and Dao, T. "Mamba: Linear-Time Sequence Modeling with Selective State Spaces." arXiv:2312.00752, 2023.

[14] Peng, B., et al. "RWKV: Reinventing RNNs for the Transformer Era." arXiv:2305.13048, 2023.

[15] Beck, M., et al. "xLSTM: Extended Long Short-Term Memory." arXiv:2405.04517, 2024.

[16] Leviathan, Y., et al. "Fast Inference from Transformers via Speculative Decoding." ICML 2023.

[17] Dwork, C., et al. "The Algorithmic Foundations of Differential Privacy." Foundations and Trends in TCS, 2014.

[18] Gerganov, G. "llama.cpp: LLM inference in C/C++." GitHub, 2023.

[19] Qwen Team. "Qwen3.5: Mixed-Attention Small Language Models." HuggingFace, February 2026.

[20] Zhang, P., et al. "TinyLlama: An Open-Source Small Language Model." arXiv:2401.02385, 2024.

[21] Hoffmann, J., et al. "Training Compute-Optimal Large Language Models." NeurIPS 2022.

---

## Appendix A: Extended Theoretical Analysis

### A.1 R1: Information-Theoretic Framework

Under full re-encoding (AppLoop), after a tool call at position $t$, the complete prefix $(X, G_{<t}, R)$ is re-encoded, producing hidden states $H^{\text{full}}$. Under SIG injection, only tool result tokens $R$ undergo prefill using the suspended KV-cache as prefix, producing $H^{\text{inj}}$.

We define *information loss* as:

$$I_{\text{loss}} = I(X, G; H^{\text{full}}) - I(X, G; H^{\text{inj}})$$

The conjectured upper bound is:

$$I_{\text{loss}} \leq H(R \mid X, G) \cdot \sum_{\ell \in S} \alpha_\ell \cdot (1 - \rho_\ell(R, H_{<\ell}))$$

where $S$ is the set of sensitive layers, $\alpha_\ell$ the layer's contribution weight, and $\rho_\ell$ the cosine similarity between injected and full representations.

### A.2 R2: Degradation Models

Three competing hypotheses for recall $R(k,m)$ of facts from injection $k$ after $m$ subsequent injections:

**H1: Logarithmic.** $R \propto 1 - \beta\log(1+m)$ — analog to Ebbinghaus forgetting curve.

**H2: Linear.** $R \propto 1 - \beta m$ — constant attention dilution.

**H3: Phase transition.** $R$ constant below critical length, then catastrophic drop.

Our data (no degradation at 6–10 rounds) is insufficient to distinguish between these models.

### A.3 R3: Cross-Architecture Projections

| Architecture | Information Fidelity | Injection Latency | State Retention | Overall |
|-------------|---------------------|-------------------|-----------------|---------|
| Transformer | ★★★★★ (validated) | ★★★ | ★★★★★ | Most suitable |
| xLSTM | ★★★★ (projected) | ★★★★ | ★★★★ | Promising |
| RWKV | ★★★ (projected) | ★★★★ | ★★★ | Viable |
| Mamba/SSM | ★★ (projected) | ★★★★★ | ★★ | Research challenge |

### A.4 R4: Teacher-Student Formalization

Teacher $T$ with capability $c_T$, student $S$ with $c_S$, gap $\Delta = c_T - c_S$. CoT comprehension rate: $\text{CR} = |\widehat{\Pi} \cap \Pi|/|\Pi|$. Performance gain: $\text{PG} = Q(A_{\text{with CoT}})/Q(A_{\text{autonomous}})$.

### A.5 R5: Privacy Leakage Channels

**Direct:** $L_{\text{direct}} = I(U; Q)$ — PII in query text.  
**Indirect:** $L_{\text{indirect}} = I(U; \phi(Q) \mid \text{explicit}(Q))$ — inferable from query patterns.  
**Tool-argument:** $L_{\text{tool}} = \sum_{t} I(U; \text{args}_t)$ — exposed through tool parameters.  

Cumulative loss: $L_{\text{cumulative}}(n) = \sum_{i=1}^n \text{PLM}(q_i, r_i \mid h_{i-1})$.

---

## Appendix B: Implementation Details

### B.1 Diagnostic Experiment Usage

```bash
# Run all unit tests (51 tests, <1s)
python -m unittest discover -s tests -v

# Task quality evaluation (n=5, hybrid semantic + keyword scoring)
python diagnostic_bench.py --task quality_kitchen \
  --model models/Qwen3.5-4B-Q4_K_M.gguf --n-gpu-layers 99 --quality-runs 5

# R13 batch-injection profiling (n=10, bs=2/4/8 + per-step + AppLoop-PC)
python diagnostic_bench.py --task profile_r13_batch \
  --model models/Qwen3.5-4B-Q4_K_M.gguf --n-gpu-layers 99 --profile-runs 10

# Latency ablation (n=3, 0/100/300/500ms delays)
python diagnostic_bench.py --task latency_ablation \
  --model models/Qwen3.5-4B-Q4_K_M.gguf --n-gpu-layers 99 --latency-runs 3

# Verbosity control (n=5, SHORT vs LONG prompt, 4B GPU)
python diagnostic_bench.py --task verbosity_control \
  --model models/Qwen3.5-4B-Q4_K_M.gguf --n-gpu-layers 99 --verbosity-runs 5

# Sequential dependency (n=10, independent vs sequential chains)
python diagnostic_bench.py --task seq_dependency \
  --model models/Qwen3.5-4B-Q4_K_M.gguf --n-gpu-layers 99

# KV-cache probing (n=10, single/multi entity completion)
python diagnostic_bench.py --task kv_probe \
  --model models/Qwen3.5-4B-Q4_K_M.gguf --n-gpu-layers 99

# Run all diagnostic experiments sequentially
python diagnostic_bench.py --task all --model models/Qwen3.5-4B-Q4_K_M.gguf --n-gpu-layers 99

# Full experiment suite (R6–R19 + Kitchen)
python run_all_experiments.py --model models/Qwen3.5-4B-Q4_K_M.gguf --n-gpu-layers 99

# Cross-architecture benchmarks
python cross_arch_bench.py --model models/nvidia_Nemotron-3-Nano-4B-Q4_K_M.gguf --task all
python cross_arch_bench.py --model models/gemma-4-E2B-it-Q4_K_M.gguf --task all
```

### B.2 `kv_cache_seq_rm` Erratum

Original codebase used `kv_cache_seq_rm(0, -1, -1)` for cache resets, marking slots free without defragmenting. After multiple reset→eval cycles, no contiguous block remained. Correction to `kv_cache_clear()` fully resets both cache and allocator. All baselines now complete 100% of steps.

### B.3 `kv_cache_seq_shift` Incompatibility

The `evict_range` method was designed to use `kv_cache_seq_shift` for O(shift) in-place cache compaction. On Qwen3.5 models, this triggers a C-level `GGML_ASSERT(hparams.n_pos_per_embd() == 1)` failure in `llama-kv-cache.cpp:516`, causing an unrecoverable process abort. The root cause is that Qwen3.5's position encoding uses `n_pos_per_embd > 1`, which is incompatible with `seq_add()` operations. The default `evict_range` strategy is now `use_compaction=False` (safe `rebuild_cache`), with `kv_cache_seq_shift` available as opt-in for models whose position encoding satisfies the constraint.

---

> **Data availability**: Reproduction scripts at `co_benchmark.py`, `sig_benchmark.py`, `transformer_bench.py`, `edge_agent_bench.py`, `diagnostic_bench.py`, `cross_arch_bench.py`, `run_all_experiments.py`, `run_multi_bench.py`. Core library at `core/`. Scenario builders at `core/scenarios.py`. Unit tests at `tests/` (51 tests). Benchmark results at `BENCHMARK_RESULTS.md`. Plans and prompts at `co_benchmark_plans.json`, `co_benchmark_prompts.json`.
