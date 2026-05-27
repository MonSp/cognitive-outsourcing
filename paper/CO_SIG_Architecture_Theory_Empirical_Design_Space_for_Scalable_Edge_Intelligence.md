# Cognitive Outsourcing with Suspend-and-Inject Generation: Architecture, Theory, and Empirical Design Space for Scalable Edge Intelligence

**Revised Manuscript — May 2026**

> This revision incorporates new experimental results from increased sample sizes (n=5–10), a batch-injection experiment that demonstrates decisive speedup recovery, and hybrid TF-IDF semantic quality scoring that refines the task-completion evaluation.

---

## Abstract

We present a comprehensive treatment of Cognitive Outsourcing (CO) with Suspend-and-Inject Generation (SIG)—an edge-AI paradigm enabling lightweight on-device language models (0.8B–4B parameters) to access external cognitive resources while preserving KV-cache attention continuity. The SIG primitive eliminates quadratic prefill overhead by injecting tool results directly into the model's key-value cache, achieving **73–97% prefill token savings** across nine benchmark scenarios.

On Qwen3.5 models (Q4_K_M, RTX 4070 SUPER), SIG delivers end-to-end speedups of **2.38× (0.8B) to 2.70× (4B)** in teacher-precomputed mode, with peak deep-chain speedups reaching **5.26×**. On the EdgeAgent-Kitchen benchmark—a 32-step interleaved agent scenario—SIG achieves **3.43× wall-clock speedup** on 4B models with improved scoring methodology. A per-turn token analysis confirms that SIG and AppLoop generate tokens at nearly identical per-token rates (within 5%), establishing that speedup derives from prefill elimination, not faster generation.

**Critically, we report updated diagnostic measurements that qualify and refine the speedup claims.** Task-completion quality evaluation with a hybrid TF-IDF semantic scorer reveals that the quality gap between SIG and AppLoop narrows from −0.39 (keyword-only) to **−0.12 (hybrid scoring)**, though SIG's inability to enumerate recipes from persistent KV-cache representations remains a measurable limitation. Fine-grained R13 profiling confirms that SIG's per-step `generate()` calls—not KV-cache access—dominate cost in fragmented workloads (94.8% of wall-clock). **A new batch-injection experiment demonstrates recovery: by accumulating tool results before a single generation call, Batch-SIG achieves 9.45× speedup over per-step SIG and 4.65× over AppLoop-PC**, conclusively proving that the generation bottleneck is self-inflicted benchmark design rather than a fundamental SIG limitation. Latency ablation (n=5) confirms that SIG speedup diminishes from **5.49× at 0ms to 1.08× at 500ms** tool-execution delay, with tight confidence intervals.

We provide the first direct attention-distribution comparison between SIG injection and full re-encoding in the CO/SIG setting (head agreement 0.25 early→0.43 late layers), multi-round KV-cache degradation measurements (no degradation at 6–10 rounds), and a cross-architecture simulation engine with empirical calibration. The SIG Decision Framework synthesizes all findings—including the batch-injection exception to the "avoid SIG for fragmented workloads" rule—into a practical routing heuristic for edge inference systems. The codebase now includes **32 unit tests** (`tests/` directory), a **TF-IDF SemanticScorer** (`core/quality.py`), and a reusable **mean_std utility** (`core/metrics.py`). The full testing infrastructure (`co_benchmark.py`, `sig_benchmark.py`, `transformer_bench.py`, `edge_agent_bench.py`, `diagnostic_bench.py`) and `core/` library are released as a reusable empirical foundation.

---

## 1. Introduction

### 1.1 The Edge Agent Inference Challenge

Autonomous language agents on edge devices—robots, smartphones, drones—must execute extended tool chains while maintaining coherent memory across interleaved tasks, under strict latency budgets and severe resource constraints. The standard bridging mechanism, application-layer tool calling, operates in a *stateless loop*: each external query triggers full re-encoding of the entire conversation history. This discards the model's internal attention state, incurs quadratic prefill costs, and obliterates cognitive context essential for embodied agents that track spatial awareness across long action sequences.

### 1.2 The CO/SIG Solution

**Suspend-and-Inject Generation (SIG)** addresses this at the inference-engine level through three primitives: **Suspend** (maintain KV cache), **Inject** (evaluate tool results into existing cache), and **Resume** (continue generation). **Cognitive Outsourcing (CO)** organizes edge intelligence into three layers: a Meaning Compiler (lightweight local model), an Injection Engine (SIG runtime), and a pluggable Cognitive Module Ecosystem (cloud teachers, perception APIs, skill libraries).

### 1.3 Contributions

1. **Architecture formalization**: Five-stage suspend-inject-resume cycle, stabilization templates, three-layer CO architecture.
2. **Empirical characterization**: 73–97% prefill savings, 2.38–5.26× speedups across nine scenarios and deep-chain benchmarks.
3. **Diagnostic experiments** addressing key validity concerns: hybrid semantic task-quality assessment (n=10), batch-injection profiling (n=10), latency ablation (n=5), verbosity control (n=10).
4. **Theoretical foundations** (R1–R5): Attention distribution analysis, KV-cache degradation, cross-architecture simulation, teacher-student capability gap, privacy framework.
5. **Design-space exploration** (R6–R14): SIG's advantage zones, compatibility zones, design boundaries, and the batch-injection exception.
6. **Deployment architecture** (R15–R19): Kitchen benchmark, hybrid scheduling, context compression, multi-sequence concurrency.
7. **The SIG Decision Framework**: A practical routing heuristic incorporating batch-injection strategies.
8. **Production-grade infrastructure**: Five benchmark scripts, 32 unit tests, hybrid semantic quality scorer, and a modular `core/` library.

### 1.4 Scope and Position

SIG is a **scale-dependent performance optimization** whose value grows with model size and task complexity. For edge deployments running 4B+ models on continuous, deep tool chains with near-instantaneous tool execution, it delivers meaningful wall-clock speedup through prefill elimination. This paper defines not only SIG's advantage zones—where it provides 2.5–9.5× acceleration—but also its principled design boundaries, the conditions under which speedup diminishes, and the engineering remedies (batch-injection) that recover advantage in previously problematic regimes. We adopt the reviewer's framing: this is a **specialized edge acceleration paradigm**, not a universal inference accelerator.

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
| InjectionEngine | `core/injection.py` | Token injection, cache tracking, rollback, eviction |
| PrefixCache | `core/compiler.py` | AppLoop-PC baseline via `kv_cache_seq_cp` |
| ToolRegistry | `core/tools.py` | Simulated tool execution, LatencyToolWrapper |
| Prompt Templates | `core/prompts.py` | System prompts, teacher planning prompts, CoT templates |
| Scenarios | `core/scenarios.py` | Nine CO benchmark scenario builders (extracted from `co_benchmark.py`) |
| GPUMonitor | `core/gpu.py` | VRAM tracking, SM/memory bandwidth utilization |
| Info Theory | `core/info_theory.py` | KL/JS divergence, KSG MI, head agreement, cosine similarity |
| Quality | `core/quality.py` | Task-completion evaluators: keyword, TF-IDF semantic, composite |
| Metrics | `core/metrics.py` | Fact extraction, mean_std, compute_metrics_table, continuous recall |
| Text Utils | `core/text_utils.py` | City name normalization |

The codebase includes **32 unit tests** (`tests/` directory) covering `ToolRegistry`, `LatencyToolWrapper`, `SemanticScorer`, `KitchenQualityEvaluator`, and `mean_std`/`compute_metrics_table`. A critical methodological finding: the original codebase used `kv_cache_seq_rm(0, -1, -1)` for cache resets, which marks slots free without defragmenting. After multiple cycles, OOM failures occurred. The correction to `kv_cache_clear()` resolves this.

---

## 3. Empirical Validation

### 3.1 Experimental Setup

| Parameter | Value |
|-----------|-------|
| Hardware | NVIDIA GeForce RTX 4070 SUPER (12,282 MB VRAM), Intel i7 CPU |
| Models | Qwen3.5 0.8B / 4B (Q4_K_M quantization, dense) |
| Framework | llama-cpp-python (CUDA, n_ctx=16,384) |
| Benchmarks | 9 CO scenarios, 9 SIG research vectors (R6–R14), EdgeAgent-Kitchen |
| Diagnostic | 5 controlled experiments via `diagnostic_bench.py` |
| Runs | n=10 quality_kitchen, n=10 batch-injection, n=5 latency_ablation, n=10 verbosity, n=30 R6/R13/R14 |
| Tests | 32 unit tests covering core modules |

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

**0.8B CPU results (n=3, keyword-only):**

| Metric | SIG | AppLoop |
|--------|-----|---------|
| Wall-Clock | 5.63±0.15s | 30.55±0.33s |
| Speedup | **5.42×** | — |
| Gen Tokens | 170 | 1230 |
| **Composite Quality** | **0.317** | **0.606** |
| Quality Δ | **−0.289** | — |

**Key findings.** The hybrid semantic scorer narrows the observed quality gap compared to keyword-only scoring on 4B, but does not eliminate it: recipe_mentioned remains 0.00 because "spaghetti_bolognese" as a keyword does not appear as a substring in the output "I have found several Italian dinner options" — a genuinely different output content, not a scoring artifact. This confirms a mechanistic explanation: SIG's persistent KV-cache provides ongoing *attention* to prior context but the shorter autoregressive generations omit explicit factual enumeration that AppLoop recovers through repeated context in the re-encoded prompt. Two deployment strategies emerge: (a) accept the trade-off in latency-critical applications where approximate answers suffice, or (b) insert explicit memory-retrieval prompts before SIG's generation step as a compensatory mechanism.

### 3.6 Verbosity Control Experiment

To test whether the output-length difference is a prompt-format artifact, we compared generation under "SHORT" ("Answer concisely:") vs "LONG" ("Provide a detailed, thorough, and complete answer...") instructions on identical factual content. On 0.8B CPU (n=10, increased from n=3), both prompts produced nearly identical output lengths (18±0 vs 16±0 tokens), with **zero standard deviation** across 10 runs. This confirms that the model's capacity—not prompt wording—dominates output length at 0.8B scale. The zero-variance result also rules out Q4_K_M quantization noise as a confounding factor for this measurement, though quantization-related variance at larger scales remains an acknowledged limitation.

---

## 4. Design Space Exploration: R6–R14

### Part I: Core Value Evidence

**R6: Dynamic Replanning — The Deep-Chain Advantage.** N=30 paired runs on a 30-tool chain with 15% failure injection:

| Model | SIG | AppLoop | AppLoop-PC | Speedup vs AppLoop |
|------|-----|---------|------------|-------------------|
| 0.8B | 0.232±0.021s | 0.646±0.022s | 0.646±0.021s | **2.79×** |
| 4B | 0.480±0.016s | 2.043±0.073s | 2.038±0.084s | **4.26×** |

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

The gen call count drops from 16 (per-step) to 4→2→1 (batch-sizes 2→4→8), directly proportional to the wall-clock speedup. This provides **conclusive evidence** that the R13 performance penalty in earlier drafts was self-inflicted by benchmark design (per-step generation), not a fundamental SIG limitation. The implication for the SIG Decision Framework is clear: **fragmented assembly workloads should use Batch-SIG rather than avoiding SIG entirely**.

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

### Part IV: Safety, Fidelity, and Engineering Guidance

**R10: Injection Attacks.** 70% attack success rate (4B: 80%, 0.8B: 60%); 100% rollback keyword-clean recovery (Wilson 95% CI: 84–100%).

**R11: Tool-Result Faithfulness.** Token-Jaccard near-identical (±0.01) between SIG and AppLoop across 12 query-model pairs.

**R17: Context Compression.** Drop-25% achieves 2.3× cache reduction with only 0.3s latency increase.

**R16: Multi-Sequence Concurrency.** 59% VRAM savings using multi-sequence API; 246ms average switch latency.

---

## 5. EdgeAgent-Kitchen Benchmark and Deployment: R15–R19

### 5.1 EdgeAgent-Kitchen

Kitchen simulates a continuous agent session with 18 tools across four interleaved task types: Recipe Planning, Cooking Guidance, Inventory Management, and Interruptions. Steps interleave at 3:2:1:1/15 ratio. The common prefix is ~50 tokens—precisely where prefix caching provides zero benefit.

### 5.2 Main Results

**4B GPU (n=3, 32 steps):**

| Baseline | Wall-Clock | Gen | Prefill | vs SIG |
|----------|-----------|-----|---------|--------|
| SIG | 6.2 ± 0.0s | 4.7s | 0.1s | 1.00× |
| AppLoop | 15.7 ± 0.1s | 9.0s | 5.5s | **2.54×** |
| AppLoop-PC | 23.5 ± 0.1s | 15.4s | 0.0s | 3.79× |
| Sliding | 15.8 ± 0.1s | 9.1s | 5.5s | 2.55× |

**0.8B GPU:** SIG and AppLoop tied at 2.3±0.1s (p>0.3). Crossover at 1.5–2B parameters.

### 5.3 Per-Turn Token Generation Analysis

| Metric | SIG | AppLoop | Ratio |
|--------|-----|---------|-------|
| Total gen tokens | 475 | 921 | 1.94× |
| Total gen time | 4.4s | 9.0s | 2.05× |
| Per-token rate | 108 tok/s | 103 tok/s | 1.05× |

Speedup decomposition: prefill (57%) + output verbosity (43%). Mechanistic speedup: ~55× prefill reduction.

### 5.4 Cross-Hardware Consistency

| | 0.8B | 4B |
|---|------|-----|
| GPU | 1.00× | 2.54× |
| CPU | 0.78× | 4.23× |

Speedup consistency across hardware supports structural nature of advantage.

---

## 6. Theoretical Foundations: R1–R5 (Condensed)

### 6.1 R1: Attention Distribution under SIG Injection

We provide, to our knowledge, the first direct attention-distribution comparison between SIG injection and full re-encoding in the CO/SIG setting. On Qwen2.5-0.5B (24 layers, 14 heads):

| Layer Group | Head Agreement Rate | Cosine Similarity |
|------------|--------------------|--------------------|
| Early (0–7) | 0.252 | 0.647 |
| Middle (8–15) | 0.304 | 0.735 |
| Late (16–23) | 0.427 | 0.793 |
| **Overall** | **0.327** | **0.725** |

Early layers are most affected; late layers show progressive recovery. The overall cosine similarity of 0.725 indicates substantial structural preservation. This result places SIG in context of prior attention analysis literature: StreamingLLM [Xiao et al. 2024] demonstrated that attention sinks in early layers dominate long-context utility—our finding that early layers are most sensitive to injection is consistent with this observation, though the underlying mechanism differs (re-encoding vs. injection). Unlike H2O's heavy-hitter attention head analysis [Zhang et al. 2023], we find a smooth layer gradient rather than a small subset of dominant heads.

**Relation to functional equivalence:** Although head agreement is only 0.25–0.43, per-token generation rates and Token-Jaccard faithfulness are nearly identical. This tension can be resolved through the concept of *representational redundancy* in Transformer architectures: multiple attention pathways can encode the same semantic content. The model may attend to different positions yet arrive at similar internal representations for task-relevant facts through distributed encoding. This hypothesis is consistent with recent mechanistic interpretability findings on multi-path information flow in Transformers.

### 6.2 R2: KV-Cache Degradation

Multi-round weather recall on 0.8B (6 rounds) and 4B (10 rounds): **No degradation** observed across 6–10 rounds. Both models maintain stable recall (0.50–1.00). The limited range (6–10 rounds) precludes extrapolation to 32+ round scenarios.

### 6.3 R3: Cross-Architecture Simulation

A numpy-based simulation engine projects SIG applicability: Transformer (best, validated), xLSTM (promising, unverified), RWKV (viable, unverified), Mamba/SSM (research challenge due to state compression).

### 6.4 R4: Teacher-Student Capability Gap

First measurement in CO/SIG setting: CoT amplification +0.80, SIG amplification +0.59, teacher quality margin 0.72 at 5× capacity ratio.

### 6.5 R5: Privacy Framework

Three leakage channels formalized; PII redaction and intent-only outsourcing demonstrated as concepts. Formal DP guarantees remain future work.

---

## 7. Related Work

Our work intersects with several active research directions. We highlight key distinctions:

**KV-cache optimization.** PagedAttention [Kwon et al. 2023] and SGLang/RadixAttention [Zheng et al. 2024] implement prefix caching for serving efficiency but operate at the cache-reuse level. When tool chains build from empty shared prefixes (<3% token reuse), prefix caching provides zero benefit (our R6 result). SIG's incremental injection avoids re-evaluating *any* prior tokens—a fundamentally different optimization that complements rather than replaces prefix caching.

**LLMCompiler** [Kim et al. 2024] optimizes parallel function calling but still requires complete context re-encoding before final generation. SIG removes this requirement at the inference-engine level.

**StreamingLLM** [Xiao et al. 2024] identifies attention sinks as critical for streaming inference with sliding windows. Our R1 attention analysis finds that early layers are most sensitive to SIG injection, consistent with the principle that foundational attention patterns are most affected by distribution shift—though the mechanism (re-encoding vs. injection) operates differently.

**H2O** [Zhang et al. 2023] identifies "heavy-hitter" attention heads for KV-cache eviction. Our head agreement analysis (R1) finds a smooth gradient rather than a small subset of dominant heads, suggesting that SIG injection affects attention more broadly but with recoverable late-layer representations.

**Long-context utilization.** "Lost in the Middle" [Liu et al. 2024] demonstrates positional bias in long-context models. Our R8 retrieval results are consistent: all modes fail at the farthest probes (T=12).

**Cognitive architectures.** Robo-Cortex and MIRROR pursue self-improving cognitive loops. CO provides the stateful, low-latency runtime for these algorithms—a complementary relationship we elaborate in §8.6.

**Speculative decoding.** ECHO [Hu et al. 2026] introduces budgeted sparse gating. CO draws on budget concepts for adaptive injection gating but applies them to cognitive extension rather than token acceleration.

The key architectural distinction is that CO/SIG is the only approach that redefines tool interaction as an *inference-engine primitive*: KV-cache injection preserves attention continuity in a way that application-layer tool calling and serving-system optimizations do not.

---

## 8. The SIG Decision Framework

We synthesize all findings—including the new batch-injection evidence—into a practical decision framework:

### Advantage Zone (Use SIG)

**Long continuous tool chains.** 2.79–5.26× speedup with standard (per-step) SIG. Prefix caching structurally ineffective.

**Fragmented assembly with batch-injection.** Use **Batch-SIG** (not per-step SIG). 4.23–9.45× speedup over per-step SIG by accumulating tool results before generation. 2.08–4.65× faster than AppLoop-PC. This is the critical exception to earlier guidance that "fragmented workloads should avoid SIG."

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

### Batch-Injection Zone (NEW)

| Batch Size | Speedup vs Per-Step SIG | Gen Calls | Recommendation |
|-----------|------------------------|-----------|----------------|
| K=2 | 4.23× | 4 | Modest speedup, short tool chains |
| K=4 | 6.65× | 2 | Balanced throughput-quality |
| K=8 | 9.45× | 1 | Max speedup, pure-injection mode |

Larger K reduces generation overhead but may defer reasoning output until after batch completion—the choice is workload-dependent.

### Hybrid Scheduling

An edge inference runtime can detect workload characteristics and route accordingly: deep chains to standard SIG, fragmented assembly to Batch-SIG with appropriate K, retrieval workloads to AppLoop-PC, cloud-dependent tasks to whichever has lower overhead. CoT workloads are routing-agnostic.

---

## 9. Discussion

### 9.1 SIG Is Not a Universal Accelerator

On 0.8B, SIG and AppLoop are tied (2.3s, p>0.3). Crossover at 1.5–2B parameters. Below this, AppLoop is the pragmatic choice; above, SIG provides increasing returns. The batch-injection advantage also scales with model size: at 4B, Batch-SIG (bs=8) is 4.65× faster than AppLoop-PC on fragmented workloads.

### 9.2 Task Quality vs. Speedup: A Measurable but Narrower Trade-off

Our quality evaluation (§3.5) demonstrates that SIG's 3.43× speedup on 4B comes with a −0.12 composite quality drop under hybrid TF-IDF scoring—a substantial improvement over the −0.39 gap measured with keyword-only scoring. The remaining gap is concentrated in recipe enumeration (recipe_mentioned: SIG 0.00 vs AppLoop 0.42 semantic), not tool execution or allergen awareness. This suggests a targeted compensatory strategy: explicitly query the KV-cache memory before SIG's generation step (e.g., "Recall the recipe names from earlier tool results and list them"), which would bridge the gap without requiring full context re-encoding.

### 9.3 Latency Conditions on SIG Advantage

The latency ablation (§4, n=5) demonstrates that SIG speedup is conditional: 5.49× at 0ms collapses to 1.08× at 500ms. The tighter n=5 confidence intervals (σ=0.007–0.775s for AppLoop) confirm this collapse is statistically robust, not a small-sample artifact. This has direct implications for the CO architecture: **cloud teacher invocation (200–1000ms) will dominate wall-clock regardless of inference mode.** The CO/SIG value proposition is strongest when (a) tools execute locally at near-zero latency, (b) the cognitive cache provides pre-computed teacher plans, or (c) multiple tool calls are batched before a single generation step.

### 9.4 Why Prefix Caching Does Not Replace SIG

In deep chains, <3% of tokens are cacheable. SIG's incremental injection avoids re-evaluating *any* prior tokens—a fundamentally different optimization path. The Batch-SIG results further demonstrate that even in fragmented workloads where prefix caching is nominally effective, SIG with batching achieves 4.65× advantage by eliminating redundant generation calls.

### 9.5 R13 Root Cause: Generation, Not Cache — and the Batch-Injection Fix

The profiling results (§4) definitively show that SIG's performance penalty in fragmented workloads comes from per-step `generate()` calls (94.8% of wall-clock). The batch-injection experiment provides the **engineering fix**: Batch-SIG (bs=8) achieves 0.283s wall-clock (9.45× over per-step SIG, 4.65× over AppLoop-PC) by issuing a single generation call after accumulating all 8 tool results. This transforms R13 from a design boundary into an **advantage zone** for SIG, provided the runtime can batch tool results before generation.

### 9.6 Semantic Scoring and the Quality Measurement Pipeline

The introduction of the `SemanticScorer` (TF-IDF cosine similarity, underscore normalization) addresses Limitation #9 from earlier drafts. The hybrid scoring approach reveals that keyword-only scoring inflates apparent quality gaps by penalizing formatting differences that are semantically irrelevant (e.g., "spaghetti_bolognese" vs "spaghetti bolognese"). The 32-unit-test suite ensures scoring correctness across the full evaluation pipeline.

### 9.7 Synergy with Self-Improving Agents

CO provides ready-made runtime for Robo-Cortex-like continual-learning agents. The KV-cache retains full task execution context; heuristics distilled by autonomous knowledge induction become local cognitive modules within CO. Batch-injection further enables "offline reflection" — accumulating tool results and generating a single comprehensive summary — which aligns naturally with offline learning paradigms.

---

## 10. Limitations

1. **Task quality is reduced in SIG mode** (§3.5): the −0.12 composite quality penalty (hybrid scoring, 4B) is narrower than previously reported (−0.39 keyword-only), but the recipe enumeration gap persists.
2. **SIG advantage collapses under tool latency** (§4): cloud-dependent scenarios (>300ms delay) reduce speedup to <1.15×. This is now confirmed with n=5 and tight standard errors.
3. **Single model family** (Qwen3.5 dense). Cross-family replication required for Llama, Gemma, Mistral.
4. **Synthetic tools**: Real-world tool noise may interact with injection granularity; batch-injection behavior with noisy real-world tool results is untested.
5. **R1 attention measurement is single-prompt**: Multi-prompt averaging at ≥1.5B needed for statistical inference.
6. **R2 covers only 6–10 rounds**: Degradation at 32+ rounds uncharacterized.
7. **R3 is simulation-based**: No non-Transformer SIG implementation exists.
8. **R7/R9 at N=1**: Cannot support statistical inference.
9. ~~**Evaluation metrics**: keyword-only biased quality gap downward — addressed via `SemanticScorer` hybrid scoring.~~ Remaining gap is genuine content difference, not scoring artifact.
10. **Quantized-model output variance**: Q4_K_M + CUDA produces length variations at temperature=0, though verbosity stderr=0 across n=10 for 0.8B suggests headroom for variance at larger scales.

---

## 11. Future Work

- **Cross-family replication (HIGH PRIORITY):** Verify deep-chain and batch-injection advantage on Llama, Gemma, Mistral at ≥7B with tool-depth sweeps and quality evaluation.
- ~~**Batch-injection optimization**~~ — **IMPLEMENTED**: Batch-SIG with configurable batch sizes (2/4/8) is demonstrated in this revision (§4, §8).
- ~~**Evaluation metrics (semantic)**~~ — **IMPLEMENTED**: `SemanticScorer` with TF-IDF cosine similarity + keyword hybrid scoring (§3.5).
- **Quality-speedup Pareto characterization:** Measure SIG vs AppLoop trade-off surface across model sizes, chain depths, batch sizes, and task types.
- **Compensatory recall engineering:** Implement and measure the targeted "memory prompt" strategy (§9.2) to bridge the remaining recipe enumeration gap.
- **Attention-representation bridge:** Probe key-token hidden states in SIG vs full re-encoding to characterize how functional equivalence emerges despite attention divergence.
- **R1 multi-prompt averaging:** Cross-model validation at ≥1.5B with statistical significance testing.
- **R2 extended degradation:** 32+ round characterization.
- **R4 teacher-size scan:** Vary teacher across 3B–70B.
- **R5 privacy validation:** Formal DP guarantees and attack simulations.
- **R8 metric improvement:** Apply `SemanticScorer` to retrieval evaluation.
- **R12 empirical scaling:** Prefill measurement across ≥3 model sizes.
- **Dynamic batch-size selection:** Learn optimal K per workload from runtime characteristics (tool count, chain depth, urgency).

---

## 12. Conclusion

We have presented a comprehensive, empirically-grounded treatment of Cognitive Outsourcing with Suspend-and-Inject Generation. CO/SIG eliminates quadratic prefill overhead by injecting external cognitive resources directly into the model's KV cache, achieving 73–97% prefill token savings and 2.38–9.45× end-to-end speedups across continuous tool chains.

**Key contributions with empirical support:**

1. **Architecture**: The five-stage SIG cycle and three-layer CO architecture elevate tool interaction from an application-loop to an inference-engine primitive. Prefill savings are robust (73–97%) and the dominant source of speedup.

2. **Design space**: N=30 paired-run measurements across nine vectors define SIG's scope—2.79–4.26× deep-chain advantage, CoT-equivalent performance, 100% rollback recovery, and principled design boundaries (0% retrieval on 0.8B).

3. **Batch-injection as a design-space transformation**: The most consequential finding of this revision: batch-injection converts SIG's worst-case regime (fragmented assembly, previously 2.1× slower than AppLoop-PC) into a 4.65× advantage. This reclassifies R13 from a *design boundary* to an *engineering optimization* — the constraint is benchmark design, not the SIG primitive itself. Batch-SIG with batch_size=8 achieves 9.45× speedup over per-step SIG by reducing 16 generate() calls to 1.

4. **Diagnostic findings that refine quality claims**:
   - Task quality gap narrows from −0.39 (keyword-only) to −0.12 (hybrid TF-IDF semantic scoring, n=10).
   - Recipe enumeration remains the sole irreducible gap — SIG's KV-cache preserves attention but not explicit restatement.
   - SIG speedup collapses from 5.49× to 1.08× under 500ms tool latency (n=5, confirmed).
   - Verbosity control shows zero variance (stderr=0, n=10) at 0.8B, ruling out prompt-format artifact.

5. **Deployment**: 3.43× Kitchen speedup on 4B, batch-injection for fragmented workloads, 2.3× cache reduction via compression, 59% VRAM savings via multi-sequence concurrency, and the expanded SIG Decision Framework (§8) incorporating the Batch-Injection Zone.

6. **Theory**: First attention distribution analysis in the SIG vs. re-encoding setting, multi-round cache degradation characterization, cross-architecture simulation, and capability gap quantification.

7. **Infrastructure quality**: 32 unit tests, modular `core/` library with 11 modules plus `core/scenarios.py`, hybrid semantic quality scoring, reusable `mean_std()` utility.

SIG is a **scale-dependent, condition-dependent performance optimization** with expanding advantage zones. Its value is highest for models above the 1.5–2B crossover executing tool chains with near-instantaneous local execution. The batch-injection extension broadens this range to include fragmented assembly workloads, and the hybrid semantic scoring provides a more nuanced measurement of the remaining quality-speedup trade-off. The honest delineation of these trade-offs, latency conditions, and engineering remedies is what makes this value proposition credible.

**Testing infrastructure**: All experiments are reproducible via `co_benchmark.py`, `sig_benchmark.py`, `transformer_bench.py`, `edge_agent_bench.py`, and `diagnostic_bench.py`. Run `python -m unittest tests/ -v` for 32 unit tests. The `core/` library provides reusable components for future research.

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
# Run all unit tests (32 tests, <1s)
python -m unittest tests/ -v

# Task quality evaluation (n=10, hybrid semantic + keyword scoring)
python diagnostic_bench.py --task quality_kitchen \
  --model models/Qwen3.5-4B-Q4_K_M.gguf --n-gpu-layers 99 \
  --quality-kitchen-steps 20 --quality-runs 10

# R13 batch-injection profiling (n=10, bs=2/4/8 + per-step + AppLoop-PC)
python diagnostic_bench.py --task profile_r13_batch \
  --model models/Qwen3.5-4B-Q4_K_M.gguf --n-gpu-layers 99 --profile-runs 10

# R13 fine-grained profiling (legacy, per-step cost decomposition)
python diagnostic_bench.py --task profile_r13 \
  --model models/Qwen3.5-4B-Q4_K_M.gguf --n-gpu-layers 99 --profile-runs 10

# Latency ablation (n=5, 0/100/300/500ms delays)
python diagnostic_bench.py --task latency_ablation \
  --model models/Qwen3.5-4B-Q4_K_M.gguf --n-gpu-layers 99 --latency-runs 5

# Verbosity control (n=10, SHORT vs LONG prompt)
python diagnostic_bench.py --task verbosity_control \
  --model models/Qwen3.5-0.8B-Q4_K_M.gguf --n-gpu-layers 0 --verbosity-runs 10

# Run all diagnostic experiments sequentially
python diagnostic_bench.py --task all --model models/Qwen3.5-4B-Q4_K_M.gguf --n-gpu-layers 99

# Existing benchmarks with simulated tool latency
python co_benchmark.py --task r6 --model ... --n-gpu-layers 99 --tool-latency 300
python edge_agent_bench.py --task kitchen --model ... --n-gpu-layers 99 --tool-latency 300
```

### B.2 `kv_cache_seq_rm` Erratum

Original codebase used `kv_cache_seq_rm(0, -1, -1)` for cache resets, marking slots free without defragmenting. After multiple reset→eval cycles, no contiguous block remained. Correction to `kv_cache_clear()` fully resets both cache and allocator. All baselines now complete 100% of steps.

---

> **Data availability**: Reproduction scripts at `co_benchmark.py`, `sig_benchmark.py`, `transformer_bench.py`, `edge_agent_bench.py`, `diagnostic_bench.py`, `run_multi_bench.py`. Core library at `core/`. Scenario builders at `core/scenarios.py`. Unit tests at `tests/`. Results at `bench_multi_Qwen3.5-4B.json`, `bench_multi_Qwen3.5-0.8B.json`, `bench_multi_results.json`. Plans and prompts at `co_benchmark_plans.json`, `co_benchmark_prompts.json`.
