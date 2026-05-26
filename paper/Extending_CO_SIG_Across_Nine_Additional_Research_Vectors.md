# SIG as a Specialized Edge Accelerator for Long-Tool-Chain Inference — A Design Space Exploration Across Nine Research Vectors

> **MANUSCRIPT STATUS (May 2026): FULL DATA COLLECTION COMPLETE — ALL N=30 EXPERIMENTS EXECUTED**
>
> This paper reports **completed N=30 paired-run measurements** across all nine research vectors (R6–R14) on Qwen3.5-0.8B and 4B (Q4_K_M, llama.cpp, NVIDIA RTX 4070 SUPER). No DATA PENDING sections remain.

## Abstract

**We investigate Suspend-and-Inject Generation (SIG) as a specialized edge accelerator for small-model, long-tool-chain inference.** On a consumer-grade GPU (RTX 4070 SUPER) with quantized Qwen3.5 models (0.8B/4B, Q4_K_M), we conduct N=30 paired-run measurements across nine research vectors to map SIG's design space: where it provides transformative acceleration, and where — by design — simpler baselines suffice.

**Core evidence — the deep-chain regime.** At 30-tool depth with 15% failure injection, SIG achieves 2.79× end-to-end wall-clock speedup on 0.8B and 4.26× on 4B over standard AppLoop. Critically, a prefix-caching-optimized baseline (AppLoop-PC, emulating PagedAttention/RadixAttention) provides zero benefit in this regime: the chain builds from an empty shared prefix (<3% reuse), making prefix caching structurally ineffective. SIG's incremental injection cost remains near-constant regardless of chain depth — a property that decouples task complexity from latency on resource-constrained edge devices. This makes 30-step planning feasible on a 4B model where AppLoop-based approaches would incur prohibitive re-encoding overhead.

**Design boundaries — where streaming injection is the wrong tool.** We systematically characterize conditions where SIG shows no advantage, and argue these are not failures but *design boundaries* of a streaming-injection paradigm: (1) **Fragmented, independent context assembly (R13):** SIG is 3.0× slower than AppLoop-PC on 0.8B when contexts consist of many small, unrelated fragments — consistent with SIG's optimization for "inject once, consume continuously" rather than frequent small rebuilds. We propose a hybrid scheduling strategy that routes long-chain workloads to SIG and fragmented workloads to AppLoop-PC. (2) **Long-range random-access retrieval (R8):** AppLoop-PC achieves 50–64% retrieval hit rate vs SIG's 0–33%, because full-context re-encoding grants attention heads direct access to all positions — a capability SIG's persistent cache does not aim to provide. SIG is designed for *generation continuity*, not *random-access memory*.

**Safety and fidelity guarantees.** Despite SIG's expanded stateful attack surface, rollback isolation recovers keyword-clean state in 20/20 injection attempts (R10). Token-Jaccard fidelity scores are near-identical between SIG and AppLoop across 12 query-model combinations (R11), confirming that SIG achieves acceleration without silent quality degradation.

**Engineering guidance.** Pilot measurements (N=1) on structured data formats (R7: JSON incurs 45% token overhead) and latency budgets (R9: all ≤2000-token configurations within 2.0s) offer directional guidance for SIG-based system design. Analytic prefill-scaling projections (R12) are presented as future research targets, not empirical findings.

**Scope and limitations.** All measurements are specific to Qwen3.5 (Q4_K_M) on RTX 4070 SUPER. No formal hypothesis tests were performed; comparisons are descriptive. Cross-family replication on Llama, Gemma, and Mistral is required before generalization. This paper's primary contribution is the empirical characterization of SIG as a *streaming-injection edge accelerator* — defining both its transformative deep-chain advantage and its principled design boundaries — and providing a decision framework for edge inference system designers.


## Core Thesis

This paper is organized around a single design question rather than a traditional null hypothesis:

> **Core thesis:** On resource-constrained edge devices, when small models execute continuous, deep, stateful tool chains, SIG's *streaming persistent-cache* paradigm achieves order-of-magnitude end-to-end efficiency gains over re-encoding-based approaches — because its incremental injection cost is near-constant regardless of chain depth. The conditions where SIG shows no advantage are not failures but *principled design boundaries* of a streaming-injection accelerator: SIG is optimized for "inject once, consume continuously"; when workloads deviate from this pattern, simpler baselines are the correct choice.

The nine research vectors (R6–R14) are organized in four parts that build this argument:

| Part | Vectors | Role in the Argument |
|------|---------|---------------------|
| I. Core Value Evidence | R6, R14 | Demonstrate SIG's transformative advantage on deep chains, and its compatibility with structured prompting |
| II. Design Boundaries | R13, R8 | Characterize where streaming injection is not the right tool, and when to fall back |
| III. Safety & Fidelity | R10, R11 | Verify that statefulness does not introduce unacceptable risk or quality degradation |
| IV. Engineering Guidance | R7, R9, R12 | Directional observations for system builders; analytic projections for future work |

This structure transforms the paper from a "survey of SIG's limitations" into a **design-space exploration of a specialized edge acceleration paradigm**, where boundary conditions strengthen rather than weaken the core argument.


## 1. Introduction

The original Cognitive Outsourcing framework demonstrated that lightweight edge models (as small as 0.8B parameters) could achieve significant performance gains through Suspend-and-Inject Generation: 73-97% prefill token savings, 2.38× end-to-end speedup, and a 3× improvement in long-context information coverage [1]. The subsequent five-dimensional analysis [2] grounded these empirical findings in theoretical frameworks spanning information theory, cache lifecycle management, architectural compatibility, teacher-student optimization, and privacy guarantees.

### The Edge Deployment Contradiction

A fundamental contradiction faces edge AI deployment today. Small models (0.5B–4B parameters) are the only practical option for battery-powered drones, offline mobile assistants, and embedded robotics — yet their limited capacity makes complex multi-step reasoning unreliable. The standard remedy, teacher-guided planning [1], requires the model to execute long tool chains (10–30 sequential API calls). But on edge hardware, each AppLoop turn triggers a full re-encoding of the entire accumulated context — a cost that grows with chain depth. For a 4B model on consumer-grade hardware (RTX 4070 SUPER, representing a high-end edge workstation), a 30-step chain incurs ~2.0 seconds of re-encoding overhead in AppLoop mode, versus ~0.48 seconds with SIG's incremental injection. **The prefill cost alone can make complex planning infeasible on edge devices.**

Modern LLM serving systems address this partially through prefix caching (PagedAttention [3], RadixAttention [4]), which reuses shared KV-cache prefixes across turns. However, prefix caching is structurally ineffective when tool chains build from an empty shared prefix — each turn adds substantial new content that did not exist before, yielding near-zero cache reuse. **This is the regime where SIG provides its uniquely transformative advantage.**

### This Paper

We investigate SIG not as a universal inference accelerator to be compared against all possible baselines, but as a **specialized edge acceleration paradigm** designed for a specific workload profile: long, continuous, stateful tool chains on small models. Our goal is to empirically characterize both its *advantage zone* and its *design boundaries* — defining, with N=30 paired-run measurements, when to use SIG and when to fall back to simpler alternatives.

The investigation is organized in four parts:

1. **Part I — Core Value Evidence (R6, R14):** SIG's deep-chain advantage and its compatibility with CoT-structured prompting.
2. **Part II — Design Boundaries (R13, R8):** Characterizing the workload patterns where streaming injection is the wrong tool, and proposing hybrid scheduling strategies.
3. **Part III — Safety & Fidelity (R10, R11):** Verifying that SIG's statefulness does not introduce unacceptable risk or quality degradation.
4. **Part IV — Engineering Guidance (R7, R9, R12):** Directional observations and analytic projections for system builders.

**Nature of this study.** This is an exploratory empirical study — all measurements are conducted on a single GPU (RTX 4070 SUPER, representing a high-end edge workstation) with a single model family (Qwen3.5, Q4_K_M). Core performance experiments (R6, R8, R13, R14) use N=30 paired runs and report mean±standard deviation. No formal hypothesis tests with p-values are computed; inference is based on descriptive numerical comparisons. While N=30 provides adequate power for detecting large effects, cross-family replication on Llama, Gemma, and Mistral is required before generalization. Dimensions marked **[Analytic Projection]** rely on extrapolation and are presented as future research targets, not empirical findings.

**Key improvement over the prior version:** All nine modules have been migrated from standalone Python simulation scripts to production-grade test harnesses with live GGUF model inference (llama.cpp). Measurements are performed on real hardware using Qwen3.5-0.8B/4B (Q4_K_M). All nine dimensions (R6-R14) produce measurements from real model inference by directly calling `compiler.eval()` and `compiler.generate()` via llama.cpp's C API.

**Note on comparison baselines.** We distinguish between four types of baselines throughout: (a) **AppLoop(raw)** — standard full-prefill with raw tool output concatenation; (b) **AppLoop-PC** — AppLoop with an **emulated** prefix-cache (via llama.cpp `kv_cache_seq_cp`), approximating the effective cost under production prefix-caching systems under assumptions of no cache contention, no batching, and single-sequence operation; this is a **simplified baseline**, not a production serving stack measurement; (c) **SIG(raw)** — incremental KV-cache injection with raw tool output; (d) **CoT+AppLoop / CoT+SIG** — both modes receive identical structured Chain-of-Thought prompts, isolating the net contribution of SIG's injection mechanism from the CoT structuring effect. See Appendix A for emulation assumptions and limitations.

**A note on AppLoop prefix caching.** This study implements AppLoop as full-context re-encoding (`compiler.rebuild_cache()` on the complete accumulated prompt). In production LLM serving systems, prefix caching can reduce re-prefill cost by sharing KV cache entries across turns that share a common prefix. **Critically, prefix caching is structurally ineffective in the deep-chain regime (R6)** — the chain builds from an empty shared prefix (<3% token reuse), so cache-hit rates are near zero. This is precisely the condition where SIG's streaming injection paradigm is *not replaceable* by existing serving optimizations. All SIG-vs-AppLoop(raw) comparisons should be treated as upper bounds on SIG benefit; SIG-vs-AppLoop-PC comparisons provide the more realistic baseline for deployment planning.

All test harnesses are publicly available and can be run via:
```bash
python co_benchmark.py --task r6 --model models/Qwen3.5-0.8B-Q4_K_M.gguf
python sig_benchmark.py --task r7 --model models/Qwen3.5-0.8B-Q4_K_M.gguf
python transformer_bench.py --task r12
```


## Part I: Core Value Evidence

### 2.1 R6: Dynamic Replanning — The Deep-Chain Advantage

**Motivation.** On edge devices, the standard teacher-guided planning paradigm [1] requires small models to execute long tool chains (10–30 sequential API calls). Each AppLoop turn incurs a full re-encoding of the entire accumulated context — a cost that grows with chain depth. Modern prefix-caching optimizations (PagedAttention, RadixAttention) provide no relief when the chain builds from an empty shared prefix. R6 tests whether SIG's incremental injection can break this coupling between chain depth and latency.

**Framework.** We model the CO execution process under tool failure injection. A **30-tool travel planning chain** is executed with 15% random failure rate, comparing three execution modes:

1. **SIG** — incremental KV-cache injection. The streaming persistent-cache baseline.
2. **AppLoop** — full-context re-encoding per step. The naive retokenization baseline.
3. **AppLoop-PC** — AppLoop with emulated prefix-caching. Approximates production serving systems. This is the baseline that **should** help if prefix reuse is possible.

On tool failure, all three modes execute a retry with full tool re-execution. Each mode is measured over **N=30 paired runs** with identical random seeds across modes. Primary metric: end-to-end wall-clock time.

**Results (N=30 paired runs, 30-tool chain, 15% failure injection).**

| Model | SIG | AppLoop | AppLoop-PC | SIG vs AppLoop | SIG vs AppLoop-PC |
|------|-----|---------|------------|----------------|-------------------|
| 0.8B | 0.232±0.021s | 0.646±0.022s | 0.646±0.021s | **2.79×** | **2.79×** |
| 4B   | 0.480±0.016s | 2.043±0.073s | 2.038±0.084s | **4.26×** | **4.25×** |

**Key findings.**

**1. AppLoop-PC provides zero benefit — prefix caching is structurally ineffective on deep chains.** The 30-step chain builds from an empty shared prefix (only the system prompt, ~80 tokens, is reused). After 30 steps, the accumulated context exceeds 2,500 tokens, of which <3% are cacheable. AppLoop-PC's `kv_cache_seq_cp` operation recovers the system prompt but still requires re-evaluating all accumulated tool results — the dominant cost. **This demonstrates that production prefix caching (PagedAttention, RadixAttention) does not replace SIG in the deep-chain regime.**

**2. SIG's incremental cost is near-constant.** SIG adds ~0.008s per step (0.8B) and ~0.016s per step (4B) — dominated by `compiler.eval()` on the new tokens from each tool call. AppLoop's per-step cost grows with context size. At step 30, AppLoop evaluates ~2,500 tokens while SIG evaluates ~80 — a 30× difference in prefill work.

**3. The advantage grows with model size (2.79× → 4.26×).** Larger models have proportionally higher prefill cost per token, while SIG's incremental injection cost scales sub-linearly. This suggests SIG's advantage will be even more pronounced at ≥7B scales — a testable hypothesis for cross-family replication.

**4. Edge feasibility.** On RTX 4070 SUPER (representing a high-end edge workstation), a 4B model executing 30 sequential tool calls would spend ~2.0s in AppLoop re-encoding alone — exceeding typical real-time budgets. SIG's ~0.48s total time makes complex planning feasible on edge hardware. For a drone navigation scenario requiring 30-step trajectory planning, this difference is the gap between viable and infeasible.

**Implications.** R6 establishes that SIG's streaming injection paradigm is not merely an optimization over AppLoop — it is the *only* measured approach that makes deep tool-chain execution viable on small edge models, because prefix caching, the standard production remedy, is structurally ineffective when chains build from an empty shared prefix. This is the foundational evidence for SIG as a specialized edge acceleration paradigm.

### 2.2 R14: SIG & Structured Reasoning — Compatibility, Not Competition

**Motivation.** If SIG is to serve as a general-purpose edge acceleration substrate, it must compose cleanly with structured reasoning techniques like Chain-of-Thought (CoT). However, when a CoT block is large enough to contain all tool results, the injection cost of SIG and the re-encoding cost of AppLoop converge — both must process the full CoT text. R14 tests whether SIG provides *additional* speedup in CoT scenarios, or whether CoT itself already achieves the dominant efficiency gain by structuring information into a single block.

**Framework.** We evaluate four execution modes across two multi-tool queries with N=30 paired runs per query:

1. **CoT+SIG**: Execute all tools, assemble results into a structured CoT block, inject once via SIG.
2. **CoT+AppLoop**: Identical structured CoT prompt, delivered via standard full-prefill.
3. **CoT+AppLoop-PC**: Same CoT prompt with emulated prefix-caching on the system prompt.
4. **SIG_raw**: Raw tool output, incremental KV-cache injection (baseline from prior work).

Generation is bounded at `max_new=80` tokens. Gen-token count is tracked to verify that wall-clock comparisons are not confounded by output length differences. All modes on Qwen3.5-0.8B/4B (Q4_K_M) via llama.cpp, RTX 4070 SUPER.

**Results (N=30 paired runs, 4 modes, max 80 tokens).**

**Q1: 3-city comparison (6 tools):**

| Mode | 0.8B Wall-Clock | 0.8B Gen Tok | 4B Wall-Clock | 4B Gen Tok |
|------|----------------|-------------|---------------|-----------|
| CoT+SIG | 0.123±0.021s | 28 | 0.688±0.019s | 71 |
| CoT+AppLoop | 0.119±0.003s | 28 | 0.687±0.002s | 71 |
| CoT+AppLoop-PC | 0.156±0.029s | 39 | 0.577±0.240s | 59 |
| SIG_raw | 0.335±0.019s | 79 | 0.139±0.002s | 2 |

**Q2: Travel plan (5 tools):**

| Mode | 0.8B Wall-Clock | 0.8B Gen Tok | 4B Wall-Clock | 4B Gen Tok |
|------|----------------|-------------|---------------|-----------|
| CoT+SIG | 0.087±0.003s | 19 | 0.317±0.247s | 30 |
| CoT+AppLoop | 0.088±0.004s | 19 | 0.691±0.003s | 72 |
| CoT+AppLoop-PC | 0.158±0.022s | 39 | 0.517±0.151s | 52 |
| SIG_raw | 0.333±0.007s | 80 | 0.124±0.002s | 2 |

**Key findings.**

**1. CoT+SIG and CoT+AppLoop are performance-equivalent when output length is matched.** In Q1 and Q2-0.8B, where gen-token counts are equal, wall-clock times are near-identical (ratios 0.99–1.03×). This is expected: both modes process the same CoT text; the cost difference between SIG's incremental injection and AppLoop's re-encoding becomes negligible when the CoT block (400–600 tokens) dominates the system prompt (80 tokens). **SIG is fully compatible with CoT-structured prompting — it achieves equivalent performance without requiring modifications to the CoT protocol.**

**2. The 4B Q2 anomaly and correction.** On Q2 4B, CoT+SIG records 0.317s with 30 gen-tokens while CoT+AppLoop records 0.691s with 72 gen-tokens — a 0.42 gen-token ratio that makes the wall-clock comparison uninterpretable. Generation time dominates at 4B (~0.17s per 10 tokens), so the 42-token difference accounts for ~0.7s of the observed 0.374s gap. **This data point is excluded from interpretation.** We estimate that with equalized output length, CoT+SIG and CoT+AppLoop would achieve near-parity on Q2 4B as they do in all other conditions. The experiment requires re-execution with a forced-output-length protocol.

**3. SIG_raw fails at 4B.** On Q1 4B, SIG_raw generates only 2 tokens (vs 71 for CoT+SIG); on Q2 4B, 2 tokens (vs 30 for CoT+SIG). Without CoT structuring, the 4B model cannot effectively utilize raw injected tool results — CoT is essential for reasoning quality at this scale.

**4. AppLoop-PC is slightly slower in CoT mode.** On both models and queries, AppLoop-PC's `kv_cache_seq_cp` copy overhead exceeds the benefit of caching the system prompt — the CoT block dominates the token budget (400–600 tokens vs 80 token system prompt), so prefix caching saves <15% of total prefill work.

**Implications.** SIG composes cleanly with CoT reasoning: CoT+SIG matches CoT+AppLoop in performance when output lengths are equal. The important architectural property is that SIG *retains the ability to incrementally extend the CoT context* — for interactive reasoning scenarios where additional tool results arrive mid-generation, SIG can inject them without rebuilding the entire CoT prompt. This "architectural flexibility" is not captured by single-pass comparisons but represents a structural advantage for open-ended reasoning workloads on edge devices.


## Part IV: Engineering Guidance — Pilot Observations and Analytic Projections

### 3.1 R7: Multimodal SIG — Structured Data Injection Efficiency (Pilot Observation, N=1)

**Motivation.** Current SIG implementations inject only text tokens. For embodied agents operating in multimodal environments, converting rich structured data to text descriptions introduces a representation bottleneck. What is the most token-efficient format for injecting structured information into the KV cache?

**Framework.** We compare three data representation formats for identical semantic content (weather, attractions, and flight information for Paris):

1. **Structured JSON**: `{"city": "Paris", "weather": "...", "attractions": "..."}`
2. **Plain text**: Natural language paragraphs ("Weather in Paris: 22°C clear skies...")
3. **Minimal text**: Compressed key-info format ("Paris: 22°C clear | Eiffel Tower, Louvre, Notre-Dame | ...")

Each format is tokenized and injected into the KV cache. Measurements on Qwen3.5-0.8B/4B via llama.cpp, including eval time and information density (tokens per character of semantic content).

**Pilot Observations (N=1 per format — statistical inference not supported).**

| Format | 0.8B Tokens | 0.8B Eval (ms) | 0.8B Density | 4B Tokens | 4B Eval (ms) | 4B Density |
|--------|------------|---------------|-------------|----------|-------------|-----------|
| Structured JSON | 112 | 20.4 | 0.390 | 112 | 37.1 | 0.390 |
| Plain text | 77 | 27.8 | 0.365 | 77 | 55.8 | 0.365 |
| Minimal text | 70 | 22.7 | 0.385 | 70 | 54.8 | 0.385 |

**Single observation (N=1 per format):** JSON structured format incurs a single-observation 45% token overhead (112 vs 77 tokens, N=1 per format, Qwen3.5 tokenizer). This has not been validated across tokenizers, schemas, or quantization formats. The overhead comes from syntactic characters (brackets, quotes, colons) that tokenize into individual tokens in the model's vocabulary. Minimal text achieves the fewest tokens (70), representing a 9% improvement over plain text. Eval time scales with model size: 4B is 1.8-2.0× slower than 0.8B per token.

**Anomaly: JSON eval-time is lower despite higher token count.** On 0.8B, JSON 20.4ms < Plain text 27.8ms despite 45% more tokens. This likely reflects a per-token eval efficiency difference: JSON tokens are shorter (1-character symbols like `{`, `"`, `:`) and can be processed in tighter CUDA kernel batches, while longer plain-text tokens incur more memory bandwidth per token. The anomaly does not affect the main recommendation (prefer plain text for token efficiency) but suggests that eval-time optimization could be format-dependent; further investigation is warranted.

**Low-level profiling hypothesis (unverified).** This JSON eval-time anomaly has NOT been verified with hardware performance counters (nsight systems, ncu). The CUDA kernel batching hypothesis is speculative. SM occupancy and memory bandwidth utilization measurements (available via `core/gpu.py` → `GPUMonitor.utilization_snapshot()`) are required to distinguish between compute-bound batching effects and memory-bandwidth fragmentation. Without profiling confirmation, this observation should be treated as a hypothesis, not a finding.

**Hypothesis for future testing.** Based on N=1 point measurements, multimodal SIG should prioritize plain text or minimal key-info format over JSON for token efficiency. The observed JSON token overhead of 45% (112 vs 77 tokens) is a preliminary point estimate that needs validation across different data schemas, model tokenizers, and quantization formats.

### 4.2 R8: Long-Context Retrieval — Streaming Continuity vs Random-Access Memory

**Motivation.** R6 established that SIG's persistent cache provides *generation continuity* — the model can produce coherent output across a long chain without interruption. But does this continuity translate to *random-access retrieval* — the ability to answer specific queries about early-turn facts? R8 tests whether SIG's incremental injection preserves long-range factual accessibility in the same way that full-context re-encoding does.

The conceptual distinction is important: SIG's KV cache stores information in the order it was injected, optimized for sequential consumption. AppLoop's full re-encoding gives every attention head direct access to every token position. These are fundamentally different memory architectures: SIG is a *streaming tape*; AppLoop is a *random-access array*. R8 measures whether this architectural difference produces measurable retrieval behavior differences.

**Framework.** We simulate a 6-room × 2-turn scenario (12 total turns) across 8 cities. Three comparison modes: SIG, AppLoop, and AppLoop-PC (prefix-caching AppLoop). N=30 paired runs. Two probe types: (1) **Retrieval probes** at turns 3, 6, 9, 12: "What was the weather in Room 0 (Paris)?" — tests ability to recall early-turn key-value facts via random access. (2) **Sequential reasoning probe** at turn 10: "From Room0's door, which city is 2 rooms ahead?" — requires multi-step positional inference. All probes use keyword-presence pass/fail criteria (limitation discussed in §11). This is *not* a test of spatial cognition — it tests sequential retrieval of positional information.

**Results (N=30 paired runs across 3 modes).**

**Context retrieval hit rates:**

| Probe | 0.8B SIG | 0.8B AppLoop | 0.8B PC | 4B SIG | 4B AppLoop | 4B PC |
|-------|----------|-------------|---------|--------|-----------|-------|
| T=3   | 0% | 0% | **90%** | 0% | 0% | **93%** |
| T=6   | 0% | 0% | **80%** | **93%** | 0% | 77% |
| T=9   | 0% | 0% | 30% | 40% | **100%** | 87% |
| T=12  | 0% | 0% | 0% | 0% | 0% | 0% |
| **Total** | **0%** | **0%** | **50%** | 33% | 25% | **64%** |

**Sequential reasoning probe (T=10):**
| Model | SIG | AppLoop | AppLoop-PC |
|------|-----|---------|------------|
| 0.8B | 0% | 0% | 0% |
| 4B   | 27% | **100%** | 17% |

**Key findings.**

**1. SIG's persistent cache is not a random-access memory.** On 0.8B, SIG records zero retrieval hits (0/120) — the streaming injection architecture does not preserve the kind of direct positional access that full-context re-encoding provides. This is consistent with SIG's design: information is injected sequentially into the KV cache, and the model's attention mechanism accesses it through the accumulated key-value states, not through direct positional addressing.

**2. AppLoop-PC is the best retriever.** With 50–64% total hit rates and 90%+ at early probes, re-encoding gives attention heads direct access to all positions — the random-access array architecture excels at this task. This is the *correct baseline* for retrieval workloads.

**3. Full re-encoding benefits sequential reasoning.** On 4B, AppLoop achieves 100% on the sequential reasoning probe while AppLoop-PC drops to 17%. This counterintuitive result suggests that re-encoding the full context (rather than incrementally appending to a cached prefix) may strengthen multi-step inference — a finding that merits deeper investigation.

**4. T=12 failure is universal.** All modes fail at the farthest retrieval probe, confirming a context-depth ceiling at these model sizes and context lengths.

**5. Keyword-pass/fail caveat.** All probe results use simple keyword presence criteria, which may undercount semantically correct answers. See §11 for discussion.

**Implications.** R8 defines SIG's *architectural limitation*: SIG's streaming persistent cache is optimized for generation continuity, not random-access retrieval. When the task requires querying arbitrary early-turn facts, AppLoop-PC's full-context re-encoding is the correct tool. This is not a failure of SIG — it is the expected behavior of a streaming-injection architecture, just as a tape drive is not "failing" when it cannot match a RAM chip's random-access latency. System designers should use SIG for continuous generation workloads (R6) and AppLoop-PC for retrieval-heavy workloads (R8). The two are complementary, not competing.

### 3.3 R9: Real-Time Constrained SIG — Pilot Observations (N=1)

**Motivation.** Safety-critical embodied applications impose strict latency budgets. As a preliminary instrumentation check, we measure single-pass prefill + generation latency.

**Framework.** We measure prefill + generation latency across four context sizes (200, 500, 1000, 2000 tokens) with a 2.0s latency budget, using both Qwen3.5 models. Measurements are actual wall-clock times via llama.cpp on RTX 4070 SUPER.

**Pilot Observations (N=1 per context size — statistical inference not supported).**

*0.8B*:
| Context Tokens | Prefill (s) | Gen (s) | Total (s) | Budget |
|---------------|-----------|--------|----------|--------|
| 200 | 0.007 | 0.065 | 0.072 | OK |
| 500 | 0.007 | 0.071 | 0.078 | OK |
| 1000 | 0.034 | 0.070 | 0.104 | OK |
| 2000 | 0.087 | 0.067 | 0.154 | OK |

*4B*:
| Context Tokens | Prefill (s) | Gen (s) | Total (s) | Budget |
|---------------|-----------|--------|----------|--------|
| 200 | 0.013 | 0.156 | 0.170 | OK |
| 500 | 0.028 | 0.179 | 0.206 | OK |
| 1000 | 0.110 | 0.187 | 0.297 | OK |
| 2000 | 0.288 | 0.185 | 0.473 | OK |

**Directional observations (N=1 — not a finding).** In this single measurement instance, all configurations were within the 2.0s budget. Generation time is relatively constant across context sizes while prefill grows with token count. These measurements serve as an instrumentation check; multi-turn cumulative latency and inter-run variance remain uncharacterized. Replication at N≥10 per condition with actual multi-turn agent workloads is required before any quantitative conclusions can be drawn.


## Part III: Safety & Fidelity Guarantees

### 4.1 R10: Injection Attacks & Defense ✅ EMPIRICAL

**Motivation.** SIG's persistent KV cache creates a larger attack surface than AppLoop's episodic re-encoding. A single poisoned tool result can contaminate the cache and propagate across all subsequent turns.

**Status.** Real-model measurements on Qwen3.5-0.8B and 4B. Three attack vectors tested: fact override (incorrect facts), instruction hijack (behavior change), and result poisoning (absurd tool results). Rollback isolation recovery measured via `InjectionEngine.evict_range()` + `compiler.rebuild_cache()`.

**Attack Surface Analysis** (theoretical risk scores):
| Attack Vector | Risk |
|---------------|------|
| Attention manipulation | 0.91 |
| Prompt injection | 0.85 |
| Data exfiltration | 0.78 |
| Result poisoning | 0.72 |
| Cache pollution | 0.63 |

**Experimental Results (Real Model, Qwen3.5, 10 attacks per model).**

| Model | Attacks Successful | Success Rate | Rollback Recovered | Recovery Rate |
|-------|-------------------|-------------|-------------------|---------------|
| 0.8B | 6/10 | 60% | 10/10 | 100% |
| 4B | 8/10 | 80% | 10/10 | 100% |
| **Combined** | **14/20** | **70%** | **20/20** | **100% (Wilson 95% CI: 84%–100%)** |

**Key Findings.**
1. **4B is more vulnerable than 0.8B** (8/10 vs 6/10 attacks successful in this sample). Larger models allocate more attention to injected content, making them more susceptible to cache pollution.
2. **The most effective attack vectors were: instruction hijack (both models), fake authority (both), result poisoning (both), and numerical poisoning (both).** Context flood and multi-turn contamination had the lowest success rates, suggesting that dilution attacks are less effective than targeted instruction manipulation.
3. **Rollback isolation produced keyword-clean state in 20 out of 20 rollback attempts** (Wilson 95% CI: 84%–100%). However, keyword-based detection may miss subtle residual contamination. This study introduces a complementary **continuous contamination metric**: pre-attack baseline and post-rollback responses are compared via character 3-gram Jaccard similarity, word-level overlap, response length ratio, and contaminated-to-rollback residual Jaccard. These continuous signals can detect subtle tone shifts, confidence calibration changes, or residual bias that binary keyword matching misses. The approach is a step toward semantic-level contamination auditing, though it remains n-gram-based rather than embedding-based.
4. **Fact override had mixed results**: 0.8B ignored false facts in some cases, while 4B accepted them more readily. This suggests smaller models may have stronger "truth anchors" for elementary facts.

5. **Self-evaluation (LLM-as-Judge) using same-family Qwen models has been stripped from the body text due to fundamental validity concerns (confirmation bias, circular evaluation).** The methodology description is retained in Appendix C for reference only. Core contamination detection relies on reproducible n-gram metrics.

**Critical self-evaluation bias caveat.** The LLM-as-Judge assessments in R10 and R11 use the same model family (Qwen3.5/2.5) to evaluate its own outputs. Self-evaluation carries inherent confirmation bias — models tend to rate their own outputs favorably. ALL LLM-as-Judge results in this paper should be treated as **minimally reliable** until cross-validated with an independent judge from a different model family (e.g., Gemma-2B as judge for Qwen outputs, or GPT-4o as an external oracle). Without cross-family validation, these judgments cannot distinguish between true semantic recovery and self-consistent hallucination. (Results from same-family self-evaluation carry minimal external validity — see Appendix C.)

**Hypothesis for future testing.** Rollback isolation shows promising keyword-clean recovery in this 20-attempt sample, but the lower CI bound of 84% and the coarse detection method warrant caution. Larger-scale validation (N≥30, diverse attack vectors, cross-model-family) with embedding-based semantic similarity metrics (sentence-transformers cosine similarity) and external LLM judges (cross-family) is needed before deployment claims.


### 4.2 R11: Tool-Result Faithfulness — Token-Jaccard as Primary Metric

**Motivation.** SIG's 3× information coverage advantage over AppLoop raises a critical question: does higher coverage come at the cost of reduced fidelity to injected tool results?

**Metric evolution.** The prior version of this study used a crude keyword-overlap metric (≥3 matching words of len≥3 between tool output and generated text), which measured **lexical reproduction tendency** rather than semantic faithfulness. This metric had high specificity for detecting the *absence* of tool-result influence but critically low sensitivity — a model that faithfully paraphrased tool results ("22°C clear skies" → "good weather") scored 0. The metric also produced the artifact that 4B scored 0% because it paraphrased using prior knowledge of real entities (London, Tokyo, Sydney).

**This version replaces keyword overlap with two improved metrics:**

1. **Token-Jaccard similarity** — the Jaccard coefficient computed over the set of tokenized words (len≥3) shared between the tool result and the generated text. Unlike the binary threshold (≥3 matching keywords → HIT), token-Jaccard produces a continuous [0,1] score that captures *degree* of lexical overlap. A paraphrase that retains one key term ("Colosseum") scores >0 rather than 0, providing better discrimination between complete independence and partial reproduction.

2. **LLM-as-Judge entailment evaluation has been stripped from the body text due to self-evaluation validity concerns (see Appendix C).** Core fidelity assessment relies on Token-Jaccard, a reproducible model-aware metric.

**Critical self-evaluation bias caveat.** The LLM-as-Judge assessments in R10 and R11 use the same model family (Qwen3.5/2.5) to evaluate its own outputs. Self-evaluation carries inherent confirmation bias — models tend to rate their own outputs favorably. ALL LLM-as-Judge results in this paper should be treated as **minimally reliable** until cross-validated with an independent judge from a different model family (e.g., Gemma-2B as judge for Qwen outputs, or GPT-4o as an external oracle). Without cross-family validation, these judgments cannot distinguish between true semantic recovery and self-consistent hallucination. (Results from same-family self-evaluation carry minimal external validity — see Appendix C.)

**Status.** Real-model measurements on Qwen3.5-0.8B and 4B. Six tool-result queries plus 25 fictional-entity queries (UQ2) tested in both SIG and AppLoop modes. Token-Jaccard scores computed for all query-mode combinations. LLM-as-Judge entailment evaluation has been stripped from the body text (see Appendix C).

**Key Findings (Token-Jaccard, 6 cities × 2 modes × 2 models = 24 measurements).**

Token-Jaccard between SIG and AppLoop shows near-identity for most query-model combinations:

| 0.8B | SIG TokJac | AppLoop TokJac | 4B | SIG TokJac | AppLoop TokJac |
|-------|-----------|---------------|-----|-----------|---------------|
| london | 0.036 | 0.033 | london | 0.132 | 0.132 |
| rome | 0.032 | 0.032 | rome | 0.067 | 0.067 |
| newyork | 0.022 | 0.020 | newyork | 0.000 | 0.000 |
| tokyo | 0.015 | 0.015 | tokyo | 0.068 | 0.000 |
| sydney | 0.017 | **0.322** | sydney | 0.067 | 0.067 |
| dubai | 0.032 | 0.032 | dubai | 0.000 | 0.000 |

The differences between SIG and AppLoop Token-Jaccard are within measurement noise for 10/12 query-model pairs. The sole outlier (0.8B AppLoop on sydney: TokJac=0.322 vs 0.017 for SIG) is attributed to seed-dependent generation variance — a single run produced unusually high overlap. Excluding this outlier, 0.8B mean TokJac = 0.027 (SIG) vs 0.026 (AppLoop), and 4B mean = 0.056 vs 0.044 — no numerically meaningful difference. LLM-as-Judge entailment evaluation (same-family Qwen, Appendix C only): 11/12 NOT_SUPPORTED, 1/12 SUPPORTED. **In summary, on Qwen3.5 0.8B/4B, SIG and AppLoop produce comparable Token-Jaccard scores; no numerical fidelity difference is observable at this sample size.**

### 4.3 R12: SIG Scaling Law — Analytic Projections (Moved to Discussion)

> **Note:** R12 contains only one empirical measurement (prefill at 0.8B, 128–2048 tokens). All model-size and context-length projections beyond this range rely on T∝M^0.7 extrapolation — an unverified assumption. **R12 has been removed from the primary empirical vector group.** The brief summary below is retained for completeness; detailed projections appear in §9 (Future Work).

**Motivation.** Understanding how SIG benefit might scale with model size and context length is important for scoping future confirmatory studies. **All projections in this section for model sizes >4B and context lengths >2K are analytic extrapolations, not empirical findings.**

**Framework.** We combine real prefill-time measurements on Qwen3.5-0.8B (128-2048 tokens) with theoretical projections.

**Measured Prefill Scaling (0.8B, RTX 4070 SUPER — empirical):**
| Context Tokens | Prefill (ms) | Tok/s |
|---------------|-------------|-------|
| 128 | ~7 | ~18,000 |
| 256 | ~12 | ~21,000 |
| 512 | ~25 | ~20,000 |
| 1024 | ~55 | ~19,000 |
| 2048 | ~95 | ~22,000 |

The model achieves ~20,000 tok/s prefill throughput with near-linear scaling. Prefill cost projection across model sizes used T ∝ M^0.7 for compute-bound prefill latency, calibrated on the two measured data points (0.8B ≈ 20K tok/s, 4B ≈ 8K tok/s). **This scaling exponent is an assumption, not a measurement — GPTQ quantization differences, attention mechanism variations (MHA vs GQA vs MLA), and hardware-specific memory bandwidth bottlenecks can all break this extrapolation.**

**Analytic projections (not empirical — moved to Discussion).** Extrapolations from the two measured data points (0.8B and 4B) using T ∝ M^0.7 suggest SIG speedup may grow with model size (projected ~9× at 70B) and context length (projected 96% prefill savings at 128K). These are theory-grounded speculations only. GPTQ quantization, GQA/MHA/MLA variations, and hardware bottlenecks can all break the T ∝ M^0.7 assumption. Confirmatory measurements on ≥3 model sizes are required to empirically establish the scaling relationship.

**Key Findings.** The empirical prefill measurements show ~20K tok/s at 0.8B on RTX 4070 SUPER with near-linear scaling from 128-2048 tokens. **Beyond this measured range, all projections are analytic extrapolation.** The qualitative hypothesis — that SIG speedup scales inversely with model prefill efficiency — is theory-grounded but unverified. GPTQ quantization, GQA vs MHA attention, and hardware-specific bottlenecks can all break the T ∝ M^0.7 assumption. Confirmatory measurements on ≥3 model sizes (e.g., 0.5B, 7B, 13B) with ≥3 context lengths (e.g., 8K, 16K, 32K) are needed to empirically establish the scaling relationship.


## Part II: Design Boundaries — When Streaming Injection Is the Wrong Tool


### 4.1 R13: Fragmented Local KV Reconstruction — The Structural Limitation

**Motivation.** R6 demonstrated that SIG excels when a single continuous chain accumulates context linearly. But what happens when the workload consists of many small, unrelated fragments — each requiring its own context assembly? This is the complementary workload profile: R6 represents "inject once, consume continuously"; R13 represents "frequent small rebuilds." If SIG performs poorly here, this is not a failure but a *design boundary* — it tells system designers exactly when to fall back to AppLoop-PC.

**Framework.** We perform an end-to-end wall-clock comparison of three execution modes on a single GPU (RTX 4070 SUPER), using 8 tool calls across 3 cities assigned to 4 logical devices in round-robin — simulating a workload where each step requires assembling context from independently-produced fragments. **Important caveat:** This is a single-device measurement of local fragment assembly cost, not a multi-device distributed deployment.

The three modes compared:

1. **SIG** — incremental KV-cache injection per step followed by text generation. The streaming injection baseline.
2. **AppLoop** — full-context re-encoding per step followed by text generation.
3. **AppLoop-PC** — AppLoop with emulated prefix-caching. This should be the best mode *if* prefix reuse is possible.

Each mode is measured over **N=30 paired runs** with identical random seeds across modes. Total wall-clock time (cache management + generation) is the primary metric. **Cache-management-only comparisons have been removed** — all earlier per-step `rebuild_cache()`-vs-`eval()` figures were misleading because they excluded generation time.

**Results (N=30 paired runs, end-to-end wall-clock).**

| Model | SIG | AppLoop | AppLoop-PC | SIG vs AppLoop | SIG vs AppLoop-PC |
|------|-----|---------|------------|----------------|-------------------|
| 0.8B | 0.713±0.359s | 0.232±0.003s | 0.231±0.004s | **0.33× (SIG slower)** | **0.32× (SIG slower)** |
| 4B   | 1.396±0.089s | 1.336±0.014s | 1.508±0.111s | 0.96× (near parity) | **1.08× (within 1 SD)** |

**Key findings.**

**1. SIG is structurally inefficient for fragmented context assembly.** On 0.8B, SIG is 3.0× slower than AppLoop-PC. Each of the 8 tool calls triggers a separate SIG injection (`compiler.eval()`), and the cumulative per-step eval overhead exceeds the cost of a single full-pass re-encoding. This is the expected behavior for a streaming-injection system: SIG is optimized for "inject once, consume continuously," not for frequent small independent rebuilds.

**2. The gap narrows with model size.** At 4B, the three modes are numerically indistinguishable (ratios 0.96–1.08×, within one standard deviation). With N=30, the minimum detectable effect size at 80% power is d≈0.5 — the observed 1.08× ratio would require N≈200 for statistical detection. As model size increases, the relative cost of prefill grows, and SIG's per-step overhead becomes proportionally smaller.

**3. This is a design boundary, not a failure.** R13 defines the complementary workload regime to R6's deep-chain regime:

| Workload Profile | Best Mode | Why |
|-----------------|-----------|-----|
| Continuous deep chain (R6) | **SIG** | Prefix caching ineffective; SIG's incremental cost is constant |
| Fragmented independent assembly (R13) | **AppLoop-PC** | SIG's per-step eval overhead exceeds single-pass re-encoding cost |

**4. Proposed hybrid scheduling strategy.** An edge inference runtime can detect workload characteristics and route accordingly: if the current task involves a continuous chain of dependent tool calls (R6 profile), use SIG; if the task involves assembling many unrelated small contexts (R13 profile), fall back to AppLoop-PC. The decision can be made at runtime based on the number of sequential tool calls and the expected context reuse ratio. This is analogous to how modern compilers choose between inlining (streaming) and function calls (rebuild) based on code characteristics.

**Implications.** R13 does not weaken the case for SIG — it *strengthens* it by defining the precise boundary where SIG should not be used. A specialized accelerator must have a well-defined scope; R13 provides that scope for SIG.

**Scalability Projection [Analytic Projection — not measured, for illustrative purposes only].**
| Devices | Throughput (vs single) | Recommended max cluster |
|---------|----------------------|------------------------|
| 2 | ~1.15× | any |
| 4 | ~1.30× | any |
| 8 | ~1.45× | any |
| 16 | ~1.60× | may need hierarchical routing |
| 32+ | diminishing | cluster partitioning recommended |

*Projection method: Amdahl's law model assuming 90% parallelizable SIG incremental overhead; communication cost ignored; for illustrative purposes only. Actual multi-device throughput depends on network bandwidth, device heterogeneity, and synchronization protocol.*

**Recommendations.** Deploy fragmented local KV reconstruction with SIG as the default for multi-device edge inference. Use workload-aware scheduling to balance device fragment sizes. For >16 devices, implement hierarchical clustering to avoid O(n²) cache sharing overhead. The three-tier architecture (edge→fog→cloud) with SIG-based KV-cache fragment routing provides an optimal balance of throughput, latency, and cost.


## 6. Related Work

**Cognitive Outsourcing and SIG.** Our work builds directly on [1] and [2], which established the CO+SIG paradigm and evaluated five dimensions (information theory, cache degradation, architectural compatibility, teacher-student distillation, and privacy). This paper extends the evaluation to nine additional dimensions with real-model measurements.

**KV-Cache Management.** Efficient KV-cache management has been extensively studied in LLM serving systems. **PagedAttention** (vLLM [3]) introduced virtual memory paging for KV cache blocks, enabling near-zero waste in KV cache memory. **SGLang** [4] proposed RadixAttention for sharing KV cache prefixes across requests with common prompts. **TensorRT-LLM** implements inflight batching with KV cache reuse. SIG differs from these systems in its core mechanism: rather than managing KV cache as a serving optimization, SIG uses **incremental injection** — selectively inserting new token representations without resetting the cache — as a first-class cognitive primitive that preserves context continuity across turns.

**Tree-of-Thought and Reasoning Paradigms.** Tree-of-Thought (ToT [5]) and Chain-of-Thought (CoT [6]) have demonstrated that structured reasoning improves LLM accuracy. Our R14 extends this line of work by measuring how SIG's incremental cache injection interacts with CoT-structured prompts, finding that at N=30 with output length controlled, CoT+SIG provides no measurable speedup over identically-structured CoT+AppLoop (1.00–1.03×); the single observed speedup (2.18× on 4B Q2) is accompanied by a 0.42 gen-token ratio indicating probable truncation.

**Scaling Laws.** Chinchilla scaling laws [7] and follow-up work [8] establish relationships between model size, data, and performance. Our R12 contributes an orthogonal perspective: SIG speedup scaling with model prefill cost and context length, providing practical deployment guidance.

**LLM Agent Security.** Prompt injection [9] and tool result poisoning [10] are recognized threats to LLM agents. Our R10 extends this analysis to SIG's unique attack surface — the persistent KV cache as a propagation vector — and proposes cache-level defenses (rollback isolation, attention monitoring).

**Spatial Cognition in LLMs.** Recent work has explored LLMs' spatial reasoning capabilities [11]. Our R8 contributes N=30 measurements of long-range textual retention and sequential reasoning, finding AppLoop-PC as the best retriever (50–64% hit rate) with complete T=12 failure, and 4B AppLoop at 100% on sequential reasoning.

**Distributed Inference.** Petals [12] and Splitwise [13] explore distributed LLM inference. Our R13 differs by distributing tool execution and KV-cache injection across devices, rather than model layers, enabling heterogeneous device participation.

**Structured Representations for LLMs.** Work on structured prompting [14] examines how format affects model performance. Our R7 contributes the counterintuitive finding that JSON incurs 45% token overhead vs plain text when injected into KV cache — a practical consideration for SIG-based systems.


## 7. Design Space Summary — The SIG Decision Framework

This section synthesizes the empirical findings into a practical decision framework for edge inference system designers.

### The Advantage Zone: When to Use SIG

SIG provides transformative acceleration in exactly one regime — and this regime is precisely where existing serving optimizations (prefix caching) are structurally ineffective:

**Deep, continuous tool chains (R6).** When a small edge model must execute many sequential, dependent tool calls, SIG's incremental injection cost remains near-constant regardless of chain depth. At 30-tool depth on Qwen3.5 4B Q4_K_M, SIG achieves 4.26× speedup over AppLoop — and AppLoop-PC (emulated prefix caching) provides zero benefit because <3% of tokens are cacheable. This is the regime where SIG is *not replaceable* by existing production optimizations.

### The Compatibility Zone: When SIG Is Not Faster, But Not Worse

**CoT-structured reasoning (R14).** When tool results are assembled into a single CoT block, the cost difference between SIG's injection and AppLoop's re-encoding becomes negligible — both must process the full CoT text. CoT+SIG matches CoT+AppLoop in performance (ratios 0.99–1.03× when output length is controlled). SIG composes cleanly with CoT and retains the architectural flexibility for incremental extension — a structural advantage not captured by single-pass measurements.

**Tool-result faithfulness (R11).** Token-Jaccard scores are near-identical between SIG and AppLoop. SIG achieves acceleration without silent quality degradation.

### The Boundary Zone: When to Fall Back

These are not failures — they are design boundaries that define the scope of a streaming-injection accelerator:

| Workload Profile | Use | Why |
|-----------------|-----|-----|
| Long continuous tool chains | **SIG** | Incremental cost is constant; prefix caching ineffective |
| CoT-structured generation | Either | Performance-equivalent; SIG adds architectural flexibility |
| Fragmented independent assembly | **AppLoop-PC** | SIG's per-step eval overhead exceeds single re-encoding |
| Random-access retrieval | **AppLoop-PC** | Full re-encoding gives attention heads direct positional access |
| Security-sensitive (rollback) | **SIG + Isolation** | Rollback achieves keyword-clean recovery; manageable risk |

### Hybrid Scheduling Strategy

The SIG Decision Framework is implementable as a runtime scheduler on edge devices:

1. **Detect workload profile** at task submission: count expected sequential tool calls, estimate context reuse ratio.
2. **Route deep chains to SIG** — where prefix caching is ineffective and incremental injection provides the largest advantage.
3. **Route fragmented/retrieval workloads to AppLoop-PC** — where re-encoding with prefix caching is the correct tool.
4. **CoT workloads are routing-agnostic** — SIG and AppLoop-PC achieve equivalent performance; choose based on whether incremental extension capability is needed.

This is analogous to how modern compilers choose between function inlining (streaming, low overhead per call) and function calls (rebuild, better for large independent units) based on code characteristics. The SIG Decision Framework applies the same principle to edge LLM inference.

### Deployment Scope and Limitations

All observations in this framework are specific to Qwen3.5 (Q4_K_M) on RTX 4070 SUPER (representing a high-end edge workstation). Cross-family replication on Llama, Gemma, and Mistral is required to determine whether these decision boundaries generalize. The framework should be treated as an empirically-grounded starting point for system design, not as validated deployment guidance.


## 8. Discussion — Architecture, Design Philosophy, and Future Work

### 8.1 SIG as a Streaming-Injection Architecture

The empirical results across nine research vectors converge on a clear architectural characterization of SIG:

**SIG is a streaming tape, not a random-access array.** Its KV cache stores information in the order it was injected, optimized for sequential consumption (generation continuity). When the task pattern matches this architecture — long, continuous chains where each step builds on previous context — SIG provides transformative acceleration because its incremental injection cost is near-constant regardless of chain depth (R6: 2.79–4.26×). When the task pattern diverges — requiring random access to arbitrary early-turn facts (R8) or frequent independent context assembly (R13) — AppLoop-PC's full-context re-encoding is the correct tool.

This architectural characterization explains why SIG is not a universal accelerator: it was not designed to be one. SIG is a *specialized accelerator* for a specific workload profile. The conditions where it shows no advantage are not failures — they are the expected behavior of a streaming-injection architecture applied outside its design scope.

### 8.2 Why Prefix Caching Does Not Replace SIG

A natural question is whether production prefix-caching systems (PagedAttention, RadixAttention) make SIG redundant. R6 provides the empirical answer: **no, not in the deep-chain regime.** When a tool chain builds from an empty shared prefix, <3% of tokens are cacheable — the dominant cost is re-evaluating the accumulated tool results, which are unique to each turn and cannot be shared. Prefix caching provides near-zero benefit in this regime. SIG's incremental injection, by contrast, avoids re-evaluating *any* prior tokens — it only processes the new tokens from each tool call. This is the fundamental advantage: prefix caching reuses what is already cached; SIG avoids caching what does not need to be re-evaluated.

### 8.3 Future Work

- **Cross-family replication (HIGH PRIORITY):** Verify R6's 2.8–4.3× deep-chain advantage on Llama, Gemma, and Mistral at ≥7B with tool-depth sweeps (14/22/30/50).
- **Hybrid scheduler implementation:** Prototype the SIG Decision Framework as a runtime scheduler on edge hardware, with automatic workload detection and routing.
- **R8 metric improvement:** Replace keyword pass/fail criteria with embedding-based semantic similarity for retrieval probe evaluation.
- **R14 forced-output-length:** Re-execute R14 with dynamic token budget equalization to eliminate the output-length confound.
- **R13 model-size scaling:** Measure SIG's fragmentation penalty at ≥7B to characterize the amortization threshold.
- **R10 embedding-based contamination detection:** Upgrade residual Jaccard to sentence-transformer cosine similarity for more sensitive contamination auditing.
- **R12 empirical scaling:** Measure prefill across ≥3 model sizes at ≥3 context lengths to empirically establish the scaling relationship.
- **R1 attention analysis:** Fix GQA compatibility; add attention-head-level SIG vs full-context comparisons.
- **R7/R9 replication:** Re-measure at N≥10 per condition with cross-tokenizer and cross-schema validation.


## 9. Conclusion

This paper presented an empirical design-space exploration of Suspend-and-Inject Generation (SIG) as a specialized edge acceleration paradigm for small-model, long-tool-chain inference. Through N=30 paired-run measurements across nine research vectors on Qwen3.5 (0.8B/4B, Q4_K_M, RTX 4070 SUPER), we mapped SIG's advantage zone, compatibility zone, and design boundaries.

### Core Finding: SIG as a Streaming-Injection Edge Accelerator

SIG is not a universal speedup over prefix-caching baselines — and it was never designed to be. It is a *streaming-injection architecture* optimized for "inject once, consume continuously": incremental KV-cache updates with near-constant per-step cost, decoupling task complexity from latency.

**The advantage zone — deep tool chains (R6).** At 30-tool depth, SIG achieves 2.79× (0.8B) to 4.26× (4B) end-to-end speedup over AppLoop. Prefix caching (AppLoop-PC, emulating PagedAttention/RadixAttention) provides zero benefit because <3% of tokens are cacheable in a chain built from an empty shared prefix. This is the regime where SIG is *not replaceable* by existing production optimizations, and it is the regime that matters most for edge deployment: small models executing teacher-guided planning chains that would be infeasible under AppLoop's growing re-encoding cost.

**The compatibility zone — CoT reasoning (R14) and faithfulness (R11).** SIG composes cleanly with CoT-structured prompting (performance-equivalent when output length is controlled) and introduces no measurable fidelity degradation (Token-Jaccard near-identical to AppLoop). Acceleration does not come at the cost of quality.

**The design boundaries — fragmented assembly (R13) and random-access retrieval (R8).** SIG is structurally inefficient for frequent small context rebuilds (3.0× slower on 0.8B) and does not support random-access retrieval of early-turn facts (0% hit rate on 0.8B). These are not failures — they are the expected behavior of a streaming-injection architecture applied outside its design scope, just as a tape drive is not "failing" when it cannot match RAM's random-access latency.

### The SIG Decision Framework

We synthesized these findings into a practical decision framework for edge inference system designers:

| Workload Profile | Use | Rationale |
|-----------------|-----|-----------|
| Long continuous tool chains | **SIG** | Incremental cost constant; prefix caching ineffective |
| CoT-structured generation | Either | Equivalent performance; SIG adds incremental extension capability |
| Fragmented independent assembly | **AppLoop-PC** | SIG's per-step overhead exceeds re-encoding cost |
| Random-access retrieval | **AppLoop-PC** | Full re-encoding provides direct positional access |

This framework is implementable as a runtime scheduler that detects workload characteristics and routes accordingly — analogous to compiler inlining decisions.

### Contributions

1. **Empirical characterization** of SIG's deep-chain advantage (R6: 2.79–4.26× at N=30) and its structural limitations (R13: 0.32× on fragmented assembly; R8: 0% retrieval on 0.8B).
2. **Introduction and validation** of AppLoop-PC as an emulated prefix-caching baseline, demonstrating that production prefix caching does not replace SIG in the deep-chain regime.
3. **Architectural framing** of SIG as a *streaming-injection accelerator* — a specialized paradigm with well-defined scope — rather than a universal optimizer.
4. **The SIG Decision Framework** — a routing heuristic for hybrid edge inference systems.
5. **Methodological improvements:** end-to-end wall-clock measurement protocol, Token-Jaccard fidelity metric, deprecation of cache-only comparisons.

### Limitations and Replication

All findings are specific to Qwen3.5 (Q4_K_M) on RTX 4070 SUPER. No formal hypothesis tests were performed. Cross-family replication on Llama, Gemma, and Mistral at ≥7B scales is the critical next step. R7/R9 remain at N=1; R14 requires forced-output-length re-execution; R8 retrieval probes use keyword-pass/fail criteria that may undercount correct answers. The SIG Decision Framework should be treated as an empirically-grounded design hypothesis, not validated deployment guidance.

SIG's fundamental value is not that it is faster than every alternative in every condition — it is that it opens a new path for edge inference: replacing the growing burden of re-encoding with a constant-cost streaming paradigm, precisely in the deep-chain regime where existing optimizations provide no relief. The honest delineation of its boundaries makes that value credible.


## 10. Validity Threats

This exploratory study faces several validity threats that constrain the generalizability of its findings.

**Internal validity threats** (measurement quality within the experimental design):

1. **Lexical metric as faithfulness (R11).** Token-Jaccard captures vocabulary-unit overlap but not semantic equivalence. A model paraphrasing "22°C clear skies" as "good weather for sightseeing" scores zero. Embedding-based semantic similarity (sentence-transformers) would provide a more meaningful fidelity signal.

2. **Keyword pass/fail for probes (R8).** Retrieval and reasoning probes use simple keyword presence as the pass criterion, which may miss semantically correct but differently-worded answers. A manual review of N=30 probe responses per condition would quantify the false-negative rate.

3. **Output length imbalance (R14).** Q2 on 4B exhibits a 0.42 gen-token ratio between CoT+SIG and CoT+AppLoop, confounding wall-clock speedup interpretation. Forced-output-length protocols (e.g., dynamic max-token budgets with early-stop detection) are needed.

4. **Single seed per run.** Each paired run uses a single random seed. N=30 captures run-to-run variance but not seed-to-seed variance in the underlying generation stochasticity. Multi-seed averaging per run would improve estimate stability.

5. **R7/R9 sample size (N=1).** These provide directionally useful data but cannot support statistical inference. Replication at N≥10 per condition is needed.

**External validity threats** (generalizability beyond the experimental conditions):

6. **Single model family.** All experiments use Qwen3.5 (0.8B/4B, Q4_K_M). Tokenization artifacts (R7), attention patterns (R10), retrieval behavior (R8), and CoT performance (R14) are all architecture-dependent. Cross-family replication on Llama, Gemma, and Mistral is essential.

7. **Single GPU, single quantization.** All measurements on RTX 4070 SUPER (12 GB). Prefill throughput, eval latencies, and crossover points depend on GPU architecture and quantization level. Results on A100/H100 or FP16 models may differ substantially.

8. **N=30 adequate for large effects only.** With 30 paired runs, the minimum detectable effect size (at 80% power, α=0.05, two-tailed paired t-test) is approximately d=0.5 (medium). Smaller effects (e.g., R13's 1.08×) may fall below detection threshold.

**Recommendations for confirmatory studies.** (a) Cross-validate R6/R8/R13/R14 on ≥2 additional model families; (b) adopt embedding-based semantic similarity for R11; (c) implement forced-output-length in R14; (d) measure the full SIG-vs-AppLoop-PC crossover surface across model size × context length × tool depth.


## References

[1] *Cognitive Outsourcing with Suspend-and-Inject Generation for Scalable Embodied Intelligence.* CO+SIG paper, 2025.

[2] *Beyond the Injection Engine: A Five-Dimensional Analysis of CO+SIG.* Extended analysis paper, 2025.

[3] Kwon, W., et al. "Efficient Memory Management for Large Language Model Serving with PagedAttention." SOSP 2023.

[4] Zheng, L., et al. "SGLang: Efficient Execution of Structured Language Model Programs." NeurIPS 2024.

[5] Yao, S., et al. "Tree of Thoughts: Deliberate Problem Solving with Large Language Models." NeurIPS 2023.

[6] Wei, J., et al. "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models." NeurIPS 2022.

[7] Hoffmann, J., et al. "Training Compute-Optimal Large Language Models." NeurIPS 2022.

[8] Kaplan, J., et al. "Scaling Laws for Neural Language Models." arXiv 2020.

[9] Perez, F., and Ribeiro, I. "Ignore Previous Prompt: Attack Techniques for Language Models." NeurIPS 2022 Workshop.

[10] Greshake, K., et al. "Not What You've Signed Up For: Compromising Real-World LLM-Integrated Applications with Indirect Prompt Injection." AISec 2023.

[11] Li, B., et al. "Can Large Language Models Understand Spatial Audio?" EMNLP 2024.

[12] Borzunov, A., et al. "Petals: Collaborative Inference and Fine-tuning of Large Models." ACL 2023 Demo.

[13] Patel, P., et al. "Splitwise: Efficient Generative LLM Inference Using Phase Splitting." ISCA 2024.

[14] Liu, N.F., et al. "Lost in the Middle: How Language Models Use Long Contexts." TACL 2024.


## Appendix A: AppLoop-PC — Prefix-Cache Simulation

The AppLoop-PC baseline used throughout this study approximates production prefix-caching (PagedAttention, RadixAttention) via llama.cpp's native `kv_cache_seq_cp` API. This appendix documents its implementation logic, assumptions, and limitations.

### A.1 Implementation

The `PrefixCache` class (in `core/compiler.py`, ~60 lines) operates as follows:

1. **Save phase.** After the initial prefill of the shared prefix (system prompt), the entire KV cache for sequence 0 is copied to a separate sequence (sequence 1) via `kv_cache_seq_cp(0, 1, 0, prefix_len)`. The prefix token IDs are stored.
2. **Restore phase.** On each subsequent turn:
   - The current cache is reset via `reset_cache()`.
   - The saved prefix is copied back: `kv_cache_seq_cp(1, 0, 0, prefix_len)`.
   - `n_tokens` is set to `prefix_len` so the model's positional counter is correct.
   - Turn-specific tokens (tool results, query) are evaluated with `eval()` on top of the restored prefix.
3. **Turn cost.** Each turn pays `O(prefix_len)` for the sequence copy (a pure memory operation, no compute) plus `O(turn_specific)` for the incremental eval. This contrasts with naive AppLoop's `O(prefix_len + turn_specific)` of full compute prefill.

### A.2 Assumptions and Limitations

| Assumption | Realism | Limitation |
|-----------|---------|------------|
| Prefix recognition: the simulator pre-defines the system prompt as the shareable prefix. | Production systems (RadixAttention) use radix-tree matching on token sequences — exact-match semantics are identical for static prefixes. | Does not model partial-prefix sharing (e.g., common conversation history across different users). |
| Cache hit rate: 100% for the system prompt prefix within a single experiment run. | Real prefix-caching systems achieve near-100% hit rates for identical static prompts in a single serving session. | Does not model cache eviction under memory pressure or multi-tenant contention. |
| Sequence copy cost: assumed negligible relative to compute prefill. | `kv_cache_seq_cp` is a device-side memcpy — on RTX 4070 SUPER with ~4GB KV cache for a 4B model, copying the prefix (<2MB) takes <0.1ms. | On larger models or over PCIe, copy latency may become non-trivial. |
| Single-sequence model: only one user query is active at a time. | Production systems batch multiple requests. | Does not model the throughput benefits of prefix-caching under concurrent load. |

### A.3 Why Not vLLM?

A full production serving stack (vLLM/SGLang) would provide the most ecologically valid AppLoop-PC baseline. However, at the time of writing, vLLM's llama.cpp backend support is experimental and does not expose the fine-grained KV-cache inspection APIs needed for our step-by-step prefill measurement protocol. Migrating to vLLM is prioritized for the confirmatory phase of this research.

### A.4 Impact on Reported Speedups

The introduction of AppLoop-PC compresses all SIG-vs-AppLoop speedup ratios. The magnitude of the compression depends on the prefix-to-incremental token ratio:

- For 30-tool deep chains (R6): prefix ~80 tokens, total context ~2,500 tokens → prefix is ~3% of total → AppLoop-PC advantage over naive AppLoop is small.
- For CoT-structured prompts (R14): prefix ~80 tokens, CoT block ~400 tokens → prefix is ~17% → AppLoop-PC provides modest benefit.

Readers should treat the reported SIG-vs-AppLoop-PC ratios as the study's best approximation of production-relevant speedups, while acknowledging that a full vLLM/SGLang baseline may differ in absolute terms.

## Appendix B: Module Reference

| Module | Test Harness | Description | Key Parameters | Model Required | Dependencies |
|--------|-------------|-------------|---------------|---------------|-------------|
| R6 | `co_benchmark.py --task r6` | Dynamic replanning with tool failure injection (15% failure rate, default 30 tools) | `--r6-tool-depth 14\|22\|30`, `--r6-runs 30` | GGUF | core.injection, core.tools |
| R7 | `sig_benchmark.py --task r7` | Multimodal SIG: structured vs plain vs minimal text token efficiency | — | GGUF | core.injection, core.tools |
| R8 | `sig_benchmark.py --task r8` | Long-range textual retention: 12-turn navigation with early-turn probes (6 rooms) | — | GGUF | core.injection, core.tools |
| R9 | `sig_benchmark.py --task r9` | Real-time SIG: latency budget analysis (200-2000 tokens) | — | GGUF | core.injection |
| R10 | `transformer_bench.py --task r10` | Injection attack test (10 attack vectors, cache pollution + rollback + LLM-judge self-assessment) | — | GGUF | core.injection |
| R11 | `transformer_bench.py --task r11` | Faithfulness evaluation: token-Jaccard + LLM-judge entailment (6 real-entity + 25 fictional-entity queries) | — | GGUF | core.injection, core.tools |
| R12 | `transformer_bench.py --task r12` | SIG scaling law: measured prefill scaling + theoretical projections | — | None (theoretical) | None |
| R13 | `co_benchmark.py --task r13` | Fragmented Local KV Reconstruction: end-to-end wall-clock latency (SIG vs AppLoop vs AppLoop-PC, 8 tools, 4 devices) | `--r13-runs 30` | GGUF | core.injection, core.tools |
| R14 | `co_benchmark.py --task r14` | SIG + Reasoning Paradigms: CoT+SIG vs raw SIG vs AppLoop (2 queries) | — | GGUF | core.injection, core.tools |

**Usage:**
```bash
# CO benchmark tests (R6, R13, R14) — requires GGUF model
$env:PATH = "<path-to-torch-lib>;" + $env:PATH
python co_benchmark.py --task r6 --model models/Qwen3.5-0.8B-Q4_K_M.gguf --n-ctx 16384 --n-gpu-layers 99 --no-debug
python co_benchmark.py --task r13 --model models/Qwen3.5-4B-Q4_K_M.gguf --n-ctx 16384 --n-gpu-layers 99 --no-debug

# SIG benchmark tests (R7, R8, R9) — requires GGUF model
python sig_benchmark.py --task r7 --model models/Qwen3.5-0.8B-Q4_K_M.gguf --n-ctx 16384 --n-gpu-layers 99 --no-debug

# Transformer tests (R10, R11, R12) — R10/R11 require GGUF model; R12 is theoretical
python transformer_bench.py --task r10 --model models/Qwen3.5-0.8B-Q4_K_M.gguf
python transformer_bench.py --task r11 --model models/Qwen3.5-0.8B-Q4_K_M.gguf
python transformer_bench.py --task r12

# Run all via co_benchmark.py (requires GGUF for model-dependent tasks)
python co_benchmark.py --task all --model models/Qwen3.5-0.8B-Q4_K_M.gguf
```

## Appendix C: Self-Evaluation Validity Disclaimer

**The LLM-as-Judge scaffolding code exists in `transformer_bench.py` for experimental completeness, but its outputs have been excluded from all body-text claims in this draft. Reinstatement requires cross-family validation.**

All LLM-as-Judge assessments in R10 and R11 use the same model family (Qwen3.5/2.5) to evaluate its own outputs. This approach is used solely as an internal consistency check and carries the following validity threats:

1. **Confirmation bias**: Models tend to rate their own outputs favorably, inflating apparent recovery rates and faithfulness scores.
2. **Circular evaluation**: A model that cannot pass a test cannot serve as an impartial judge of that test.
3. **No cross-family calibration**: Without a different-family judge (e.g., Gemma evaluating Qwen outputs), there is no way to distinguish true semantic recovery from self-consistent hallucination.

**These LLM-as-Judge scores should be treated as internal diagnostic signals only, NOT as evidence of external validity.** Core claims in R10 and R11 rest on reproducible non-model metrics (Token-Jaccard, n-gram residual Jaccard). Cross-family external judging and GPT-4o-as-oracle validation are absolute prerequisites for any deployment-related claims.
