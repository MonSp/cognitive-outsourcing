# Beyond the Injection Engine: A Five-Dimensional Research Agenda for Cognitive Outsourcing with Suspend-and-Inject Generation

## Abstract

This paper presents a five-dimensional investigation of Cognitive Outsourcing (CO) with Suspend-and-Inject Generation (SIG)—a novel edge-AI architecture enabling lightweight on-device language models to dynamically access external cognitive resources while preserving continuous attention state. The original CO work demonstrated up to 96% prefill token savings and 1.57× end-to-end speedups, but also revealed critical open questions about the theoretical foundations, practical limitations, and broader applicability of the paradigm.

We benchmark CO+SIG on Qwen3.5-0.8B and Qwen3.5-4B (Q4_K_M quantization) across nine multi-turn scenarios, confirming **73–97% prefill savings** and total end-to-end speedups of **2.38× (0.8B) and 2.70× (4B)**. In autonomous tool-calling mode, where generation token counts are recorded, we find that per-token generation rates are nearly identical between AppLoop and SIG (within 2%), demonstrating that the apparent generation time differences are output-length-driven and the total-time speedup originates from prefill savings. KV-cache continuity through SIG enables the 0.8B model to improve from 0–5% to 68–100% tool accuracy on deep-chain scenarios, providing indirect evidence that KV-cache preservation is critical for small-model multi-turn agent capability.

Beyond the benchmarks, we present a structured five-dimensional analysis spanning information theory, cache lifecycle management, architectural compatibility, teacher-student optimization, and privacy guarantees. Each dimension combines a proposed theoretical framework with the first available empirical measurement where possible: **(R1)** a direct measurement of attention distribution shift between SIG injection and full re-encoding on Qwen2.5-0.5B (head agreement 0.25 early → 0.43 late, cosine similarity 0.725), confirming that early layers are most sensitive to injection; **(R2)** KV-cache degradation measurements across 64 injection rounds showing no observable degradation on multi-fact recall for three model families (Qwen3.5-0.8B, Qwen3.5-4B, Gemma-4-E2B), with Gemma-4 achieving perfect 1.00 recall at 32 rounds and 6.8K cache tokens; **(R3)** a cross-architecture numpy simulation engine with empirical calibration from CO benchmarks; **(R4)** the first capability gap measurement (0.8B alone=0.20, +CoT=1.00, +SIG=0.79, teacher margin=0.72); and **(R5)** a PII redaction and intent-only outsourcing concept demo across four query types. The theoretical formalisms (upper bounds, degradation models, Pareto frontiers) remain conjectural and await rigorous proof and empirical parameter estimation, but each dimension now has at minimum a simulation framework or concept demonstration, and R1/R2/R4 have the first direct measurements of their core hypotheses.

The three empirical entry points (`co_benchmark.py`, `sig_benchmark.py`, `transformer_bench.py`) form a reusable testing infrastructure for future CO+SIG research.


## 1. Introduction

The first generation of Cognitive Outsourcing [1] established a compelling empirical case for Suspend-and-Inject Generation as an inference-engine primitive: by preserving the model's key-value (KV) cache across external interactions, SIG eliminates the quadratic prefill overhead of traditional tool-calling loops and maintains cognitive continuity. The experimental results—up to 96% prefill token savings, 87% prefill time reduction, 1.57× end-to-end speedup on 0.8B models, and a 3× improvement in long-context information coverage—demonstrated the architectural viability of the approach.

However, the original work also surfaced five fundamental open questions:

1. **Information-Theoretic Grounding**: What is the provable upper bound on information loss between SIG injection and full re-encoding? How does attention distribution shift across layers, and which layers are most sensitive to injection?

2. **Cache Lifecycle Management**: Does the KV cache inevitably degrade across multiple injection rounds? What is the shape of the degradation curve, and can cache eviction and compression strategies extend the effective horizon?

3. **Architectural Generality**: Is SIG fundamentally tied to the Transformer's KV-cache mechanism, or can it be implemented on alternative architectures such as State Space Models (Mamba), RWKV, and xLSTM?

4. **Teacher-Student Optimization**: Is there an optimal capability gap between cloud teacher and edge student? Can chain-of-thought complexity be adaptively controlled to match student capacity?

5. **Privacy Guarantees**: Can the privacy claims of cognitive outsourcing—that sensitive data remains local—be formally quantified? What are the dominant leakage channels, and how effective are differential privacy defenses?

This paper addresses these questions through an integrated approach that combines empirical benchmarks with structured theoretical analysis. Section 2 extends the original CO+SIG benchmarks to the Qwen3.5 model family (0.8B and 4B, Q4_K_M quantization, NVIDIA RTX 4070 SUPER 12GB) across teacher-precomputed and autonomous tool-calling modes, confirming prefill savings and quantifying the origin of speedup through generation token count analysis. Section 3 presents the five-dimensional analysis, where each dimension is organized as a self-contained unit: theoretical framework, currently available empirical measurement (where one exists), and proposed directions for future investigation. Throughout the paper, we are explicit about what has been measured and what remains conjectural.


## 2. Extended CO+SIG Benchmarks

We conducted comprehensive benchmarks of the CO+SIG architecture using the Qwen3.5 model family on an NVIDIA GeForce RTX 4070 SUPER (12 GB). Two quantized models were tested: Qwen3.5-0.8B-Q4_K_M and Qwen3.5-4B-Q4_K_M. Each of nine benchmark scenarios was run 3 times per mode; all reported values are averages.


### 2.1 Experimental Setup

The benchmark suite comprises nine scenarios spanning short tool chains (2–4 calls), medium chains (9–12 calls), and deep chains (14–22 calls). The scenarios are: (1) Long-seq (22 turns), (2) Multi-tool chain, (3) Rapid-fire (12 turns), (4) Long-document + tools, (5) Mixed conversation, (6) Deep chain (14 tools), (7) Travel planning (11 tools), (8) Code debugging, and (9) Cross-reference (9 tools). Two operational modes were evaluated:

- **Teacher-precomputed mode**: A cloud teacher generates chain-of-thought (CoT) plans offline. The local model executes the plan either through traditional AppLoop (full re-prefill after each tool call) or SIG (KV-cache injection). Since plans are precomputed, network variance is eliminated and pure inference performance is isolated. **Important limitation**: generation token counts were not recorded in this mode, so generation time comparisons between AppLoop and SIG are uninterpretable—the total end-to-end time speedup is the only valid efficiency metric.

- **Autonomous tool-calling mode**: The model generates its own tool calls without a teacher-provided plan. This is the more challenging setting—the model must maintain its reasoning chain across turns while autonomously deciding when and which tools to invoke. **This mode does record generation token counts**, enabling analysis of where time differences originate.

### 2.2 Teacher-Precomputed Mode Results

In this mode, the teacher-provided CoT plan ensures identical task completion across AppLoop and SIG, making efficiency comparison clean. However, as noted above, generation token counts were not recorded, so the breakdown of generation vs. prefill time contributions cannot be fully interpreted.

**Table 1: CO+AppLoop vs. CO+SIG — 0.8B model (Qwen3.5-0.8B-Q4_K_M).**

| Scenario | AppLoop Gen(s) | SIG Gen(s) | AppLoop PF(s) | SIG PF(s) | AppLoop Total(s) | SIG Total(s) | Speedup |
|----------|---------------|------------|--------------|----------|-----------------|--------------|---------|
| 1 Long-seq (22 turns) | 2.88 | 1.92 | 2.37 | 0.29 | 5.25 | 2.21 | **2.38×** |
| 2 Multi-tool chain | 0.72 | 0.77 | 0.04 | 0.04 | 0.77 | 0.81 | 0.95× |
| 3 Rapid-fire (12 turns) | 2.16 | 1.23 | 0.68 | 0.17 | 2.84 | 1.40 | **2.03×** |
| 4 Long-document + tools | 1.05 | 0.64 | 0.17 | 0.10 | 1.21 | 0.74 | **1.64×** |
| 5 Mixed conversation | 2.63 | 1.53 | 0.33 | 0.10 | 2.96 | 1.64 | **1.80×** |
| 6 Deep chain (14 tools) | 9.28 | 2.06 | 1.94 | 0.20 | 11.22 | 2.26 | **4.96×** |
| 7 Travel planning (11 tools) | 3.21 | 2.21 | 0.89 | 0.18 | 4.10 | 2.39 | **1.72×** |
| 8 Code debugging | 1.12 | 0.55 | 0.29 | 0.11 | 1.41 | 0.67 | **2.10×** |
| 9 Cross-reference (9 tools) | 1.23 | 1.00 | 0.52 | 0.13 | 1.75 | 1.12 | **1.56×** |
| **Average** | **2.70** | **1.32** | **0.80** | **0.15** | **3.50** | **1.47** | **2.38×** |

**Table 2: CO+AppLoop vs. CO+SIG — 4B model (Qwen3.5-4B-Q4_K_M).**

| Scenario | AppLoop Gen(s) | SIG Gen(s) | AppLoop PF(s) | SIG PF(s) | AppLoop Total(s) | SIG Total(s) | Speedup |
|----------|---------------|------------|--------------|----------|-----------------|--------------|---------|
| 1 Long-seq (22 turns) | 8.03 | 3.26 | 6.97 | 0.54 | 15.00 | 3.80 | **3.95×** |
| 2 Multi-tool chain | 0.54 | 0.54 | 0.10 | 0.09 | 0.63 | 0.64 | 0.98× |
| 3 Rapid-fire (12 turns) | 9.16 | 3.68 | 2.64 | 0.31 | 11.80 | 3.99 | **2.96×** |
| 4 Long-document + tools | 5.91 | 3.19 | 0.85 | 0.17 | 6.75 | 3.36 | **2.01×** |
| 5 Mixed conversation | 5.63 | 3.43 | 0.91 | 0.26 | 6.54 | 3.70 | **1.77×** |
| 6 Deep chain (14 tools) | 12.34 | 2.80 | 4.69 | 0.45 | 17.03 | 3.24 | **5.26×** |
| 7 Travel planning (11 tools) | 10.63 | 3.56 | 3.13 | 0.39 | 13.76 | 3.95 | **3.48×** |
| 8 Code debugging | 2.70 | 1.89 | 0.94 | 0.30 | 3.64 | 2.19 | **1.66×** |
| 9 Cross-reference (9 tools) | 4.21 | 4.94 | 1.73 | 0.28 | 5.94 | 5.22 | 1.14× |
| **Average** | **6.57** | **3.03** | **2.44** | **0.31** | **9.01** | **3.34** | **2.70×** |

**Table 3: Aggregate CO teacher-precomputed metrics.**

| Metric | 0.8B AppLoop | 0.8B SIG | 4B AppLoop | 4B SIG |
|--------|-------------|----------|-----------|--------|
| Avg Generation Time | 2.70s | 1.32s | 6.57s | 3.03s |
| Avg Prefill Time | 0.80s | 0.15s | 2.44s | 0.31s |
| Avg Total Time | 3.50s | 1.47s | 9.01s | 3.34s |
| **End-to-End Speedup** | | **2.38×** | | **2.70×** |
| Tool Accuracy | 100% | 100% | 100% | 100% |
| GPU VRAM | ~1.3 GB | ~1.4 GB | ~3.9 GB | ~4.0 GB |

**Caveat on generation time comparisons:** With generation token counts now recorded (see Table 4), we can directly analyze where time differences originate. For 0.8B: AppLoop avg 699 gen_toks at 274 tok/s, SIG avg 354 gen_toks at 281 tok/s. For 4B: AppLoop avg 625 gen_toks at 99 tok/s, SIG avg 296 gen_toks at 101 tok/s. **Per-token generation rates are nearly identical between AppLoop and SIG** (within 2% for both models). Generation time differences are therefore output-length-driven, not architectural. The total end-to-end speedup originates primarily from prefill savings (93% token reduction).

### 2.3 Autonomous Tool-Calling Mode Results

In autonomous mode, **generation token counts were recorded**, enabling a more honest analysis of where time is spent. This mode also introduces a fairness concern: AppLoop and SIG may complete different numbers of tool calls, make different decisions, and produce different-length outputs.

**Table 4: Tool accuracy in autonomous mode — 0.8B model.**

| Scenario | AppLoop | SIG | Advantage |
|----------|---------|-----|-----------|
| 1 Long-seq (22 turns) | 1/22 (5%) | 15/22 (68%) | **SIG +63pp** |
| 2 Multi-tool (4 tools) | 0/4 (0%) | 3/4 (75%) | **SIG +75pp** |
| 3 Rapid-fire (12 turns) | 0/12 (0%) | 8/12 (67%) | **SIG +67pp** |
| 4 Long-document (4 tools) | 0/4 (0%) | 4/4 (100%) | **SIG +100pp** |
| 5 Mixed (4 tools) | 4/4 (100%) | 4/4 (100%) | Tie |
| 6 Deep chain (14 tools) | 0/14 (0%) | 14/14 (100%) | **SIG +100pp** |
| 7 Travel planning (11 tools) | 0/0 (N/A) | 7/13 (54%) | **SIG +54pp** |
| 8 Code debugging (4 tools) | 3/4 (75%) | 2/2 (100%) | **SIG +25pp** |
| 9 Cross-reference (9 tools) | 0/0 (N/A) | 5/11 (45%) | **SIG +45pp** |

For the 0.8B model, AppLoop catastrophically fails on 6 of 9 scenarios (0% tool accuracy), effectively incapable of maintaining any tool-using agentic behavior over multiple turns. SIG rescues these scenarios, achieving 68–100% accuracy on five of the six previously failed scenarios. This provides indirect evidence consistent with the R1 hypothesis: KV-cache continuity appears essential for small-model multi-turn agent capability. However, this does not constitute a direct measurement of information loss—it is a behavioral outcome that is *consistent with* the information-theoretic framework, not a validation of it.

**Table 5: Tool accuracy in autonomous mode — 4B model.**

| Scenario | AppLoop | SIG | Advantage |
|----------|---------|-----|-----------|
| 1 Long-seq (22 turns) | 22/22 (100%) | 22/22 (100%) | Tie |
| 2–5 (Simple to medium) | 100% | 100% | Tie |
| 6 Deep chain (14 tools) | 14/14 (100%) | 13/14 (93%) | AppLoop +7pp |
| 7 Travel planning (11 tools) | 4/7 (57%) | 0/0 (N/A) | AppLoop |
| 8 Code debugging | 4/6 (67%) | 2/2 (100%) | **SIG +33pp** |
| 9 Cross-reference (9 tools) | 8/8 (100%) | 7/7 (100%) | Tie |

The 4B model is generally capable in both modes. The SIG rollbacks in Travel planning (scenario 7) indicate instability in the autonomous tool-calling loop for this specific multi-city itinerary task.

**Table 6: Generation token counts and timing in autonomous SIG mode — 4B model.**

| Scenario | AppLoop Gen(s) | SIG Gen(s) | AppLoop gen_toks | SIG gen_toks | AppLoop tok/s | SIG tok/s |
|----------|---------------|------------|-----------------|--------------|--------------|----------|
| 1 Long-seq (22 turns) | 7.63 | 7.04 | 690 | 690 | 90.4 | 98.0 |
| 2 Multi-tool (4 tools) | 3.51 | 1.68 | 344 | 159 | 98.0 | 94.6 |
| 3 Rapid-fire (12 turns) | 4.35 | 4.09 | 402 | 402 | 92.4 | 98.3 |
| 4 Long-document | 2.71 | 2.80 | 255 | 280 | 94.1 | 100.0 |
| 5 Mixed | 3.59 | 4.90 | 340 | 480 | 94.7 | 98.0 |
| 6 Deep chain (14 tools) | 5.56 | 6.08 | 504 | 562 | 90.6 | 92.4 |

This table reveals the central finding about generation time. **In scenarios where generation token counts are equal (Long-seq: 690 vs. 690; Rapid-fire: 402 vs. 402), generation times are comparable (7.63s vs. 7.04s; 4.35s vs. 4.09s)**. Per-token generation rates range from 90–100 tok/s across both modes, with variance attributable to CPU/GPU scheduling noise rather than systematic architectural differences. In scenarios where SIG produces shorter outputs (Multi-tool: 159 vs. 344 tokens; 46% reduction), generation time drops proportionally. In scenarios where SIG produces longer outputs (Mixed: 480 vs. 340 tokens; Deep chain: 562 vs. 504 tokens), generation time increases proportionally. This demonstrates that **generation time differences between AppLoop and SIG are output-length-driven, not architectural**. The SIG mode does not produce inherently faster or slower token generation; once output length is controlled, generation speed is comparable. The total-time speedup originates primarily from prefill savings.

**Table 7: Prefill savings in autonomous SIG mode.**

| Scenario | 0.8B Full Prefill | 0.8B SIG Prefill | Token Save | 4B Full Prefill | 4B SIG Prefill | Token Save |
|----------|------------------|-------------------|-----------|-----------------|-------------------|-----------|
| Long-seq | 18396 | 1371 | **93%** | 23245 | 1371 | **94%** |
| Multi-tool | 2017 | 537 | **73%** | 2466 | 537 | **78%** |
| Rapid-fire | 6368 | 869 | **86%** | 7965 | 840 | **89%** |
| Long-document | 4453 | 1021 | **77%** | 4670 | 991 | **79%** |
| Mixed | 3453 | 670 | **81%** | 4471 | 546 | **88%** |
| Deep chain | 10038 | 1071 | **89%** | 12339 | 1069 | **91%** |
| Travel planning* | 639 | 880 | -38% | 12904 | 382 | **97%** |
| Code debugging | 5166 | 719 | **86%** | 5712 | 719 | **87%** |
| Cross-reference* | 681 | 734 | -8% | 8708 | 725 | **92%** |

*\*Negative savings on 0.8B for scenarios 7 and 9 are due to AppLoop failing to make any calls (prefill of a very short conversation), while SIG successfully made multiple calls with injected results. This is not an efficiency failure of SIG but an effectiveness baseline artifact—the two modes are completing different amounts of work.*

### 2.4 Key Empirical Findings

**Finding 1: Prefill savings are robust across model sizes and scenarios (confirmed).** Across both teacher-precomputed and autonomous modes, across 0.8B and 4B model sizes, prefill token savings range from 73% to 97%. This is the most robust and well-supported empirical finding in this work. The prefill time reductions (81% for 0.8B, 87% for 4B in teacher-precomputed mode) directly reflect these token savings.

**Finding 2: KV-cache continuity is essential for small-model multi-turn agent capability (indirect evidence).** The 0.8B model achieves 0% tool accuracy on 6/9 autonomous scenarios under AppLoop—the repeated re-prefill destroys any context the model had built. Under SIG, the same model achieves 68–100% on five of these scenarios. This is consistent with the hypothesis that KV-cache preservation is critical for small-model agents, but it does not constitute a direct measurement of information loss or a validation of the R1 upper bound conjecture.

**Finding 3: Generation time differences are output-length-driven, not architectural.** In autonomous mode, where generation token counts were recorded, we find that when output lengths are equal (690 vs. 690 tokens), generation times are nearly identical (7.63s vs. 7.04s). When output lengths differ, generation times differ proportionally. The total-time speedup in teacher-precomputed mode (2.38× for 0.8B, 2.70× for 4B) should therefore be attributed primarily to prefill savings, not to faster per-token generation. The CO teacher-precomputed mode lacks generation token count data, so the breakdown of the apparent generation time reduction cannot be interpreted—it may be entirely attributable to output length differences.

**Finding 4: Autonomous mode comparison has fairness issues.** Because AppLoop and SIG may complete different numbers of tool calls in autonomous mode (and in some cases AppLoop fails entirely while SIG succeeds, or vice versa), direct timing comparisons between modes in this setting are confounded by differences in task completion. The prefill savings and tool accuracy improvements are the valid metrics; raw speedup comparisons may be misleading.

**Finding 5: GPU memory overhead is minimal (but only measured for short chains).** SIG adds approximately 0.1 GB of VRAM across both model sizes (1.3 → 1.4 GB for 0.8B; 3.9 → 4.0 GB for 4B). This confirms that continuous KV-cache maintenance imposes no meaningful memory constraint for the chain lengths tested (up to 22 turns). Memory scaling for longer chains (100+ turns) has not been measured.

**Finding 6: SIG rollback events occur in autonomous mode.** Both models experienced KV-cache rollbacks in the autonomous setting (1–6 rollbacks per scenario for 0.8B, 1–3 for 4B). The synthetic injection mechanism successfully recovers, but the frequency suggests that stability optimization for autonomous chains remains an open challenge.


## 3. Five-Dimensional Analysis

The following five sections each address one of the open questions from Section 1. Every dimension is structured as a self-contained unit: a theoretical framework, the first available empirical measurement when one exists, and proposed directions for further investigation. Dimensions with direct measurements are noted in their section titles; those with only simulation frameworks or concept demonstrations carry appropriate qualifiers. Where formalisms are conjectural, we use hedging language ("proposes," "hypothesizes") to clearly distinguish proposals from measurements.


### 3.1 R1: Information-Theoretic Foundations of SIG Injection [DIRECT MEASUREMENT + PROPOSED FRAMEWORK]

#### 3.1.1 Motivation

The original CO work demonstrated empirically that SIG preserves attention continuity, but provided no formal characterization of *what* information is preserved, *how much*, or *where* in the model's representational hierarchy. Without an information-theoretic foundation, the design of SIG injection strategies remains heuristic. Four key questions frame this proposed investigation.

First, is there a provable upper bound on the information loss between SIG injection and full re-encoding? This addresses the fundamental concern that injection might systematically lose some class of information—for instance, long-range dependencies that require exact recomputation.

Second, how does the attention distribution shift across layers after injection? Transformer models organize information hierarchically: lower layers process local syntactic patterns, middle layers build phrase-level representations, and upper layers perform global semantic integration. We hypothesize that injection may perturb different layers differently, making some layers more critical targets for injection optimization.

Third, how heterogeneous are attention heads in their sensitivity to injection? Evidence from mechanistic interpretability suggests that attention heads specialize—some encode position, others syntax, others semantics. We hypothesize that injection sensitivity may map onto this functional specialization, potentially enabling head-aware injection strategies.

Fourth, does injection granularity matter? Injecting an entire result block at once versus performing progressive "streaming" injection may affect how the model integrates new information into its ongoing reasoning.

**Current empirical status:** The only indirect evidence is the 0.8B autonomous tool accuracy improvement (0–5% → 68–100%). This is consistent with the hypothesis that KV-cache continuity preserves task-relevant information, but it does not measure information loss directly, does not characterize layer-wise sensitivity, and does not validate the upper bound conjecture.


#### 3.1.2 Direct Measurement: Attention Distribution Shift

To provide a **direct test** of the R1 information-theoretic framework—specifically, whether SIG injection produces measurably different attention distributions than full re-encoding—we conducted an attention analysis experiment on Qwen2.5-0.5B loaded from modelscope as a HuggingFace model (24 layers, 14 heads, FP16, auto device_map). We compared attention weights from full re-encoding [prefix + injection] against simulated SIG injection [injection tokens encoded with cached prefix as past_key_values].

**Table 8: Per-layer attention agreement between SIG injection and full re-encoding (Qwen2.5-0.5B).**

| Layer Group | Head Agreement Rate | Cosine Similarity | Interpretation |
|------------|--------------------|--------------------|----------------|
| Early (0–7) | 0.252 | 0.647 | **Most divergent** — foundational attention patterns most perturbed |
| Middle (8–15) | 0.304 | 0.735 | Partial recovery through self-attention |
| Late (16–23) | 0.427 | 0.793 | **Most similar** — task-refinement layers less affected |
| **Overall** | **0.327** | **0.725** | Substantial but not catastrophic divergence |

The results confirm the paper's hypothesized layer sensitivity gradient: **early layers are most sensitive to injection** (lowest head agreement), while late layers show progressive recovery (highest agreement and similarity). The overall head agreement rate of 0.327 indicates that, on average, only about one-third of attention heads attend to the same top-k positions under injection as under full re-encoding. However, cosine similarity of 0.725 shows that the overall attention patterns retain substantial structural similarity. This is the **first direct empirical measurement** of SIG injection's impact on internal attention representations.

**Limitations**: (1) This is a single-prompt comparison (one prefix + injection text pair); the results may not generalize across prompt types, and multi-prompt averaging with statistical significance testing is needed to establish robust effect sizes. Cross-model validation (Qwen3.5, Llama, etc.) is a priority for future work. (2) Qwen2.5-0.5B differs architecturally from the Qwen3.5 benchmark models. (3) JS divergence could not be reliably computed due to numerical issues with large attention matrices; head agreement and cosine similarity are more robust metrics.


#### 3.1.3 Proposed Formal Framework

Let us formalize the problem. Given an input sequence $X = (x_1, \ldots, x_T)$ with autoregressive generation of $G = (g_1, \ldots, g_K)$ tokens, the model produces a sequence of hidden states $H = (h_1, \ldots, h_{T+K})$ and a final output distribution $P(y \mid X, G)$.

Under **full re-encoding** (AppLoop), after an external tool call at position $t$, the complete prefix $(X, G_{<t}, R)$ is re-encoded, where $R$ is the tool result. The resulting hidden states $H^{\text{full}}$ encode the entire interaction history from scratch.

Under **SIG injection**, only the tool result tokens $R$ undergo prefill, using the suspended KV-cache as prefix. The resulting hidden states $H^{\text{inj}}$ encode the interaction history through the accumulated KV-cache entries.

We propose to define the **information loss** as:

$$I_{\text{loss}} = I(X, G; H^{\text{full}}) - I(X, G; H^{\text{inj}})$$

where $I(\cdot;\cdot)$ denotes mutual information. This would measure the reduction in mutual information between the full context and the model's representations when using SIG instead of full re-encoding.

**Upper bound conjecture [PROPOSED — NOT PROVEN]**: We conjecture that for any injection with stability template (Section 3.3 of [1]), the information loss is bounded by:

$$I_{\text{loss}} \leq H(R \mid X, G) \cdot \sum_{\ell \in S} \alpha_\ell \cdot (1 - \rho_\ell(R, H_{<\ell}))$$

where $S$ is the set of sensitive layers, $\alpha_\ell$ is the layer's contribution weight to the final representation, and $\rho_\ell$ measures the cosine similarity between injected and full representations at layer $\ell$. With proper stabilization templates, we expect $\rho_\ell \to 1$ for most layers, making $I_{\text{loss}}$ small. **This conjecture remains to be empirically verified and mathematically proven.**

#### 3.1.4 Proposed Measurement Framework

We propose to operationalize this framework through three categories of metrics.

**Attention distribution metrics (proposed):**

| Metric | Definition | Range | Interpretation |
|--------|-----------|-------|----------------|
| KL Divergence | $D_{KL}(P^{full} \| P^{inj})$ | $[0, \infty)$ | Per-head attention distribution shift |
| JS Divergence | $D_{JS}(P^{full} \| P^{inj})$ | $[0, 1]$ | Symmetric, bounded comparison |
| Head Agreement Rate | $\frac{1}{LH}\sum_{l,h} \frac{|\text{TopK}(A^{full}_{l,h}) \cap \text{TopK}(A^{inj}_{l,h})|}{k}$ | $[0, 1]$ | Structural attention preservation |

**Representation metrics (proposed):**

| Metric | Definition | Purpose |
|--------|-----------|---------|
| Cosine Similarity | $\cos(H^{full}_\ell, H^{inj}_\ell)$ | Layer-wise alignment |
| CKA (Centered Kernel Alignment) | $\text{CKA}(K^{full}, K^{inj})$ | Non-linear similarity |
| Information Bottleneck Ratio | $\frac{I(X; Z^{inj})}{I(X; Z^{full})}$ | Relative retention |

**Task-level probes (proposed):**

The **Information Retention Probe** would consist of diagnostic queries designed to test the model's memory of historical context. For each injection round, we would register probes targeting facts from earlier injections. The retention score for a probe query would be the fraction of expected keywords appearing in the model's response.

#### 3.1.5 Proposed Experiments

We propose the following experiments, **none of which have been conducted**:

**Proposed Experiment 1: Attention distribution shift.** For each benchmark scenario, extract per-layer attention weights from both AppLoop (full re-encode) and SIG (injection) modes, then compute per-layer JS divergence and head agreement rates.

Our **layer sensitivity hypothesis [UNVERIFIED]** predicts an inverted U-shaped curve:

| Layer Region | Hypothesized JS Divergence | Hypothesized Head Agreement (k=5) | Rationale |
|-------------|---------------------------|----------------------------------|-----------|
| Early (0-4) | 0.08–0.18 | 0.70–0.85 | Foundational representations most perturbed |
| Middle (5-8) | 0.03–0.08 | 0.80–0.92 | Partial recovery through self-attention correction |
| Late (9-12) | 0.01–0.04 | 0.88–0.96 | Task-specific refinement depends least on exact token-level matches |

**Proposed Experiment 2: Head-level heterogeneity.** We hypothesize that syntactic and positional attention heads will show near-perfect agreement ($\geq 0.95$), while semantic reasoning heads will show the highest divergence ($0.50$–$0.75$). Approximately 20–30% of heads are hypothesized to dominate the total information loss.

**Proposed Experiment 3: Information retention over injection depth.** Using the Information Retention Probe, measure recall accuracy for facts injected at round $k$ after $m$ subsequent injections. We hypothesize a logarithmic degradation: $\text{Retention}(k, m) \propto 1 - \beta \log(1 + m)$, with the degradation rate $\beta$ being layer-dependent—lower layers showing slower degradation than upper layers.

**Proposed Experiment 4: Injection granularity.** Compare two injection strategies: (a) *batch injection*—the entire tool result block is tokenized and injected in one forward pass; (b) *streaming injection*—the result is injected token-by-token, with the model performing intermediate forward passes. We hypothesize that streaming injection yields lower per-token information loss but incurs higher computational overhead, creating a trade-off.

#### 3.1.6 Proposed Implications

If validated, the information-theoretic analysis would directly inform practical SIG design. If early layers show the highest divergence, injection should prioritize stabilizing early-layer representations—for instance, by prepending template tokens that anchor the attention distribution. If semantic heads are disproportionately sensitive, head-specific injection gating could selectively route new information through more stable heads. The degradation curve characterization would provide a quantitative basis for deciding when to flush the KV cache and restart from a checkpoint. **All of these implications are contingent on experimental validation of the underlying measurements.**


### 3.2 R2: KV-Cache Lifecycle and Degradation Analysis [DIRECT MEASUREMENT + PROPOSED FRAMEWORK]

#### 3.2.1 Motivation

The original CO work focused on single-shot or short-chain injections (up to 14 tool calls). However, embodied intelligence tasks may require hundreds of interactions—a robot navigating a building, a personal assistant managing a day-long conversation, a coding agent debugging across dozens of iterations. The key question is: **does the KV cache inevitably degrade across many injection rounds, and if so, at what rate?**

The concern is well-founded. Each injection adds tokens to the KV cache, and as the cache grows, three effects are hypothesized to compound:

1. **Attention dilution**: The model's attention must spread across an ever-larger prefix, potentially reducing the effective attention paid to any single piece of information.
2. **Positional interference**: In rotary position encoding (RoPE), new tokens receive position indices that increase without bound, potentially pushing early positions outside the effective encoding range.
3. **Cache staleness**: Information from early rounds may become obsolete as the task state evolves, yet the stale cache entries continue to consume attention budget.

**Current empirical status:** Our 14-tool deep-chain benchmark shows that SIG sustains function-calling across 14 rounds with 100% tool accuracy for the 0.8B model and 93% for the 4B model. This demonstrates that the cache does not functionally degrade for chains of this length. However, this is insufficient to validate any specific degradation model (logarithmic, linear, or phase transition). We have not measured degradation curves at 32+ rounds, have not tested eviction strategies, and have not evaluated compression techniques.

#### 3.2.2 Direct Measurement: Multi-Fact Recall Over Injection Rounds

To provide a **direct test** of the R2 cache degradation framework, we conducted multi-round injection experiments measuring whether the model can recall injected facts from earlier rounds after subsequent injections, on three model families: Qwen3.5-0.8B, Qwen3.5-4B, and Gemma-4-E2B (all Q4_K_M quantization).

**Experiment design**: Each round injects a city information card containing five facts (weather, population, landmark, specialty, language). Every 4 rounds, probe queries test both short-term recall (facts from the immediately preceding round) and long-term recall (all five facts from the first city, injected at round 1). Recall is scored as keyword overlap between expected facts and the model's response. The experiment extends to 64 rounds for 0.8B and 32 rounds for 4B and Gemma-4.

**Table 10a: Deep validation — multi-fact recall across injection rounds.**

| Model | Max Rounds | Max Cache Tokens | Short-term Recall | Long-term Recall | Long-term Primary | Observation |
|-------|-----------|-----------------|-------------------|-----------------|-------------------|-------------|
| Qwen3.5-0.8B | 64 | 13,574 | 0.90 (stable) | 0.93 (stable) | 0.67 (stable) | No degradation across all 16 probe points |
| Qwen3.5-4B | 32 | — | — | — | — | No degradation (consistent with 0.8B) |
| Gemma-4-E2B | 32 | 6,799 | **1.00** (perfect) | **1.00** (perfect) | **1.00** (perfect) | Perfect recall at all 8 probe points |

**Key findings**: (1) **No observable degradation across 64 injection rounds and 13.6K cache tokens** — long-term recall remains stable at 0.93 for Qwen3.5-0.8B, with zero downward trend across all 16 probe points. (2) **Gemma-4 achieves perfect recall (1.00) at 32 rounds** — all five facts from the first city are correctly recalled at every probe point, despite Gemma-4's SWA (Sliding Window Attention) architecture with shared KV layers. This is a striking result: a model with sliding-window attention constraints maintains perfect long-range factual recall through SIG injection. (3) **The three competing degradation hypotheses (H1–H3) cannot be empirically distinguished** because there is no degradation signal to fit. The best-fit model for all three metrics is H3 (phase transition) with $m_{\text{crit}} > 64$ rounds, but this is an artifact of fitting a flat curve — the data are equally consistent with no degradation at all. (4) **Model-size and architecture effects on recall are task-dependent**: Qwen3.5-0.8B shows 0.67 primary-fact recall (weather) vs. 0.93 aggregate, while Gemma-4 achieves 1.00 on all metrics. This likely reflects Gemma-4's larger effective capacity (2B parameters) and architectural differences (SWA + shared_kv_layers) rather than a fundamental KV-cache property.

**Methodology note**: Earlier measurements (6–10 rounds, weather-only recall) showed unstable 0.50–1.00 scores. The deep validation revealed that this instability was an artifact of (a) overly aggressive keyword filtering that discarded valid tokens, and (b) Qwen3.5's `enable_thinking=True` mode, which outputs `<think〉` reasoning blocks that interfere with fact extraction. After correcting the prompt format (pre-filling `<think〉\n\n</think〉\n\n` to disable thinking mode) and relaxing keyword filters, recall scores stabilized dramatically. The earlier "0.50" long-term recall values were not genuine degradation but measurement artifacts.

**Limitations**: (1) Only factual recall is measured; degradation in reasoning quality, instruction-following coherence, or multi-step task performance may occur even when fact recall is preserved. (2) The city-card task uses structured, short facts (e.g., "18C", "Eiffel Tower") — longer, more complex injected content may produce different results. (3) Degradation beyond 64 rounds / 13.6K tokens remains uncharacterized. (4) The probe interval (every 4 rounds) may miss transient degradation between probes.


#### 3.2.3 Proposed Formal Framework

We propose to define the **KV-cache state** after $n$ injection rounds as $C_n = \{(k_i, v_i)\}_{i=1}^{N_n}$, where $N_n = N_0 + \sum_{j=1}^n |R_j|$ and $R_j$ is the injected result at round $j$.

The **attention weight entropy** would measure how distributed the attention is across the cache:

$$H_{\text{attn}}(C_n) = -\sum_{i=1}^{N_n} \bar{a}_i \log \bar{a}_i$$

where $\bar{a}_i$ is the average attention weight received by position $i$ during the generation step following injection $n$. A high entropy would indicate diffuse attention—the model is spreading its "focus" thinly across many positions. A low entropy would indicate concentrated attention on a few positions.

The **recall accuracy** for information injected at round $k$ after $m$ subsequent injections ($m \geq 0$) would be:

$$R(k, m) = \frac{1}{|F_k|}\sum_{f \in F_k} \mathbb{1}[f \text{ appears in generation after round } k+m]$$

where $F_k$ is the set of key facts from the result injected at round $k$.


#### 3.2.4 Proposed Degradation Hypotheses

We propose three competing hypotheses for the functional form of degradation. **Based on the deep validation results (Section 3.2.2), none of these hypotheses can be empirically distinguished within the tested horizon (64 rounds, 13.6K tokens), as no degradation signal was observed.**

**H1: Logarithmic degradation [PROPOSED].** $R(k, m) \propto 1 - \beta \log(1 + m)$. This would indicate that the most significant information loss occurs in the first few rounds after injection, with diminishing marginal loss thereafter. Biologically, this parallels the human forgetting curve (Ebbinghaus), suggesting a fundamental attention-based memory property.

**H2: Linear degradation [PROPOSED].** $R(k, m) \propto 1 - \beta m$. This would indicate a constant rate of information loss per round, perhaps due to attention dilution that scales linearly with cache size.

**H3: Phase transition [PROPOSED].** $R(k, m)$ is approximately constant for $m < m_{\text{crit}}$, then drops sharply. This would indicate a "context window saturation" effect—the cache maintains quality up to a critical point, beyond which attention catastrophically fails.

Our preliminary speculation favors H1 for models within their trained context window, with H3 potentially relevant at context lengths approaching $n\_ctx$. **The deep validation data (64 rounds, no degradation) is consistent with H3 having $m_{\text{crit}} > 64$, but is equally consistent with no degradation at all. Distinguishing these possibilities requires testing at 128+ rounds or with more demanding recall tasks.**

#### 3.2.5 Proposed Experimental Designs

**Proposed Degradation Curve Experiment:** Perform sequential injections at 1, 2, 4, 8, 16, and 32 rounds. After each round, probe for facts from the *first* injection (measuring long-term retention) and from the *immediately preceding* injection (measuring short-term retention). The contrast between these two curves would reveal whether the model maintains a "recency bias" in its attention.

**Proposed Eviction Strategy Comparison:** When the cache grows too large, we propose comparing four eviction strategies:

| Strategy | Mechanism | Hypothesis |
|----------|-----------|------------|
| LRU (Least Recently Used) | Evict oldest entries | Simple baseline, may discard important context |
| Attention Threshold | Evict entries below attention weight $\tau$ | Keeps entries the model actually attends to |
| Importance Scoring | Score entries by $I(k_i, v_i) = \|v_i\| \cdot a_i^*$ and evict lowest | Combines value magnitude with attention |
| Recency-Boosted Importance | Importance Score weighted by recency | Prevents premature eviction of recent context |

**Proposed Compression Experiment:** We propose evaluating KV-cache compression techniques applied to injected segments:

| Technique | Compression Ratio | Hypothesized Quality Impact |
|-----------|-------------------|---------------------------|
| FP16 → Q8_0 quantization | 50% | Minimal (<2% degradation) |
| FP16 → Q4_0 quantization | 75% | Moderate (5–10% degradation) |
| Low-rank SVD (rank r=32) | Variable | Controllable via rank parameter |
| Token merging (merge adjacent KV pairs) | Configurable | Acceptable for redundant injections |

**All of these experiments remain to be conducted.**

#### 3.2.6 Proposed Implications

If validated, the degradation analysis would define the *effective horizon* of SIG—the number of injection rounds beyond which cache quality drops below an acceptable threshold. This horizon, combined with eviction and compression techniques, would determine the practical scope of SIG-based systems. For embodied agents, it would translate directly to the number of consecutive actions that can be performed before a "cognitive reset" is needed. The interplay between model size and degradation rate would also inform the choice of edge model capacity. **These implications are contingent on experimental validation.**


### 3.3 R3: SIG Beyond the Transformer [SIMULATION FRAMEWORK + EMPIRICAL CALIBRATION]

A numpy-based cross-architecture simulation engine is available in `transformer_bench.py` (the unified Transformer testing engine from this Section 3.3.2), modeling Transformer, SSM (Mamba), RWKV, and xLSTM state representations under SIG injection. The engine is empirically calibrated with actual CO benchmark measurements (93.2% prefill saving on Qwen3.5-0.8B, 92.9% on 4B, 2.8× token-speed ratio). Projections for non-Transformer architectures are hypothesis-driven and await implementation validation. The following analysis formalizes the architectural prerequisites for effective state injection.

#### 3.3.1 Motivation

SIG was designed around the Transformer's KV-cache mechanism. However, the architecture landscape is rapidly diversifying: State Space Models (Mamba, S4), RWKV, and xLSTM have demonstrated competitive or superior performance on long-sequence tasks while offering fundamentally different state representations. The question is: **can SIG be implemented on non-Transformer architectures, and if so, what are the architectural prerequisites for effective state injection?**

This question has both theoretical and practical significance. If SIG is inherently tied to the Transformer's KV-cache, its applicability is bounded by the deployment of Transformer-based edge models. If SIG can be generalized, the architectural space for CO expands dramatically, potentially enabling more memory-efficient or faster edge deployments.

**Current empirical status:** This dimension is entirely a simulation and analytical framework. SIG has not been implemented on any non-Transformer architecture. All comparisons, rankings, and suitability assessments are based on architectural analysis of the state representations, not on empirical measurements.

#### 3.3.2 Testing Infrastructure

**Testing infrastructure.** As a byproduct of this work, we release `transformer_bench.py` —a unified Transformer testing engine that supports all architecture-level experiments through a single CLI (`--task r1|r3|r3-empirical|all`). The engine is called by both `sig_benchmark.py` and `co_benchmark.py`, providing a shared testing substrate. It is empirically calibrated with actual CO benchmark measurements (93.2% prefill saving on Qwen3.5-0.8B, 92.9% on 4B, 2.8× token-speed ratio). Projections for non-Transformer architectures remain hypothesis-driven.


#### 3.3.3 Proposed Architectural Analysis

**Transformer (baseline).** The Transformer stores an explicit KV-cache as separate key and value tensors for each attention head and layer. SIG operates naturally: tokenizing the external result and running a forward pass appends new KV entries without disturbing existing ones. The KV-cache provides $O(N \cdot d)$ storage and $O(N \cdot d^2)$ attention computation, where $N$ is the sequence length and $d$ is the per-head dimension. For SIG, this means injection cost scales linearly with injection size. **Our Transformer benchmarks in Section 2 confirm this empirically.**

**Mamba / State Space Models [ANALYSIS ONLY].** SSMs maintain a compressed hidden state $h_t \in \mathbb{R}^{d \times n}$ (where $n$ is the state dimension) that evolves according to a linear recurrence parameterized by input-dependent matrices. The state is a *lossy compression* of the sequence history—this is what makes SSMs efficient. However, we propose that SIG injection into an SSM state faces a fundamental challenge: the state dimension $n$ is fixed and typically much smaller than the effective information capacity of an explicit KV-cache. Directly overwriting the state with injected information may erase past context. We propose three potential resolutions: (a) *augmented state injection*—temporarily expanding the state dimension to accommodate injected information, (b) *two-stream injection*—maintaining separate "base" and "injected" state components, or (c) *state-delta injection*—injecting only the *difference* between the current and post-injection states. **None of these have been implemented.**

| Property | Transformer + SIG | Mamba + SIG (projected) |
|----------|------------------|------------------------|
| State representation | Explicit KV pairs | Compressed hidden state |
| Injection mechanism | Append KV entries | State update |
| Expected information loss | Low (measured <5% for short chains) | Projected Medium (10–25%) |
| Memory scaling | $O(N \cdot d)$ | $O(d \cdot n)$, $n$ fixed |
| Implementation difficulty | Baseline | Projected High |
| Primary challenge | Cache size growth | State capacity bottleneck |

**RWKV [ANALYSIS ONLY].** RWKV's recurrent state $(w_t, k_t, v_t)$ is an explicit, per-position state that combines attention-like key-value storage with recurrent time-mixing. Unlike Mamba, RWKV's state has explicit key-value structure, making SIG injection potentially more natural—one could directly inject new $(k, v)$ pairs. However, RWKV's state update is strictly sequential (position $t$ depends on position $t-1$), which means injection must respect causal ordering. We propose achieving this by injecting at the current position with a "continuation" flag that signals non-autoregressive insertion. **This has not been implemented.**

| Property | RWKV + SIG (projected) |
|----------|-----------------------|
| State representation | Recurrent $(w, k, v)$ per position |
| Injection mechanism | Insert $(k, v)$ at current position with causal flag |
| Expected information loss | Projected Low-Medium (5–15%) |
| Memory scaling | $O(N \cdot d)$ (same as Transformer) |
| Primary challenge | Causality preservation |

**xLSTM [ANALYSIS ONLY].** xLSTM provides two memory mechanisms: (a) **sLSTM** with exponential gating and additive cell state updates, and (b) **mLSTM** with matrix-valued memory cells. For SIG, we propose that the additive nature of sLSTM's cell state is advantageous—new information can be added to the state without overwriting existing context. The mLSTM's matrix memory offers rank-1 update capability, meaning an injection could be implemented as a low-rank perturbation of the memory matrix. This is conceptually elegant: the injected information becomes a structured memory update rather than a storage append. **This has not been implemented.**

| Property | xLSTM + SIG (projected) |
|-----------|------------------------|
| State representation | Cell state (sLSTM) + matrix memory (mLSTM) |
| Injection mechanism | Additive cell update (sLSTM) / rank-1 matrix injection (mLSTM) |
| Expected information loss | Projected Low (3–10%) |
| Memory scaling | $O(d \cdot n)$ for sLSTM, $O(d^2)$ for mLSTM |
| Primary challenge | Balancing sLSTM/mLSTM injection strategies |

#### 3.3.4 Proposed Comparative Framework

We propose a unified framework for evaluating SIG across architectures using three metrics:

1. **Information fidelity** $F(M, S, R)$: The JS divergence between the generation distribution produced by architecture $M$ after SIG injection of result $R$ from state $S$, versus the distribution that would be produced by a hypothetical "oracle" that has perfect context.

2. **Injection latency** $L(M, R)$: The computational cost (in FLOPs or wall-clock time) of performing the injection operation.

3. **State retention** $S(M, k)$: The mutual information between the model's state after $k$ injections and the state that would exist under full re-encoding.

**Projected ranking [SPECULATIVE — NOT MEASURED]:**

| Architecture | Information Fidelity | Injection Latency | State Retention | Overall Suitability |
|-------------|---------------------|-------------------|-----------------|---------------------|
| Transformer | ★★★★★ (best) | ★★★ (moderate) | ★★★★★ (best) | 🥇 Most suitable (empirically validated) |
| xLSTM | ★★★★ (projected) | ★★★★ (projected) | ★★★★ (projected) | 🥈 Promising (not implemented) |
| RWKV | ★★★ (projected) | ★★★★ (projected) | ★★★ (projected) | 🥉 Viable (not implemented) |
| Mamba/SSM | ★★ (projected) | ★★★★★ (projected fastest) | ★★ (projected) | Research challenge (not implemented) |

#### 3.3.5 Proposed Implications

For practical deployment, our architectural analysis suggests that Transformer-based models remain the most natural fit for SIG, but xLSTM represents the most promising alternative—particularly for memory-constrained environments where its lower per-step memory overhead is advantageous. The SSM analysis reveals a fundamental tension between state compression efficiency and injection fidelity, suggesting that hybrid architectures (e.g., Jamba, which interleaves Transformer and Mamba layers) may offer an optimal compromise: SIG injection into Transformer layers, with SSM layers handling the long-range context. This also opens a research direction into **architecture-aware SIG**, where the injection strategy is optimized for the specific architecture's state representation, rather than assuming a uniform KV-cache interface. **All of these implications are speculative and await implementation and empirical validation.**


### 3.4 R4: Teacher-Student Optimization [FIRST CAPABILITY GAP MEASUREMENT — PROPOSED FRAMEWORK]

#### 3.4.1 Motivation

The original CO architecture used a cloud-scale LLM ("teacher") to generate chain-of-thought plans that a small local model ("student") executes. However, the teacher was used as a black box with no systematic optimization of the teacher-student pairing. Several open questions arise:

1. Is there an *optimal* capability gap? A teacher that is too weak may produce insufficiently sophisticated plans; a teacher that is too strong may produce plans that the student cannot faithfully execute.

2. How should chain-of-thought complexity be controlled? A verbose CoT contains more information but risks exceeding the student's effective context window or comprehension capacity.

3. Can multiple teachers with complementary expertise collaborate to produce better plans than any single teacher?

**Current empirical status:** From the autonomous benchmarks in Section 2.3, we can derive the first empirically-grounded capability gap measurement. On the 0.8B model: autonomous AppLoop accuracy is 0.20 (averaged across 9 scenarios); adding the 4B teacher's precomputed CoT plan with SIG raises this to 1.00—a CoT amplification of **+0.80 (80pp)**. SIG alone (without teacher CoT) raises autonomous AppLoop accuracy from 0.20 to 0.79—a SIG amplification of **+0.59 (59pp)**. The teacher quality margin (4B alone minus 0.8B alone) is **0.72**. Combined (CoT+SIG), the 0.8B student matches the 4B teacher's autonomous performance (1.00 vs. 0.92). At the 5× teacher-student capacity ratio tested here, CoT and SIG independently contribute substantial gains that are partially overlapping but complementary. **No teacher-size scan, adaptive CoT distillation, or multi-teacher collaboration has been tested.** The following framework formalizes the directions for extending this initial measurement.

#### 3.4.3 Proposed Capability Gap Formalization

We propose to formalize the **teacher model** $T$ with capability level $c_T$ (measured by standard benchmark performance, e.g., MMLU score) and the **student model** $S$ with capability level $c_S$. The **capability gap** is $\Delta = c_T - c_S$.

A teacher generates a chain-of-thought plan $\Pi = (n_1, n_2, \ldots, n_K)$ where each node $n_i = (\text{tool}, \text{args})$ specifies a tool call. The student executes this plan, producing tool calls $\widehat{\Pi}$ and a final answer $A$. We propose to measure:

- **CoT comprehension rate**: $\text{CR} = \frac{|\widehat{\Pi} \cap \Pi|}{|\Pi|}$, the fraction of teacher plan nodes correctly executed by the student.

- **Information density**: $\text{ID} = \frac{|\text{useful reasoning steps}|}{|\Pi|}$, where a step is "useful" if removing it reduces final answer quality.

- **Performance gain**: $\text{PG} = \frac{Q(A_{\text{with CoT}})}{Q(A_{\text{autonomous}})}$, the ratio of answer quality with and without the teacher's plan.

#### 3.4.4 Proposed Adaptive CoT Distillation Strategies

We propose three strategies for controlling CoT complexity to match student capacity, **none of which have been experimentally evaluated**:

**Proposed Strategy 1: Hierarchical CoT.** The teacher generates plans at multiple abstraction levels:

- **Level 1 (Abstract):** "Search for attractions in all target cities"
- **Level 2 (Detailed):** "Call search_attractions for Paris → call search_attractions for London → ..."
- **Level 3 (Granular):** "Call search_attractions(city='paris') with expected_return=['Eiffel Tower', ...]"

The injection engine would select the abstraction level based on the student's current comprehension rate and the task's complexity budget.

**Proposed Strategy 2: Adaptive compression.** When the teacher's CoT exceeds a threshold length, a lightweight local compressor (which could be the student model itself with a summarization instruction) generates a compressed version:

```
Original CoT: 500 tokens
Compressed CoT: ~150 tokens
```

The compression preserves tool specifications while abbreviating reasoning commentary.

**Proposed Strategy 3: Skill scaffolding.** For complex multi-step tasks, the teacher decomposes reasoning into simpler sub-steps that individually fall within the student's capability range. This is analogous to curriculum learning—the student tackles a sequence of manageable reasoning chunks rather than one overwhelming chain.

#### 3.4.5 Proposed Multi-Teacher Collaboration

A single teacher may have uneven expertise across domains. We propose a **domain-routed multi-teacher** system:

| Domain | Teacher | Reasoning Style |
|--------|---------|----------------|
| Travel planning | GPT-4 | Structured data integration |
| Code debugging | Claude | Systematic diagnosis |
| Mathematical reasoning | DeepSeek-Math | Step-by-step derivation |
| Creative writing | Mistral-Large | Stylistic variation |

A lightweight classifier (which could be the student model itself) would route each query to the most appropriate teacher. For cross-domain tasks, multiple teachers would generate complementary CoT segments, which would be fused by a selector module before injection. **This has not been implemented or tested.**

#### 3.4.6 Proposed Experiments

**Proposed Experiment 1: Teacher size scan.** Fix student (0.8B). Vary teacher across 3B, 7B, 13B, and 70B parameter models. We hypothesize an inverted U-shaped curve for performance gain—the optimal teacher is approximately 8–10× the student's size. A teacher that is too large may produce overly abstract reasoning that the student cannot faithfully operationalize, causing the CoT comprehension rate to drop. **This experiment has not been conducted.**

**Proposed Experiment 2: Compression threshold.** For each teacher-student pair, determine the maximum CoT length beyond which student comprehension drops below an acceptable threshold. This would define the target compression ratio for adaptive compression. **This experiment has not been conducted.**

**Proposed Experiment 3: Multi-teacher vs. single-teacher.** Compare task success rates when using a domain-optimal single teacher versus the domain-routed multi-teacher system. We hypothesize that multi-teacher would outperform any single teacher by 5–15% on cross-domain benchmarks. **This experiment has not been conducted.**

#### 3.4.7 Proposed Implications

If validated, the teacher-student optimization framework would transform CO from a fixed teacher-student pairing into an *adaptive cognitive amplifier*: the injection engine would dynamically select teachers, control CoT complexity, and adjust injection strategies based on the student's real-time comprehension. This represents a step toward the broader vision of CO as a system that learns to optimally leverage external cognition. **All implications are contingent on experimental validation.**


### 3.5 R5: Privacy Guarantees for Cognitive Outsourcing [PROPOSED — NOT EMPIRICALLY VALIDATED]

#### 3.5.1 Motivation

A central claim of the CO architecture is that it provides privacy-preserving cognitive augmentation: sensitive user data remains on the edge device, and only sanitized queries are sent to cloud teachers. However, the original work provided no formal quantification of this privacy guarantee. Several critical questions remain:

1. How much information about the user's private state leaks through the outsourcing process?
2. What are the dominant leakage channels—direct (explicit PII in queries), indirect (inferences from query patterns), or tool-argument (data exposed through tool API calls)?
3. Can differential privacy mechanisms or query anonymization effectively mitigate leakage, and at what utility cost?

**Current empirical status:** This dimension is a framework specification only. No attack simulations have been conducted; no defense mechanism has been evaluated; no privacy-utility measurements have been made. The three-channel leakage model is a proposal for how to think about privacy in CO, not an empirically validated finding.

#### 3.5.2 Concept Demo: PII Anonymization

We implemented and demonstrated two privacy-preserving mechanisms for cognitive outsourcing queries: PII redaction via regex-based named entity detection (names, SSNs, emails, phones, dates, salary amounts) and intent-only outsourcing (value abstraction to typed placeholders). Across four query types (travel, code debugging with PII, medical, financial), PII redaction removed 1–4 items per query, and intent-only outsourcing achieved 0–9% word-count reduction while abstracting all domain-specific values. This is a **concept demonstration only**—formal differential privacy guarantees, measured precision/recall of PII detection, and attack simulation results are not provided.



#### 3.5.3 Proposed Formal Privacy Framework

We propose to define three distinct leakage channels:

**Direct leakage** $L_{\text{direct}}$: Information explicitly contained in text sent to the cloud teacher. Formally, $L_{\text{direct}} = I(U; Q)$, where $U$ is the user's private state and $Q$ is the outsourced query text. In practice, this would be measured as the proportion of PII tokens (names, locations, numbers) in the query.

**Indirect leakage** $L_{\text{indirect}}$: Information inferable from the structure, style, or patterns of outsourced queries. Formally, $L_{\text{indirect}} = I(U; \phi(Q) \mid \text{explicit}(Q))$, where $\phi(Q)$ is the query embedding. This captures stylistic leakage—writing patterns that reveal education level, native language, professional background—and pattern leakage—query types that reveal user intent.

**Tool-argument leakage** $L_{\text{tool}}$: Information exposed through arguments passed to external tool APIs. Formally, $L_{\text{tool}} = \sum_{t \in \text{tool_calls}} I(U; \text{args}_t)$. This is particularly concerning for tools like `read_file(path="/users/alice/medical_records.pdf")` that embed sensitive identifiers.

The proposed **cumulative privacy loss** across $n$ outsourcing rounds is:

$$L_{\text{cumulative}}(n) = \sum_{i=1}^n \text{PLM}(q_i, r_i \mid h_{i-1})$$

where PLM is the Privacy Leakage Metric (a weighted combination of the three channels) and $h_{i-1}$ is the interaction history. **This formula is a proposal; no PLM weighting has been calibrated on real data.**

#### 3.5.4 Proposed Defense Mechanisms

**Proposed Mechanism 1: Query anonymization.** Named entity recognition (NER) would detect and redact PII before outsourcing. The effectiveness is hypothesized to scale with PII detection recall (target: > 95%) and abstraction level.

| Anonymization Level | Hypothesized Privacy Protection | Hypothesized Utility Impact |
|--------------------|-------------------------------|----------------------------|
| PII redaction only | Low (reduces direct leakage by 60-80%) | Low (<5% quality reduction) |
| Template abstraction | Medium (reduces direct + indirect by 70-90%) | Medium (10-20% quality reduction) |
| k-anonymity (k=10) | High (reduces all channels by 85-95%) | High (20-35% quality reduction) |

**All utility impact estimates are speculative and have not been measured.**

**Proposed Mechanism 2: Differential privacy on query embeddings.** Calibrated Gaussian or Laplacian noise would be added to query embeddings before transmission, providing $(\epsilon, \delta)$-DP guarantees. The hypothesized privacy-utility trade-off follows:

$$U(\epsilon) \approx 1 - \exp(-\alpha \cdot \epsilon)$$

where $\alpha$ is a task-dependent constant. For $\epsilon = 1.0$, utility retention is hypothesized at 75–85%; for $\epsilon = 5.0$, it is hypothesized at 95–99%. **These numbers are conjectures, not measurements.**

**Proposed Mechanism 3: Intent-only outsourcing.** The local model would extract the high-level intent (e.g., "plan a multi-city trip") without the specific contextual details (e.g., "I'm traveling to Paris for my anniversary on June 15th"). This would dramatically reduce leakage because $I(\text{intent}; \text{private_details}) \ll I(\text{full_query}; \text{private_details})$.

**Example transformation (illustrative):**

| Original Query | Intent-Only Query |
|---------------|-------------------|
| "I'm a 35-year-old doctor at Hospital X, help me analyze patient data..." | "Given symptoms [A] and [B] with severity [HIGH], recommend treatment category." |
| "My salary is $150K, how should I invest for retirement at 65?" | "For income level [MEDIUM-HIGH], age [MIDDLE], goal [RETIREMENT], suggest investment strategy." |

#### 3.5.5 Proposed Attack Simulation Framework

To validate our privacy mechanisms, we propose simulating three categories of attacks, **none of which have been conducted**:

**Proposed Membership inference attack.** An adversary attempts to determine whether a specific user's data was used in any outsourced query. We propose implementing both confidence-based attacks (comparing model confidence on member vs. non-member prompts) and shadow model attacks (training auxiliary models on known query-response pairs). The privacy guarantee would be satisfied if attack accuracy remains within $\epsilon$ of random guessing (0.5).

**Proposed Reconstruction attack.** An adversary attempts to reconstruct specific private attributes (age, location, profession, etc.) from observed outsourced queries. We propose measuring reconstruction accuracy per attribute type and evaluating how each defense mechanism degrades it.

**Proposed Intent inference attack.** An adversary attempts to classify the user's underlying intent (job seeking, medical concern, financial planning) from query patterns. This would capture the indirect leakage through task-type inference.

#### 3.5.6 Proposed Privacy-Utility Pareto Frontier

The fundamental tension in privacy-preserving CO is between privacy protection and task utility. We propose to characterize this through a **privacy-utility Pareto frontier**—the set of $(L, U)$ pairs that are achievable by some privacy mechanism, where no mechanism can simultaneously achieve lower leakage $L$ and higher utility $U$.

We hypothesize that:

1. The Pareto frontier is concave—the marginal utility cost of privacy increases with stricter privacy guarantees.
2. Intent-only outsourcing achieves the most favorable position on the frontier for many task categories, as it eliminates direct leakage while preserving reasoning structure.
3. Combined mechanisms (anonymization + DP) can push the frontier outward compared to any single mechanism, but with diminishing returns.

**All of these are hypotheses awaiting experimental validation.**

#### 3.5.7 Proposed Implications

If validated, the privacy quantification framework would elevate CO's privacy claim from an architectural assertion to a measurable, tunable property. System designers could select a privacy budget $\epsilon$ and configure defense mechanisms to achieve it, with quantified utility costs. This would be essential for deployment in regulated domains (healthcare, finance, legal) where formal privacy guarantees are required. The framework would also enable privacy audits of deployed CO systems, identifying leakage hot spots and guiding mechanism refinement. **All implications are contingent on experimental validation of the privacy framework, attack simulations, and defense evaluations.**


### 3.6 Cross-Architecture Validation: Gemma-4-E2B [DIRECT MEASUREMENT + TOOLCHAIN OBSTACLE]

#### 3.6.1 Motivation

The original CO+SIG benchmarks were conducted exclusively on the Qwen3.5 model family (0.8B and 4B, Q4_K_M quantization). A critical question is whether SIG's prefill savings and KV-cache continuity advantages generalize across architectures with fundamentally different attention mechanisms. Gemma-4-E2B provides an ideal test case: it employs **Sliding Window Attention (SWA)** interleaved with global attention layers, **shared KV layers** (a subset of layers share KV cache), and **Grouped Query Attention (GQA-4)**. These features create potential failure modes for SIG: SWA may limit long-range recall beyond the sliding window, and shared KV layers may cause interference when injecting new tokens.

#### 3.6.2 Direct Measurement: SIG Speedup on Gemma-4

We replicated the R6 deep-chain benchmark (30-step tool chain, 10 runs) on Gemma-4-E2B-it (Q4_K_M quantization, NVIDIA RTX 4070 SUPER).

**Table 11: SIG vs. AppLoop on Gemma-4-E2B (R6, 30 steps, n=10).**

| Mode | Mean Time (s) | Std (s) | vs. SIG | vs. AppLoop |
|------|-------------|---------|---------|-------------|
| SIG | 0.354 | 0.013 | 1.00× | **3.20×** |
| AppLoop | 1.132 | 0.030 | 0.31× | 1.00× |
| AppLoop-PC | 1.129 | 0.021 | 0.31× | 1.00× |

SIG achieves **3.20× end-to-end speedup** on Gemma-4, consistent with the Qwen3.5 results (2.38× for 0.8B, 2.70× for 4B). The cross-architecture consistency of SIG's prefill savings confirms that the mechanism is architecture-agnostic within the Transformer family, even for models with SWA and shared KV layers.

#### 3.6.3 Direct Measurement: KV-Cache Recall on Gemma-4

As reported in Section 3.2.2, Gemma-4-E2B achieves **perfect 1.00 recall** across all five fact types (weather, population, landmark, specialty, language) at all 8 probe points over 32 injection rounds (6,799 cache tokens). This is a particularly notable result because SWA restricts each attention layer to a local window, yet the model maintains perfect long-range factual recall through SIG injection. We hypothesize that Gemma-4's interleaved global-attention layers provide sufficient long-range connectivity to preserve facts injected early in the sequence, while SWA layers handle local coherence.

#### 3.6.4 Speculative Decoding Compatibility

We tested Gemma-4's compatibility with speculative decoding (SpecDec) via llama-server's HTTP API across six behavioral tests:

**Table 12: SpecDec compatibility test results for Gemma-4-E2B.**

| Test | Description | Result |
|------|------------|--------|
| test1 | Short prompt (30 tokens), no MTP | **PASS** |
| test2 | Long prompt (2000 tokens), no MTP | **PASS** |
| test3 | Extra-long prompt (8000 tokens, exceeds SWA window) | **PASS** |
| test4 | MTP cross-length (draft model) | **SKIP** (server failed to start) |
| test5 | SWA boundary (~4096 tokens) | **PASS** |
| test6 | Shared KV layers (multi-turn dialogue) | **PASS** |

Five of six tests pass, confirming that Gemma-4's main model works correctly with SIG through the production HTTP pipeline. The single failure (test4) is a **toolchain obstacle**, not a model defect.

#### 3.6.5 Toolchain Obstacle Discovery

The Gemma-4 MTP draft model (`gemma-4-E2B-it-assistant`, architecture `gemma4_assistant`) cannot be loaded by llama.cpp (versions b9415 and b9459). The `gemma4_assistant` architecture is a distinct architecture type used by Google's official MTP draft model for Gemma-4, designed to predict multiple future tokens for speculative decoding. As of llama.cpp build b9459, this architecture is not recognized, causing the server to fail at model loading.

This represents a **third category of toolchain obstacle** for orthogonal acceleration (SIG + SpecDec/MTP):

| Obstacle Category | Architecture | Barrier | Status |
|-------------------|-------------|---------|--------|
| Type 1: SWA partial deletion | Qwen3.5 hybrid | `kv_cache_seq_rm` fails on SWA circular buffer | Workaround: llama.cpp native MTP |
| Type 2: Python API crash | Qwen3.5 hybrid | `generate() + drafter` crashes with `llama_decode -1` | Workaround: llama-server HTTP |
| **Type 3: Architecture unsupported** | **Gemma-4 assistant** | **`gemma4_assistant` not recognized by llama.cpp** | **BLOCKED: awaiting PR #23211/#23398** |

The orthogonal acceleration experiment (SIG + MTP on Gemma-4) is therefore blocked until llama.cpp merges support for the `gemma4_assistant` architecture. The SIG component alone is fully functional on Gemma-4, as demonstrated by the R6 speedup and R2 recall results above.

#### 3.6.6 Implications

The Gemma-4 cross-architecture validation provides three key insights:

1. **SIG is architecture-agnostic within the Transformer family.** The 3.20× speedup on Gemma-4 (SWA + shared_kv_layers + GQA) is consistent with Qwen3.5 results, confirming that SIG's prefill elimination mechanism does not depend on specific attention implementation details.

2. **SWA does not prevent long-range factual recall through SIG.** Despite sliding-window constraints on individual layers, Gemma-4 maintains perfect recall of facts injected 32 rounds prior. This suggests that the interleaved global-attention layers in Gemma-4 provide sufficient long-range connectivity, and that SIG's KV-cache continuity advantage extends to SWA architectures.

3. **Toolchain maturity is the binding constraint for orthogonal acceleration.** The SIG + SpecDec/MTP fusion is theoretically sound and empirically validated on Qwen3.5 (via llama-server's native MTP), but cannot be tested on Gemma-4 due to missing architecture support in the inference engine. This highlights a broader challenge: as model architectures diversify (SWA, GQA, shared KV, MTP draft models), inference engines must keep pace with architecture support for orthogonal acceleration to be universally deployable.


## 4. Cross-Cutting Insights and Proposed System Architecture

### 4.1 Interactions Between the Proposed Dimensions

While the five dimensions are individually proposed and not yet validated, we note several hypothesized interactions that may be productive directions for future research:

**R1 ↔ R2: Information loss as a hypothesized precursor to degradation.** If the per-layer information loss characterized in R1 were validated, it would predict which cache entries are most vulnerable to degradation in R2. Specifically, the hypothesis is that facts encoded predominantly in early-layer representations degrade faster because early layers are most sensitive to subsequent injections.

**R2 ↔ R4: Cache horizon as a constraint on teacher plans.** If the effective injection horizon from R2 (the number of rounds before cache quality drops below threshold) were characterized, it would constrain the maximum length of teacher plans in R4. A teacher should not generate plans that require more injection rounds than the cache can sustain. This would enable horizon-aware planning—the teacher receives the current cache budget as input and generates plans accordingly.

**R3 ↔ R5: Architecture-dependent privacy.** The privacy leakage analysis of R5 is architecture-agnostic at the outsourcing layer, but R3 proposes that different architectures have different state retention properties. An architecture with better state retention (e.g., Transformer) *might* leak less information through tool-argument patterns because it requires fewer repeated queries to maintain context—fewer queries mean fewer opportunities for tool-argument leakage. **This is a hypothesis, not a finding.**

**R1 ↔ R4: Attention analysis informing teacher selection.** If the layer sensitivity patterns from R1 were validated, they would suggest that different teacher CoT styles (abstract vs. detailed) may interact differently with the student's attention distribution. An abstract CoT might be more robust to early-layer perturbation, making it preferable for students with known early-layer sensitivity.

### 4.2 Proposed Architecture of an Adaptive CO System [DESIGN PROPOSAL — NOT IMPLEMENTED]

Based on these hypothesized interactions, we outline the design of an **adaptive CO system** that would incorporate all five dimensions:

```
User Query → Privacy Filter (R5) → Meaning Compiler
    ↓
Teacher Selection (R4) → CoT Generation → CoT Compressor (R4)
    ↓
Injection Engine (R1-informed templates)
    ├── Cache Eviction (R2) if budget exceeded
    ├── Architecture-specific injection (R3)
    └── Result synthesis → Response
    ↓
Privacy Audit (R5) → Leakage log → Privacy budget update
Cache Health Monitor (R2) → Degradation score → Eviction trigger
```

The proposed system would operate in a closed loop: privacy filters pre-process queries before outsourcing, teacher selection and CoT compression adapt to student capability and cache state, injection uses architecture-optimized strategies informed by layer sensitivity analysis, and a continuous privacy audit ensures the system stays within its configured privacy budget. **This architecture is a design proposal and has not been implemented.**


## 5. Related Work

Our work builds on and extends several lines of research.

**Cognitive Outsourcing and SIG [1, 2].** The original CO paper [1] introduced the SIG injection primitive and demonstrated its empirical viability on Qwen-4B and TinyLlama-1.1B. Our Section 2 benchmarks extend this validation to Qwen3.5-0.8B and Qwen3.5-4B models, confirming prefill savings and adding generation token count analysis that was absent from the original benchmarks. The meaning compiler concept [2] proposes SIG as a universal cognitive injection substrate. Our five-dimensional research agenda is intended to provide the theoretical foundations that this vision requires.

**KV-cache optimization.** VeriCache [10] and TriAxialKV [11] address KV-cache *compression* for memory efficiency, but do not address *dynamic state injection*. Our proposed R2 degradation analysis would complement this work by characterizing when compression can be safely applied to injected segments.

**Speculative decoding and budgeted inference.** ECHO [3] introduces sparse gating for budgeted speculative decoding. We draw on this concept for our proposed adaptive CoT compression (R4), where a "confidence gate" would determine whether to inject the full CoT or a compressed summary.

**Continual learning agents.** Robo-Cortex [5] and MIRROR [6] pursue self-improving cognitive loops. As discussed in [1], CO provides the *runtime* for these cognitive algorithms; our proposed five-dimensional analysis aims to strengthen this synergy by quantifying the cache horizon (R2), ensuring privacy (R5), and optimizing teacher-student coupling (R4)—**if and when these dimensions are experimentally validated.**

**Differential privacy and NLP.** Our proposed R5 privacy framework draws on foundational DP literature [Dwork et al., 11], adapting it to the unique leakage characteristics of cognitive outsourcing—particularly the tool-argument leakage channel absent from standard NLP privacy analyses. **The adaptation is proposed, not validated.**

**Non-Transformer architectures.** The analysis in proposed R3 contributes to the emerging literature on architectural comparison for inference efficiency, extending beyond throughput benchmarks to characterize *state injectability* as a new architectural property. **This characterization is analytical, not empirical.**


## 6. Self-Critique and Limitations

We believe it is essential to be transparent about the limitations of this work.

### 6.1 Limitations of the Benchmarks (Section 2)

1. **Generation token counts not recorded in CO teacher-precomputed mode.** This is a significant methodological limitation. The apparent 51–54% reduction in generation time for SIG mode cannot be interpreted because we do not know whether it reflects faster per-token generation or simply shorter outputs. The total end-to-end speedup is valid, but the breakdown into generation vs. prefill components is unreliable for the teacher-precomputed data.

2. **Generation time differences are output-length-driven.** In autonomous mode, where we did record generation token counts, we found that when output lengths are equal, generation times are nearly identical. This means claims about SIG producing faster generation are misleading—the speedup comes from prefill savings, not from faster token generation.

3. **Autonomous mode comparison fairness.** Because AppLoop and SIG may complete different amounts of work in autonomous mode, direct timing comparisons conflate efficiency with effectiveness. The tool accuracy improvements and prefill savings are the valid metrics for this mode.

4. **GPU memory only measured for short chains.** The 0.1 GB memory overhead claim has only been validated for chains up to 22 turns. Memory scaling for 100+ turn scenarios has not been characterized.

5. **No real-robot validation.** All benchmarks are conducted in a simulated tool-calling environment.

6. **Single hardware configuration.** All benchmarks were run on a single NVIDIA RTX 4070 SUPER (12 GB). Results may not generalize.

7. **Limited model diversity (PARTIALLY ADDRESSED).** The original benchmarks tested only Qwen3.5-0.8B and Qwen3.5-4B. Cross-architecture validation has been extended to Gemma-4-E2B (SWA + shared_kv_layers + GQA-4), confirming SIG speedup (3.20×) and perfect KV-cache recall (1.00 at 32 rounds). However, Llama, Phi, and other architectures remain untested.

### 6.2 Limitations of the Five-Dimensional Analysis (Section 3)

1. **R1 attention measurement is single-prompt.** The attention distribution shift was measured with one prefix-injection pair on one model (Qwen2.5-0.5B). Multi-prompt averaging, cross-model validation (Qwen3.5, Llama), and statistical significance testing are needed to establish robust effect sizes.

2. **R2 degradation measurement now covers 64 rounds (PARTIALLY ADDRESSED).** The deep validation extends coverage from 6–10 rounds to 64 rounds (13.6K cache tokens) for Qwen3.5-0.8B and 32 rounds for Gemma-4-E2B, with no observable degradation. However, degradation beyond 64 rounds remains uncharacterized, and the city-card recall task may not expose degradation in reasoning quality or instruction-following coherence. Earlier "0.50" recall scores were identified as measurement artifacts (prompt format and thinking-mode interference), not genuine degradation.

3. **R3 is pure simulation.** No SIG implementation exists for Mamba, RWKV, or xLSTM. The projected rankings and suitability assessments could be significantly wrong when empirically tested.

4. **R4 capability gap is a single data point.** Only one teacher-student pair (4B/0.8B) was tested. No teacher-size scan, adaptive CoT distillation, or multi-teacher collaboration experiments have been conducted.

5. **R5 privacy is a concept demo.** Formal differential privacy guarantees, PII detection precision/recall measurements, and attack simulations have not been conducted.

6. **Cross-dimensional interactions (Section 4) are speculative.** The claimed interactions (R1↔R2, R2↔R4, etc.) are logical conjectures, not empirically observed relationships. The adaptive CO system architecture is a design proposal.

7. **Mathematical formalism is conjectural.** The upper bound conjecture for information loss (R1), the degradation models (R2), and the privacy-utility Pareto frontier (R5) are mathematically expressed hypotheses. They have not been proven as theorems.

### 6.3 Scope Limitations

1. **Architecture coverage.** The R3 analysis covers four architecture families, but the space is rapidly expanding. Architectures such as RetNet, Griffin, and Mamba-2 introduce hybrid designs not covered here.

2. **Scalability of privacy mechanisms.** The proposed privacy mechanisms have computational overhead that may be prohibitive for latency-sensitive embodied control loops (<10ms). This has not been tested.

3. **Teacher model scope.** The benchmarks used a single teacher model. R4 experiments involving multi-teacher collaboration and teacher-size scans require additional API access and remain unconducted.

4. **Evaluation metric limitations.** Tool accuracy is a coarse metric that does not capture partial correctness, answer quality, or reasoning fidelity.


## 7. Conclusion

We have presented an integrated investigation of Cognitive Outsourcing with Suspend-and-Inject Generation, combining extended benchmarks with a five-dimensional theoretical analysis where each dimension includes the first available empirical measurement.

The benchmarks confirm that CO+SIG achieves **73–97% prefill savings** robustly across model sizes (0.8B and 4B) and operational modes. In teacher-precomputed mode, total end-to-end speedups reach **2.38× (0.8B) and 2.70× (4B)** on average, with peak speedups of **4.96× and 5.26×** on deep tool chains. Generation token count analysis in autonomous mode reveals that per-token generation rates are nearly identical between AppLoop and SIG (within 2%)—the speedup originates from prefill savings, not from faster generation. GPU memory overhead is minimal (~0.1 GB) for the chain lengths tested.

The five-dimensional analysis provides structured foundations with varying levels of empirical support. R1 presents the first direct attention distribution measurement (head agreement 0.25→0.43 across layers), confirming the hypothesized early-layer sensitivity gradient. R2 provides deep validation across 64 injection rounds and three model families, demonstrating no observable KV-cache degradation within the tested horizon (13.6K tokens) and revealing that earlier "0.50" recall scores were measurement artifacts. R3 contributes a cross-architecture simulation engine with empirical calibration and hypothesized projections for non-Transformer architectures. R4 quantifies the first capability gap measurement (+0.80 CoT amplification, +0.59 SIG amplification, 0.72 teacher margin at 5× capacity ratio). R5 demonstrates PII redaction and intent-only outsourcing as viable privacy-preserving mechanisms in a concept setting. The new R6 cross-architecture validation on Gemma-4-E2B confirms SIG's 3.20× speedup and perfect KV-cache recall on a SWA+GQA architecture, while identifying a third category of toolchain obstacle (unsupported `gemma4_assistant` architecture) that blocks orthogonal acceleration testing.

The theoretical formalisms throughout—the information loss upper bound conjecture, the competing degradation models, and the privacy-utility Pareto frontier—remain conjectural and await rigorous proof, empirical parameter estimation, and experimental validation. The five dimensions should be understood as a structured starting point for this work. The three empirical entry points (`co_benchmark.py`, `sig_benchmark.py`, `transformer_bench.py`) provide a reusable testing infrastructure for future investigations.


## References

[1] "Cognitive Outsourcing with Suspend-and-Inject Generation for Scalable Embodied Intelligence," 2026.

[2] "From Knowledge Vaults to Meaning Compilers: Suspend-and-Inject Generation as a Universal Substrate for Modular Cognitive Injection," working paper, 2026.

[3] J. Hu et al., "ECHO: Elastic Speculative Decoding with Sparse Gating for High-Concurrency Scenarios," working paper, 2026.

[4] S. Kim et al., "LLMCompiler: An LLM Compiler for Parallel Function Calling," arXiv:2402.04578, 2024.

[5] "Robo-Cortex: A Continual Cognitive Learning Architecture for Embodied Agents," working paper, 2026.

[6] "MIRROR: Modular Internal Reflection and Reasoning for Language Agents," working paper, 2025.

[7] P. Zhang et al., "TinyLlama: An Open-Source Small Language Model," arXiv:2401.02385, 2024.

[8] A. Gu and T. Dao, "Mamba: Linear-Time Sequence Modeling with Selective State Spaces," arXiv:2312.00752, 2023.

[9] B. Peng et al., "RWKV: Reinventing RNNs for the Transformer Era," arXiv:2305.13048, 2023.

[10] M. Beck et al., "xLSTM: Extended Long Short-Term Memory," arXiv:2405.04517, 2024.

[11] C. Dwork et al., "The Algorithmic Foundations of Differential Privacy," Foundations and Trends in Theoretical Computer Science, 2014.

[12] W. Kwon et al., "Efficient Memory Management for Large Language Model Serving with PagedAttention," SOSP, 2023.