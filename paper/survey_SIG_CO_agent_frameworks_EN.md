# From Cognitive Outsourcing to Agent Inference Engines: Core Ideas of Suspend-and-Inject Generation and Their Implications for Modern Agent Frameworks

**A Survey**

> This article provides a systematic distillation and survey based on the six-paper CO-SIG research program [1-6].
>
> **Date**: June 2026

---

## Abstract

Modern LLM agent frameworks (LangChain, AutoGPT, SWE-agent, etc.) universally adopt **stateless application-layer loops** to orchestrate tool calls—after each external interaction, the model must re-encode all context from scratch, leading to quadratically growing prefill costs and catastrophic disruption of attention states. This survey reviews the core ideas of the Cognitive Outsourcing (CO) and Suspend-and-Inject Generation (SIG) research program: elevating tool interaction from an application-layer interface to an **inference-engine-level primitive**, achieving cross-tool-call cognitive continuity via KV-Cache injection.

Based on the complete empirical and theoretical body of the seven-paper series, this survey is organized across three dimensions: (1) **SIG's core ideas**—the KV-Cache continuity principle, the five-stage suspend-inject-resume loop, and the "trading storage for computation" design philosophy; (2) **CO's cognitive architecture**—how the three-layer system (meaning compiler, injection engine, cognitive module ecosystem) endows lightweight edge models with dynamic access to external cognitive resources; and (3) **implications for modern agent frameworks**—how stateful runtimes, composable acceleration orthogonality, Batch-SIG's architecture-agnosticism, and the KV-Cache-as-First-Class-Citizen (KFC) unified optimization framework can reshape agent system design.

This survey aims to provide agent inference framework designers with a complete roadmap from principles to practice, delineating SIG's zone of advantage, design boundaries, and its profound impact on next-generation edge-cloud collaborative agent architectures.

**Keywords**: Cognitive Outsourcing, Suspend-and-Inject Generation, KV-Cache continuity, edge agents, inference engine optimization, agent frameworks

---

## 1. Introduction: The Paradigm Dilemma of Agent Inference

### 1.1 Fundamental Deficiencies of Stateless Loops

Current mainstream LLM agent frameworks adopt a uniform execution pattern: the model generates a tool call → pauses generation → executes the tool → appends the tool result to context → **re-encodes the entire conversation history from scratch** → resumes generation. This "Application Layer Loop" (AppLoop) has three fundamental deficiencies:

1. **Quadratic prefill overhead**. After each tool call, the accumulated context must be fully re-encoded. For a 4B model on a 30-step tool chain, prefill cost grows from tens of milliseconds at step one to several seconds at step thirty, with total prefill time exceeding 2 seconds [4].
2. **Catastrophic disruption of attention states**. Re-encoding means the model's KV-Cache—its "working memory"—is discarded and rebuilt from zero. For embodied agents tracking spatial relationships, maintaining task progress, and preserving reasoning traces, this is a cognitive catastrophe [1].
3. **Destruction of small-model tool-use capability**. Empirical data shows that the 0.8B model achieves 0% tool accuracy on 6/9 scenarios under AppLoop—repeated re-encoding completely obliterates any contextual understanding the model has built [2].

### 1.2 Cloud Optimizations Cannot Descend to the Edge

FlashAttention, RadixAttention, and continuous batching have reduced prefill overhead to near-negligible levels in cloud serving systems (vLLM, TensorRT-LLM, SGLang). However, these optimizations **cannot descend to edge devices**:

- llama.cpp (the de facto edge inference library) does not implement FlashAttention;
- Embedded devices (Jetson, smartphone NPUs) lack the required GPU architecture;
- Single-user, single-instance edge deployments have no concurrent requests for batching amortization;
- Edge tool calls (sensors, local perception pipelines) have latencies under 100ms, where prefill genuinely dominates end-to-end time.

CO+SIG is explicitly positioned at this "orthogonal frontier": edge-native single-instance agent inference [1].

---

## 2. SIG's Core Ideas

### 2.1 The KV-Cache Continuity Principle

SIG's core idea can be summarized in one sentence: **preserve the model's working memory rather than discarding it after each tool call**.

Standard autoregressive decoding maintains a KV-Cache encoding the attention state of all processed tokens. This cache is the model's *working memory*—its implicit understanding of conversation history and the current reasoning trajectory. SIG's innovation is: when the model needs external information, rather than discarding this cache and rebuilding from scratch, **inject the external information into the existing cache**, allowing the model to resume reasoning from the exact point where it was interrupted, now equipped with new information.

The formal characterization of this idea is: under standard AppLoop, the full prefix (X, G_{<t}, R) is re-encoded after a tool call, producing hidden state H^{full}; under SIG injection, only the tool result tokens R are prefilled using the suspended KV-Cache as prefix, producing H^{inj}. The key empirical finding is [2]:

> **SIG and AppLoop generate text at nearly identical per-token rates (108 vs 103 tok/s, ±5% difference).** The speedup comes entirely from prefill elimination, not changes to the generation path.

This finding has profound implications: SIG changes what is in the KV-Cache at generation start, but does not alter the autoregressive decoding path itself. It is a **pure prefill-layer optimization**.

### 2.2 The Five-Stage Suspend-Inject-Resume Loop

SIG elevates tool interaction from an application-layer interface to an inference-engine primitive through a carefully designed five-stage loop [1]:

1. **Suspend**: When a predefined suspend token (e.g., <<<TOOL>>>) is detected in the generation stream, pause autoregressive decoding; the entire KV-Cache is preserved.
2. **Resolve**: Parse the text following the token to identify the requested cognitive module and its parameters.
3. **Fetch**: The injection engine calls the specified module—local sensors, cloud teachers, skill libraries, etc.
4. **Inject**: The module's text response is tokenized, wrapped in a stabilization template, and a single forward pass is executed with the suspended KV-Cache as prefix. This appends the injected tokens to the cache without recomputing any prior context.
5. **Resume**: Autoregressive decoding continues from the extended cache, which now contains the new information.

Since only the injected tokens (plus template) are prefilled, cost scales linearly with injection size and is independent of total conversation length—starkly contrasting AppLoop's quadratic re-encoding cost.

**Stabilization templates** are a critical engineering detail of this mechanism: small language models (<1B parameters) are sensitive to distributional shift from external tool outputs. Each injection is wrapped in a structured preamble that reduces formatting error rates from >30% to <2%, at the cost of only ~15-20 additional tokens per injection [1].

### 2.3 The "Trading Storage for Computation" Design Philosophy

SIG and Mooncake (cloud KV-Cache-centric disaggregated serving architecture, FAST 2025 Best Paper) independently converged on the same design philosophy [6]:

> **Trading Storage for Computation**: expanding KV-Cache storage to reduce GPU recomputation yields benefits that outweigh storage costs.

This philosophy manifests differently across deployment regimes:

| Dimension | Cloud (Mooncake) | Edge (SIG) |
|----------|-----------------|------------|
| Storage medium | VRAM → DRAM → SSD (cross-cluster, TB-scale) | VRAM → DRAM (single-device, GB-scale) |
| Cache reuse scope | Cross-request, cross-node (global pool) | Cross-tool-call, intra-request (injection continuity) |
| Compute-storage weight | γ ≫ α (storage-dominated) | α ≫ γ (prefill-dominated) |
| Scheduling complexity | Global Conductor | No scheduling (direct execution) |

The KV-Cache-as-First-Class-Citizen (KFC) framework [6] formalizes both as regime-specific specializations of the same optimization objective:

$$\min_{\mathcal{A}} \Phi(\mathcal{A}) = \sum_{r \in \mathcal{R}} \alpha \cdot C_{\text{prefill}}(r, \mathcal{A}) + \beta \cdot C_{\text{transfer}}(r, \mathcal{A}) + \gamma \cdot C_{\text{storage}}(r, \mathcal{A})$$

Mooncake is the optimal solution under cloud constraints (multi-tenant, cluster-scale, throughput-oriented); SIG is the optimal solution under edge constraints (single-user, device-scale, latency-oriented). They are duals of the same principle [6].

---

## 3. CO's Cognitive Architecture

### 3.1 Three-Layer System Architecture

Cognitive Outsourcing organizes edge intelligence into three layers [1]:

**Layer 1: Meaning Compiler**. A lightweight autoregressive language model (0.8B-4B parameters) running entirely on-device. Its scope is narrow: parse user instructions into structured intent, determine which cognitive modules are needed, emit SIG suspend tokens with correct module specifications, and synthesize coherent responses from all injected context. The key insight is that the meaning compiler need not store world knowledge or possess advanced reasoning skills; it only needs robust instruction-following and in-context synthesis capability.

**Layer 2: Injection Engine**. A thin runtime layer extending the on-device inference library (e.g., llama.cpp). It intercepts the generation stream, manages the KV-Cache, parses tool descriptions, coordinates module invocations, and enforces safety policies (attention masking, output sanitization, module isolation).

**Layer 3: Cognitive Module Ecosystem**. Text-based services conforming to a standard manifest, including:
- **Local perception and action modules**: object detection, SLAM queries, motion planning—all running on-device, protecting sensor privacy.
- **Cloud teacher modules**: frontier LLMs (GPT-4, Claude) packaged as reasoning scaffolds—zero-shot, in-context knowledge distillation.
- **Local cognitive cache**: persistent memory of successful reasoning chains supporting sub-millisecond reuse—a natural repository for the heuristic principles distilled from continual learning agents like Robo-Cortex [5].
- **Skill library**: pre-encoded atomic action sequences that can be injected as structured prompts.

### 3.2 Cloud Teachers and In-Context Imitation

When the meaning compiler encounters a task beyond its reasoning capacity, it invokes a cloud teacher. The teacher's response is wrapped in a template that guides the local model in summarizing and adapting expert reasoning—a zero-shot, in-context knowledge distillation [1].

Empirical measurements show [2]: on the 0.8B model, autonomous AppLoop achieves 0.20 accuracy; adding a 4B teacher's precomputed CoT plan plus SIG lifts it to 1.00—CoT amplification +0.80 (80 percentage points), SIG amplification +0.59 (59 percentage points). Teacher quality margin (4B alone minus 0.8B alone) is 0.72. At a 5× teacher-student capacity ratio, CoT and SIG contribute independently significant, partially overlapping but complementary gains.

---

## 4. Empirical Landscape: From Zone of Advantage to Design Boundaries

### 4.1 Robustness of Prefill Savings

SIG's prefill savings exhibit high robustness across model sizes, operating modes, and architectures [2, 3]:

| Metric | 0.8B | 4B |
|--------|------|-----|
| Prefill token savings | 73-93% | 78-97% |
| End-to-end speedup (teacher precomputed) | 2.38× | 2.70× |
| Deep-chain peak speedup | 4.96× | 5.26× |
| 30-step tool chain speedup (N=30) | 2.79× | 4.26-4.38× |
| 50-step Kitchen benchmark speedup | — | 5.3× |
| GPU VRAM overhead | +0.1 GB | +0.1 GB |

Cross-architecture validation [3] reveals critical architectural dependencies:

| Architecture | SIG Prefill Speedup | Batch-SIG vs AppLoop-PC |
|-------------|--------------------|-----------------------|
| Qwen3.5 (dense attention) | 2.38-2.70× ✅ | 4.65× ✅ |
| Gemma-4-E2B (GQA + SWA) | 1.12× ⚠️ | 6.82× ✅ |
| Nemotron-3-Nano-4B (hybrid Mamba+attention) | 0.98× ❌ | 4.24× ✅ |

**Core finding**: Raw prefill savings are Qwen-family-specific, but **Batch-SIG is architecture-agnostic** [3]. By accumulating tool results before a single generation call, Batch-SIG achieves 4.24-6.82× cross-architecture speedup—the most robust cross-architecture finding in the entire research program.

### 4.2 The Rescuing Effect of KV-Cache Continuity on Small Models

SIG's impact on small-model multi-turn tool-use capability is dramatic [2]:

Under autonomous mode with the 0.8B model, AppLoop catastrophically fails on 6/9 scenarios (0% tool accuracy)—repeated re-encoding completely destroys any context the model has built. SIG rescues 5 of these scenarios, achieving 68-100% accuracy. This provides indirect but compelling evidence that **KV-Cache continuity is critical for small-model multi-turn agent capability**.

Deeper KV-Cache probing experiments [3] sharpen the mechanism of this finding:
- **Single-entity completion**: 100% success after SIG injection—information is indeed present in the KV-Cache and attention-accessible.
- **Multi-entity enumeration**: 0/5 for both SIG and AppLoop—this is a model capacity limitation, not a SIG-specific defect.

This reveals a fundamental distinction: KV-Cache injection preserves **associative access** (analogies, constraint satisfaction, partial completion) but loses **enumerative access** (exhaustive lists, precise formatted recall) [3].

### 4.3 Design Boundaries: When Not to Use SIG

A mature acceleration paradigm must have well-defined boundaries. SIG's design boundaries have been systematically characterized [3, 4]:

**Fragmented context assembly (R13)**. When the workload consists of many small, unrelated fragments, SIG is 3.0× slower than AppLoop-PC (0.8B). SIG optimizes "inject once, consume continuously," not frequent small-scale reassembly.

**Random-access retrieval (R8)**. AppLoop-PC achieves 50-64% retrieval hit rates, while SIG achieves 0% on 0.8B. SIG's persistent cache is a streaming tape, not a random-access array.

**Cloud-dependent tools (>300ms latency)**. SIG's speedup collapses from 5.49× (0ms) to 1.08× (500ms). When tool latency dominates, prefill savings become negligible.

**Non-Qwen dense attention architectures**. Raw prefill savings do not transfer to hybrid Mamba+attention (Nemotron, 0.98×) or GQA architectures (Gemma-4, 1.12×).

These boundaries are not SIG failures—they are expected behaviors of the streaming injection paradigm outside its design scope, just as a tape drive does not "fail" when it cannot match RAM's random-access latency [3].

---

## 5. Theoretical Foundations: The Five-Dimensional Analysis Framework

### 5.1 Information-Theoretic Foundations (R1)

The attention distribution shift between SIG injection and full re-encoding has been directly measured for the first time on Qwen2.5-0.5B [2]:

| Layer Group | Head Agreement | Cosine Similarity | Interpretation |
|------------|---------------|-------------------|----------------|
| Early (0-7) | 0.252 | 0.647 | **Most sensitive** — base attention patterns most disrupted |
| Mid (8-15) | 0.304 | 0.735 | Partially recovered via self-attention |
| Late (16-23) | 0.427 | 0.793 | **Most stable** — task refinement layers least affected |

Despite head agreement of only 0.25-0.43, per-token generation rates and Token-Jaccard fidelity are nearly identical. This tension is resolved through the concept of **representational redundancy**: multiple attention pathways in Transformer architectures can encode the same semantic content—the model may attend to different positions but arrive at similar internal representations of task-relevant facts through distributed encoding [2].

### 5.2 KV-Cache Lifecycle and Degradation Analysis (R2)

64-round deep validation experiments [2, 3] measured multi-turn factual recall across three model families:

| Model | Max Rounds | Max Cached Tokens | Long-term Recall | Degradation? |
|-------|-----------|-------------------|-----------------|--------------|
| Qwen3.5-0.8B | 64 | 13,574 | 0.93 (stable) | No |
| Qwen3.5-4B | 32 | 6,800 | 0.93 (stable) | No |
| Gemma-4-E2B | 32 | 6,799 | 1.00 (perfect) | No |

Within the range of 64 injection rounds and 13.6K cached tokens, **no measurable degradation was observed**. Previously reported 0.50 recall scores were confirmed to be measurement artifacts (prompt formatting errors and thinking-mode interference), not genuine degradation [2].

### 5.3 The Orthogonal Acceleration Framework (R5 Paper)

SIG's macro-level prefill elimination and speculative decoding's micro-level generation compression operate on **orthogonal dimensions** of the inference pipeline [5]. The orthogonality ratio ρ is formally defined as:

ρ = S_{SIG+MTP} / (S_{SIG} × S_{MTP})

When ρ = 1.0, the two accelerations compose perfectly multiplicatively. Empirical validation [5]:

| Validation Method | ρ | Verdict |
|------------------|--------|---------|
| n-gram order validation (4B) | 0.851 | PASS (≥ 0.85) |
| Native MTP parallel validation (4B, Kitchen) | 1.239 | PASS, slightly super-multiplicative |

Composite SIG+MTP achieves **4.52×** end-to-end speedup (SIG 3.50× + MTP 1.27×), validating the practical viability of the orthogonal acceleration framework. The slight super-multiplicativity (ρ > 1.0) stems from SIG's cache persistence improving MTP draft acceptance rates—a positive interaction effect.

---

## 6. Implications for Modern Agent Frameworks

### 6.1 Stateful Runtimes: From Stateless Loops to Cognitive Continuity

Current agent frameworks (LangChain, LlamaIndex, AutoGPT) universally adopt stateless application-layer loops. SIG's core implication is: **agent frameworks should treat KV-Cache management as a first-class citizen, rather than relying on application-layer context concatenation**.

Specifically, this requires the inference engine to expose three primitives:
- `suspend()`: pause generation, preserve KV-Cache;
- `inject(tokens)`: inject external information into the existing cache;
- `resume()`: continue generation from the extended cache.

This transforms tool interaction from "API call + context reconstruction" to "cache state transition," fundamentally altering the agent's interaction model with the external world. For continual learning agents (e.g., Robo-Cortex [1]), this means:

- **Continuous reflection**: the KV-Cache preserves full task execution context; when the agent enters a reflection phase, it can inject historical summaries without resetting spatial memory.
- **Heuristic caching**: principles derived from autonomous knowledge induction are stored as local cognitive modules, injected directly upon encountering similar situations at zero additional inference cost.
- **From outsourcing to internalization**: the path from "outsourcing" cognition to the cloud, to "caching" successful patterns, to gradually "internalizing" them as local, parameter-free skills—edge cognitive evolution.

### 6.2 The Orthogonality Principle of Composable Acceleration

The SIG research program reveals a broader **design principle** [5]: accelerations targeting different stages of the inference pipeline can compose multiplicatively, provided they act on independent computational dimensions.

The practical implications of this principle are profound. Modern agent frameworks can systematically combine multiple optimization strategies:

| Acceleration Dimension | Mechanism | Target Stage | Relationship to Others |
|----------------------|-----------|-------------|----------------------|
| SIG (macro) | KV-Cache injection | Prefill | Orthogonal to MTP |
| MTP (micro) | Multi-token prediction | Generation | Orthogonal to SIG |
| Batch-SIG (structural) | Batched injection | Generation calls | Architecture-agnostic |
| CompSIG (memory) | Periodic compression | Cache management | Orthogonal to above |

The orthogonality framework provides a quantitative tool for evaluating whether any composite acceleration is worth the implementation effort: if ρ is projected below 0.70, accelerations may interfere and should not be naively combined [5].

### 6.3 Privacy-Preserving Paradigms for Edge Agents

CO's privacy guarantees stem from a structural property: **sensitive user data remains on the edge device, with only sanitized queries sent to cloud teachers** [1, 2].

Privacy leakage is formalized through three channels [2]:
- **Direct leakage** L_{direct} = I(U; Q): PII explicitly contained in query text.
- **Indirect leakage** L_{indirect} = I(U; φ(Q) | explicit(Q)): information inferable from query structure, style, or patterns.
- **Tool parameter leakage** L_{tool} = Σ_t I(U; args_t): information exposed through external tool API parameters.

Conceptual demonstrations [2] showcase two mechanisms: PII editing (regex-based named entity detection) and intent-only outsourcing (abstracting domain-specific values into typed placeholders). While formal differential privacy guarantees remain future work, CO's architectural privacy properties—edge orchestration with cloud assistance—provide a structural foundation for agent deployment in privacy-sensitive domains (healthcare, finance, legal).

### 6.4 KV-Cache as First Class Citizen: The KFC Unified Framework

The KFC framework [6] unifies SIG and Mooncake as regime-specific specializations of the same optimization principle. Its core implication for agent framework design is:

> **KV-Cache is not a byproduct of computation, but the central resource of inference system design.**

Practical implementations of this principle include:

1. **KV-Cache persistence**. Storing cache prefix hashes and blocks on local SSDs, enabling session-to-session prefix reuse without re-encoding—equivalent to Mooncake's prefix hash matching localized to the edge [6]. Engineering cost: ~2-3 months.

2. **Multi-tier storage**. Extending CompSIG's compression pipeline with DRAM offload: VRAM holds active cache (~8,000 tokens), DRAM holds compressed cache (~20,000 tokens), achieving 3.5× effective context extension [6].

3. **Batch-SIG as the default deployment strategy**. Batch-SIG is recommended as the default across all architectures [3]. By accumulating tool results before a single generation call, Batch-SIG converts SIG's worst case (fragmented assembly) into its zone of advantage (4.24-6.82× vs AppLoop-PC). Automatic batch size selection is achieved via the `DependencyAnalyzer`, which classifies tool calls as independent (batchable) and sequentially dependent (non-batchable) [3].

### 6.5 Hybrid Scheduling and Decision Framework

The decision framework produced by the SIG research program [3, 4] can serve directly as routing heuristics for edge inference runtimes:

| Workload | Use | Rationale |
|----------|-----|-----------|
| Long continuous tool chains | **SIG** | Incremental cost constant; prefix cache invalid |
| Fragmented independent assembly | **AppLoop-PC or Batch-SIG** | SIG's stepwise overhead exceeds single re-encoding |
| Random-access retrieval | **AppLoop-PC** | Full re-encoding provides direct positional access |
| CoT structured generation | Either | Performance equivalent; SIG adds incremental extension capability |
| Cloud-dependent tools (>300ms) | Either | SIG advantage collapses to <1.15× |
| Safety-critical (rollback) | **SIG + isolation** | Rollback enables clean keyword-level recovery |

Edge inference runtimes can detect workload characteristics at runtime and route accordingly: deep chains to SIG, fragmented assembly to Batch-SIG, retrieval workloads to AppLoop-PC, cloud tasks to lower-overhead schemes. This is analogous to modern compilers choosing between inlining (streaming) and function calls (reassembly) based on code characteristics [3].

---

## 7. Limitations and Future Directions

### 7.1 Known Limitations

1. **Single model family**. Most core benchmarks were conducted on Qwen3.5 (0.8B/4B, Q4_K_M). Cross-architecture validation has been extended to Gemma-4 and Nemotron, but Llama, Phi, and other architectures remain to be tested [3].
2. **Quality-speed tradeoff**. Task quality under SIG mode exhibits a measurable but narrower gap: mixed TF-IDF score of −0.12 on 4B (keyword score of −0.39). The gap concentrates on recipe enumeration, not tool execution or allergen awareness [3].
3. **Tool latency sensitivity**. SIG's end-to-end speedup collapses to 1.08× at 500ms tool latency [3]. This constrains SIG's value proposition to scenarios with fast local tool execution (<100ms).
4. **Cache enumeration limitation**. KV-Cache injection supports single-entity associative access (100% completion) but not multi-entity exhaustive enumeration (0%) [3]. This is a fundamental architectural limitation of SIG.
5. **Synthetic tools**. All benchmarks use synthetic tool implementations. Real-world tool latency, noise, and variable response lengths may interact with SIG's injection granularity in uncharacterized ways.
6. **Conditional value of state externalisation.** SECM-H [7] experiments reveal an important methodological lesson: pre-scripted benchmarks bypass the evaluation of module management capabilities. In pre-scripted scenarios, the model's implicit KV-Cache tracking is sufficient, and injecting module management state *degrades* quality ($\Delta Q_{content} = -0.141$). Only in agent-driven noisy scenarios does SECM-H demonstrate positive value ($\text{ToolAcc} = 97.1\%$ vs SIG's $94.3\%$). This shows that the value of state externalisation depends on whether the task genuinely tests the externalised capability.

### 7.2 Future Directions

1. **Local KV-Cache disk persistence** (highest priority). Implementing disk-backed KV-Cache persistence in llama.cpp, enabling cross-session prefix reuse without network connectivity. Engineering cost ~2-3 months, with benefits equivalent to a full hybrid architecture at 1/6 the cost [6].
2. **KV-Cache architectural remediation**. SIG's fundamental limitation of reliably enumerating entities from distributed KV-Cache state demands architecture-level solutions, such as hybrid KV-Cache layouts with explicit token slots reserved for critical entities [3].
3. **Cross-family replication**. Validating the deep-chain advantage from R6 on ≥7B scales of Llama, Gemma, and Mistral.
4. **Runtime integration of Batch-SIG dependency classifier**. Integrating the `DependencyAnalyzer` into edge inference runtimes for automatic batch size selection [3].
5. **Physical robot deployment for embodied agents**. Current benchmarks abstract physical actions as text-based tool calls. Deploying SIG+MTP on physical robots validates the framework under real-world conditions.
6. **KV-Cache standardization**. Proposing standardization of KV-Cache metadata schemas to resolve format fragmentation across llama.cpp, vLLM, SGLang, and TensorRT-LLM [6].
7. **Deeper investigation of module management state externalisation.** Paper 8 [7] demonstrates that in autonomous agent scenarios, module reliability tracking (SECM-H) improves content quality ($\Delta Q_{content} = +0.101$ under noise). Key open questions include: (a) Path A/B causal decoupling—whether quality improvement stems primarily from better tool selection or changed generation behaviour; (b) whether natural-language state rendering ($\Delta Q_{content} = +0.113$) can replace structured template injection; (c) the model-size boundary—the precise crossover point where models transition from benefiting (2B) to not benefiting (0.8B).
8. **Agent-driven evaluation benchmarks.** Paper 8 reveals a fundamental flaw in pre-scripted benchmarks: when tool selection is pre-determined, any optimisation affecting module selection capability cannot be measured. Future research should establish standardised agent-driven evaluation frameworks that systematically vary module count, failure rate, and dependency depth to accurately assess the value of cognitive architecture layers.

---

## 8. Conclusion

The core ideas of Suspend-and-Inject Generation can be distilled into three principles:

1. **Tool interaction is an inference engine primitive, not an application-layer interface.** By implementing the suspend-inject-resume loop at the inference engine layer, SIG fundamentally changes the agent's interaction model with the external world—from "API call + context reconstruction" to "cache state transition."

2. **KV-Cache is working memory, not transient data.** Preserving the model's attention state across tool call boundaries not only eliminates 73-97% of redundant prefill, but more importantly maintains cognitive continuity—which has a decisive impact on small models' multi-turn tool-use capability.

3. **Trading storage for computation is a cross-regime unifying principle.** From edge injection continuity to cloud global KV-Cache pools, the principle of treating KV-Cache as a first-class citizen is unified, merely manifesting differently across deployment regimes.

Paper 8 [7] adds an important qualification to this paradigm: **the value of state externalisation depends on whether the task genuinely tests the externalised capability.** SECM-H's module management state is pure noise in pre-scripted benchmarks, but provides genuine decision support in agent-driven noisy scenarios. This not only revises our understanding of cognitive architecture layers—from "universal Layer 2" to "generation stabilizer under specific conditions"—but raises a general methodological caution: evaluation benchmarks must match the capability being evaluated.

For modern agent frameworks, the implications of the CO-SIG research program are structural: it is not merely another performance optimization trick, but a proposal to **rethink the agent inference pipeline paradigm**—a stateful, KV-Cache-centric, composable-acceleration, privacy-preserving edge agent runtime. This paradigm's value grows with model scale and task complexity; its design boundaries have been systematically characterized; its core mechanisms have been cross-architecturally validated.

As Mooncake and SIG independently converging on the same design philosophy from opposite starting points implies: as models grow larger and context windows grow longer, KV-Cache management will become the central concern of inference system design, regardless of deployment regime. Systems that treat KV-Cache as a first-class citizen—preserving it, sharing it, moving it efficiently—will outperform those that treat it as a transient byproduct.

---

## References

[1] Cognitive Outsourcing with Suspend-and-Inject Generation for Scalable Embodied Intelligence. *CO+SIG Research Program, Paper 1*, 2026.

[2] Beyond the Injection Engine: A Five-Dimensional Analysis of CO-SIG. *CO+SIG Research Program, Paper 2*, 2026.

[3] CO-SIG Architecture, Theory, and Empirical Design Space for Scalable Edge Intelligence. *CO+SIG Research Program, Paper 3*, 2026.

[4] Suspend-and-Inject Generation as an Edge Inference Runtime Primitive for Long-Horizon Agent Tasks. *CO+SIG Research Program, Paper 4*, 2026.

[5] Orthogonal Acceleration: Fusing Speculative Decoding with Suspend-and-Inject Generation. *CO+SIG Research Program, Paper 5*, 2026.

[6] Convergent KVCache Architectures: Bridging Cloud-Scale Disaggregated Serving and Edge-Native Injection Continuity. *CO+SIG Research Program, Paper 6*, 2026.

[7] State-Externalizing Cognitive Module Harnesses: Elevating Cognitive Outsourcing from Injection-Level to System-Level Orchestration. *CO+SIG Research Program, Paper 8*, 2026.

[8] R. Qin et al. Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving. *arXiv:2407.00079*, 2024. FAST 2025 Best Paper.

[9] W. Kwon et al. Efficient Memory Management for Large Language Model Serving with PagedAttention. *SOSP*, 2023.

[10] L. Zheng et al. SGLang: Efficient Execution of Structured Language Model Programs. *NeurIPS*, 2024.

[11] S. Kim et al. LLMCompiler: An LLM Compiler for Parallel Function Calling. *arXiv:2402.04578*, 2024.

[12] Robo-Cortex: A Continual Cognitive Learning Architecture for Embodied Agents. *Working Paper*, 2026.

[13] J. Hu et al. ECHO: Elastic Speculative Decoding with Sparse Gating for High-Concurrency Scenarios. *Working Paper*, 2026.

[14] G. Xiao et al. Efficient Streaming Language Models with Attention Sinks. *ICLR*, 2024.

[15] Z. Zhang et al. H2O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models. *NeurIPS*, 2023.

[16] C. Dwork et al. The Algorithmic Foundations of Differential Privacy. *Foundations and Trends in TCS*, 2014.

[17] DeepSeek-AI. DeepSeek-V3 Technical Report. 2025.

[18] Y. Leviathan et al. Fast Inference from Transformers via Speculative Decoding. *ICML*, 2023.
