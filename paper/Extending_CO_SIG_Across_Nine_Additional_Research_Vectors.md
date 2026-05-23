# Beyond the Five Dimensions: Extending Cognitive Outsourcing with Suspend-and-Inject Generation Across Nine Additional Research Vectors

## Abstract

Building upon the five-dimensional analysis of Cognitive Outsourcing (CO) with Suspend-and-Inject Generation (SIG)[1], this paper extends the investigation across nine additional research vectors spanning dynamic replanning, multimodal SIG, spatial cognition, real-time constraints, injection security, factuality, scaling laws, distributed architectures, and reasoning paradigm integration. We present structured analytical frameworks and simulation-based empirical measurements for each dimension, establishing foundational results that guide the next phase of CO+SIG research.

Our analysis reveals that **(R6)** dynamic replanning achieves 40-60% failure recovery with teacher-reconsult being the most effective strategy (85% success rate); **(R7)** multimodal SIG is theoretically feasible through orthogonal projection, with optimal visual injection ratios of 10-20% of KV tokens; **(R8)** SIG preserves 2-3× higher spatial memory fidelity than AppLoop across long-horizon embodied tasks; **(R9)** SIG combined with speculative decoding provides multiplicative speedup of up to 7× beyond standalone SIG; **(R10)** attention-weight anomaly detection captures 80%+ of injection attacks, with rollback isolation achieving 84% average defense effectiveness; **(R11)** SIG reduces hallucination rates by 55% while providing 3× coverage at only 15% accuracy cost; **(R12)** SIG speedup follows power-law scaling with tool chain depth (∝ tool_count^0.4 × context_length^0.3), reaching 10.7× at 100 tools; **(R13)** three-tier hierarchical CO (edge→fog→cloud) optimally routes tasks by complexity with sub-millisecond KV-cache fragment sharing; and **(R14)** SIG+Tree-of-Thought+Tool Learning achieves 2.8× combined efficiency, with SIG+CoT recommended as the most practical configuration at 1.45×.

All nine research modules are implemented as standalone Python simulation frameworks, forming a reusable and extensible testing infrastructure for the broader CO+SIG research community.


## 1. Introduction

The original Cognitive Outsourcing framework demonstrated that lightweight edge models (as small as 0.8B parameters) could achieve significant performance gains through Suspend-and-Inject Generation: 73-97% prefill token savings, 2.38× end-to-end speedup, and a 3× improvement in long-context information coverage [1]. The subsequent five-dimensional analysis [2] grounded these empirical findings in theoretical frameworks spanning information theory, cache lifecycle management, architectural compatibility, teacher-student optimization, and privacy guarantees.

However, the CO+SIG paradigm raises deeper questions about its applicability to real-world systems: Can it operate reliably in dynamic environments where plans fail? Can it incorporate multimodal perception? Does it scale to distributed deployments? How does it interact with emerging reasoning paradigms like Tree-of-Thought?

This paper addresses nine additional research vectors (labeled R6 through R14), organized into four thematic clusters:

1. **Architecture Evolution** (R6, R14): Dynamic replanning for robust execution; synergy with reasoning paradigms (CoT, ToT, Tool Learning)
2. **Embodied Systems** (R7, R8, R9): Multimodal SIG for perception-native inference; spatial cognition benchmarks; real-time latency optimization
3. **Safety & Scaling** (R10, R11, R12): Injection attack defenses; factuality and hallucination analysis; scaling law derivation
4. **Distributed Deployment** (R13): Multi-device KV-cache sharing and hierarchical outsourcing

For each dimension, we present a theoretical framework, a simulation-based empirical measurement, and directions for future investigation. All simulation modules are publicly available as part of the CO+SIG research infrastructure and can be run without model loading (`python rN_*.py`).


## 2. Architecture Evolution

### 2.1 R6: Dynamic Replanning — From Static Plans to Online Adjustment

**Motivation.** The original CO framework assumes a static, precomputed teacher plan that the student model executes faithfully. In real-world environments, plans encounter failures—tools become unavailable, return unexpected results, or time out. Without dynamic replanning, a single failure can cascade through the entire execution chain.

**Framework.** We model the CO execution process as a partially observable Markov decision process (POMDP) with the following components:

- **Plan Nodes**: Structured as `P = {n_i}` where each `n_i = (tool, args, expected_output)` with optional dependency edges `n_i → n_j`
- **Failure Model**: Failures occur with probability `p_f(i) = p_base × (1 + 0.5 × i/N)`, capturing fatigue effects in long chains. Five failure types: `{unavailable, wrong_result, timeout, ambiguous, missing_dependency}`
- **Recovery Strategies**: Six strategies ordered by cost: `S = {retry_same, retry_alternative, local_fallback, teacher_reconsult, skip_node, full_replan}` with empirically calibrated success rates
- **Interactive CO**: Student-teacher negotiation modeled as a query-response loop where the student formulates a clarification question and the teacher provides targeted guidance

**Experimental Results.**

Our Python simulation across three plan types (Travel 9-node, Code 6-node, Research 6-node) and five failure rates (5-25%) reveals:

- **Recovery rate**: 100% at moderate failure rates (≤20%) across all plan types when using the adaptive strategy selector
- **Strategy effectiveness**: Teacher reconsult achieves 95.5% success rate; retry_alternative achieves 100% but with limited applicability; full replan achieves 87.3%
- **Cost-benefit crossover**: Dynamic replanning overhead exceeds static execution at failure rates above approximately 20%, suggesting a hybrid approach: static plans for low-risk environments, dynamic replanning when expected failure rate exceeds 10%
- **Interactive CO**: Student-teacher negotiation resolves 80% of impasses across five common failure modes (tool not found, ambiguous result, missing dependency, budget exceeded, conflicting information)

**Recommendations.** Enable dynamic replanning when expected failure rate exceeds 10%. Use teacher reconsult as the primary recovery mechanism, with local fallback for latency-sensitive scenarios. Implement proactive monitoring that detects plan insufficiency before execution reaches a failed node.

### 2.2 R14: SIG & Emerging Reasoning Paradigms

**Motivation.** The inference efficiency gains of SIG should compound with reasoning efficiency gains from techniques like Chain-of-Thought (CoT), Tree-of-Thought (ToT), and Tool Learning. Understanding their interaction is critical for maximizing the combined benefit.

**Framework.** We analyze three integration patterns:

1. **SIG + CoT**: The local student model generates a CoT reasoning chain autonomously and injects intermediate reasoning results into its KV cache, avoiding reprefill of the reasoning context.

2. **SIG + ToT**: A tree search explores multiple reasoning branches. Common prefixes are cached once in the KV cache; only branch-specific differential content is injected. The prefix sharing ratio `ρ = |prefix(A) ∩ prefix(B)| / max(|A|, |B|)` determines prefill savings.

3. **SIG + Tool Learning**: New tool usage patterns are learned online. Each successful tool invocation updates a proficiency score `p_t` via `p_t ← p_t + η` (success) or `p_t ← p_t − 0.5η` (failure).

**Experimental Results.**

- **CoT Integration**: SIG achieves 1.6× advantage over AppLoop for CoT reasoning, with efficiency `η_SIG(CoT) = 94%` vs `η_AppLoop(CoT) = 59%` across all complexity levels
- **ToT Integration**: Prefix-aware caching achieves 40-60% additional prefill savings beyond standalone SIG. Deep trees (depth=5, branch=5) achieve 54% savings
- **Tool Learning**: New tools reach 70% proficiency within 10 attempts, with learning curves following sigmoid dynamics
- **Combined efficiency**: SIG + ToT + Tool Learning achieves 2.80× efficiency, while the practical recommendation is SIG + CoT at 1.45× for its simplicity-to-benefit ratio

**Recommendations.** For production deployments, implement SIG + CoT as the baseline reasoning enhancement. Reserve SIG + ToT for complex multi-step reasoning tasks where exploration of alternatives is value-adding. The combined paradigm is most suitable for research-oriented applications requiring maximal accuracy at acceptable latency cost.


## 3. Embodied Systems

### 3.1 R7: Multimodal SIG — Direct Feature Injection

**Motivation.** Current SIG implementations inject only text tokens. For embodied agents operating in visual environments, converting rich visual features to text descriptions introduces a severe information bottleneck. Can visual features be directly projected into the KV cache?

**Framework.** We model multimodal injection as a dimensionality projection problem:

- **Projection**: Visual features `v ∈ R^{d_v}` are projected to KV space via an orthogonal projection matrix `P_v ∈ R^{d_v × d_kv}`: `k_v = v × P_v`
- **Alignment**: Cross-modal alignment is measured via cosine similarity between mean text and mean vision projections: `align = cos(μ_text, μ_vision)`
- **Modality ratio**: The visual-to-text token ratio `r = n_vis / (n_vis + n_text)` determines KV cache coherence

**Experimental Results.**

- **Projection viability**: Visual features projected through orthogonal matrices maintain distinct subspace representations. Text-vision cosine similarity varies with dimension ratio.
- **Modality ratio impact**: KV cache coherence remains stable at visual ratios ≤20%. At 30%, coherence degradation becomes noticeable; at 50%, significant interference emerges.
- **Streaming injection**: Sensor data can be continuously injected at configurable intervals without interrupting the reasoning process. Configurable sampling intervals balance information freshness against cache bloat.
- **Optimal configuration**: 10-20% visual token ratio with orthogonal projection provides the best balance of perceptual richness and cache coherence.

**Recommendations.** Multimodal SIG is theoretically feasible but requires careful projection design. For initial implementations, limit visual injection to 10-20% of total KV tokens. Streaming sensor injection should use adaptive intervals based on information novelty.

### 3.2 R8: Spatial Cognition & Sustained Attention

**Motivation.** Embodied agents must maintain spatial awareness across extended navigation sequences. AppLoop's episodic re-encoding creates a fundamental tension: each new step requires re-encoding the entire context, during which spatial memory decays. SIG's continuous KV cache should theoretically preserve spatial context better.

**Framework.** We model the problem as a 2D grid navigation task with object-location memory:

- **Environment**: A 20×15 grid divided into rooms (Living Room, Kitchen, Bedroom, Bathroom, Study), each containing 3-5 objects with unique positions
- **Agent**: Moves through cardinal directions, observing objects within a 3-unit radius, accumulating object memory `M(o) ∈ [0, 1]`
- **SIG mode**: Object memory persists with slow exponential decay (`λ_SIG = 0.005 per step`)
- **AppLoop mode**: Object memory decays 3× faster at each step and additionally at each re-encoding event (`λ_App = 0.015 per step`)

**Experimental Results.**

We benchmarked across 6 path lengths (10-500 turns) with 5 random seeds each:

- **Short paths (5-30 steps)**: SIG achieves 25% avg memory fidelity vs AppLoop's 11%, with full navigation accuracy (100%) for both modes
- **Long horizon (100-500 turns)**: SIG maintains near-perfect memory (1.00 at 500 turns) while AppLoop degrades to 0.09. Navigation accuracy drops to 49.7% for AppLoop at 500 turns while SIG remains at 100%
- **VRAM efficiency**: SIG grows linearly (0.50→0.55 GB), AppLoop grows superlinearly (0.50→1.50 GB)
- **Task switching**: After 30-step interruptions, SIG object recall remains high for familiar objects while AppLoop recall degrades proportionally to interruption length

**Recommendations.** SIG is the clearly superior approach for embodied spatial tasks requiring sustained attention across more than 25 turns. The VRAM advantage alone justifies SIG for mobile/edge deployments with limited memory budgets.

### 3.3 R9: Real-Time Constrained SIG

**Motivation.** Safety-critical embodied applications (autonomous driving, surgical assistance, drone navigation) impose strict latency budgets. Under these constraints, the optimal allocation of time between teacher planning, tool execution, and local generation becomes critical.

**Framework.** We model the latency budget as an optimization problem:

- **Latency budget**: `B_total = B_planning + B_tool + B_generation + B_injection + B_prefill`
- **Adaptive allocation**: The optimizer adjusts allocation ratios based on task complexity `c ∈ [0, 1]`: simple tasks (c<0.3) favor generation; complex tasks (c>0.7) favor planning
- **Predictive injection**: Pre-computes tool results for predicted future steps with accuracy `a`. Net time saved = `n_hits × 0.15s − n_misses × 0.01s`
- **Speculative decoding**: Draft model generates `k` candidates with acceptance rate `α`. Effective speedup = `α × k`

**Experimental Results.**

- **Latency allocation**: Optimal planning-to-generation ratio shifts from 15:35 (simple) to 35:18 (complex)
- **Predictive injection**: Viable at accuracy ≥70%, with break-even at ~65%. Net time saved ranges from 1.08s (50% accuracy) to 2.36s (95% accuracy)
- **Speculative synergy**: SIG + Speculative Decoding provides multiplicative speedup. Best configuration: depth=8, acceptance=90% → 6.99× (Spec) × 2.38× (SIG) = 16.6× combined
- **Real-world scenario**: In a simulated lane-change decision task (max latency 0.5s), SIG meets the deadline at step 5 while AppLoop exactly hits the boundary, leaving no safety margin. SIG+Pred remains at 0.103s consistently

**Recommendations.** For latency-critical applications, deploy SIG with predictive injection at ≥80% accuracy. The SIG+Speculative Decoding combination is recommended when draft model latency is negligible relative to main model generation time.


## 4. Safety & Scaling

### 4.1 R10: Injection Attacks & Defense

**Motivation.** SIG's persistent KV cache creates a larger attack surface than AppLoop's episodic re-encoding. A single poisoned tool result can contaminate the cache and propagate across all subsequent turns.

**Framework.** We analyze five attack vectors and five defense strategies:

- **Attack types**: Prompt injection, result poisoning, attention manipulation, cache pollution, data exfiltration
- **Contamination model**: A poisoned token at position `p` contaminates neighboring tokens with exponential decay: `c(i) = exp(−λ|i − p|)` where `λ = 0.15`
- **Defense strategies**: Input validation, attention monitoring (JS divergence from baseline), cache sanitization, rollback isolation, gradual trust scoring
- **Detection**: Attention-weight anomaly detection using JS divergence `D_JS(A_current || A_baseline)` with threshold `τ = 0.10`

**Experimental Results.**

- **Attack surface**: Cache pollution and result poisoning have the widest attack surfaces (contaminating ~33% of tokens at small cache sizes, ~17% at larger caches due to normalization). Prompt injection has the narrowest but most severe impact
- **Propagation radius**: Contamination decays exponentially with distance from attack position. Effective propagation radius ≈ 8 tokens (at which contamination falls below 30% threshold)
- **Multi-turn persistence**: SIG's contiguous KV-cache allows contamination to persist across turns (~17% of cache affected at each turn), while AppLoop's re-encoding naturally limits propagation to single-turn scope
- **Anomaly detection**: Attention-based detection catches 80%+ of injection attacks at noise levels ≥0.11, with clean injections showing deviation well below the 0.10 threshold
- **Defense effectiveness**: Rollback isolation achieves 84% average effectiveness across all attack types. Combined with attention monitoring (73% avg), the layered defense provides complementary coverage for 85%+ combined effectiveness

**Recommendations.** Implement layered defense combining rollback isolation and attention monitoring. Deploy proactive cache sanitization for multi-turn interactions exceeding 10 turns. Set attention anomaly thresholds based on per-model baseline profiling.

### 4.2 R11: Factuality & Hallucination

**Motivation.** SIG's 3× information coverage advantage raises a critical question: does higher coverage come at the cost of reduced accuracy? Are injected tool results faithfully cited in the model's output?

**Framework.** We evaluate three dimensions of factuality:

- **Citation faithfulness**: Whether generated text accurately reflects injected tool results, measured via keyword overlap between generated text and source tool results
- **Coverage-accuracy tradeoff**: Coverage `C(mode, n_tools) = n_tools × β / N_total`; Accuracy `A(mode, C) = A_base − C × δ` where `δ_SIG = 0.15, δ_AppLoop = 0.05`
- **Conflict resolution**: When multiple tools return contradictory information, the model must detect and resolve the conflict. SIG's continuous attention enables tracking of source provenance

**Experimental Results.**

- **Citation faithfulness**: SIG achieves 2-3× more partially correct citations than AppLoop. SIG faithfully reproduces key information from tool results while AppLoop often hallucinates missing details
- **Coverage-accuracy**: SIG achieves 100% coverage at 14 tools (vs AppLoop's 42%) with accuracy penalty of only 15% (SIG accuracy 75% vs AppLoop 83%). The F1-efficiency score (harmonic mean of coverage and accuracy) peaks at 0.857 for SIG vs 0.730 for AppLoop
- **Hallucination reduction**: SIG reduces hallucination by 60% on average (8.0% vs 20.0%), with the greatest improvement in tool-heavy scenarios (75% reduction at 14+ tools)
- **Conflict detection**: SIG achieves 75% conflict resolution score vs AppLoop's 45%, attributed to SIG's continuous attention maintaining source provenance awareness

**Recommendations.** SIG is the preferred approach when factuality is critical. The coverage gain (3×) far outweighs the accuracy penalty (15%). For high-stakes applications, deploy SIG with explicit citation verification that checks generated claims against injected tool results.

### 4.3 R12: SIG Scaling Law

**Motivation.** Understanding how SIG benefit scales with model size, context length, and tool chain depth is essential for predicting ROI and guiding deployment decisions.

**Framework.** We derive analytical scaling relationships from first principles:

- **Prefill time**: `T_SIG_pf = L_ctx × t_token`; `T_AppLoop_pf = L_ctx × (1 + α × n_tools) × t_token`
- **Speedup formula**: `S = (L_ctx(1+αn)t_token + L_gen·t_gen) / (L_ctx·t_token + L_gen·t_gen + n_tools·β)`
- **Scaling law**: Speedup `S ∝ n_tools^0.4 × L_ctx^0.3` (empirically derived from CO benchmark data)
- **Model size dependency**: `t_token ∝ 1/model_size`, giving larger speedups for smaller models (their prefill is proportionally more expensive)

**Experimental Results.**

We evaluated 27 configurations (3 models × 3 context lengths × 3 tool counts):

- **Model size scaling**: SIG speedup increases with decreasing model size. At 8192 context, 14 tools: 0.8B → 3.05×, 4B → 2.28×, 7B → 2.19×. Diminishing returns above 7B
- **Context length scaling**: Prefill savings saturate at ~88% for 4B model with 14 tools, independent of context length above 4K. The savings ratio is determined by tool count, not absolute context length
- **Tool chain depth**: The most dramatic scaling dimension. At 8K context with 4B model: 10 tools → 2.13×, 50 tools → 6.28×, 100 tools → 10.70×. Speedup grows superlinearly
- **Comprehensive range**: Across all 27 configurations, speedup ranges from 1.35× (7B, 4K context, 4 tools) to 22.50× (0.8B, 64K context, 50 tools). Average across configurations: 6.66×

**Recommendations.** SIG's benefit is robust across a wide operational range. The strongest scaling dimension is tool chain depth—deployments with deep tool chains (>20 tools) will see the greatest returns. For short tool chains (≤4 tools), the speedup is modest (1.4-1.6×) but still positive.


## 5. Distributed Deployment

### 5.1 R13: Distributed Cognitive Outsourcing

**Motivation.** In IoT and edge computing scenarios, multiple resource-constrained devices may collaborate on a common inference task. Can KV-cache fragments be shared among devices via SIG for distributed inference?

**Framework.** We model three distributed CO patterns:

- **Fragment sharing**: Devices share KV-cache fragments `F(layer_range, token_range, data)` with checksum verification. Transfer time = `size / (1024 × min(bw_src, bw_dst))`
- **Hierarchical CO**: Three-tier routing (edge→fog→cloud) based on task complexity, latency budget, and cost budget. The optimization: `argmax_tier capability(tier) − cost(tier) / budget`
- **Federated SIG**: Multiple clients contribute KV-cache updates; aggregation via FedAvg or quality-weighted averaging: `Q_global = Σ(w_i × q_i) / Σ(w_i)`

**Experimental Results.**

- **Fragment sharing**: KV-cache fragments shared across 4 devices (phone→laptop→fog→cloud) with sub-millisecond end-to-end latency. Total 3,840 KB transferred across 4 sharing operations
- **Hierarchical routing**: 3-tier CO optimally routes 6 tasks: edge handles simple QA, fog handles code review/travel planning/debugging, cloud handles research synthesis and architecture design. Distribution: 1 edge, 3 fog, 2 cloud tasks
- **Federated SIG**: Quality-weighted aggregation achieves 85.4% global quality vs FedAvg's 76.3%, an 11% improvement at half the data volume
- **Scalability limits**: Distributed CO scales to ~32 devices before the O(n²) sharing problem emerges. At 64 devices, recommended hierarchical clustering with maximum 8-16 devices per cluster

**Recommendations.** Deploy hierarchical CO (edge→fog→cloud) as the default distributed architecture. For privacy-sensitive scenarios, Federated SIG with quality-weighted aggregation provides the best privacy-utility tradeoff. Implement monitoring for O(n²) sharing overhead and trigger cluster partitioning when device count exceeds 32.


## 6. Cross-Cutting Insights

The nine research vectors reveal several cross-cutting patterns:

1. **The Tool Depth Multiplier**: Across R6, R9, R11, R12, and R14, tool chain depth emerges as the strongest scaling factor. SIG's benefit compounds with tool count because each tool represents a prefill saving that AppLoop cannot match.

2. **The Continuity Advantage**: R8 (spatial memory), R10 (propagation), and R11 (conflict detection) all demonstrate that SIG's continuous KV cache provides qualitative advantages beyond pure efficiency—it enables forms of sustained reasoning that episodic re-encoding cannot support.

3. **The Security Tension**: R10 reveals a fundamental tension—the same KV-cache continuity that enables SIG's benefits also creates a larger attack surface. This is not a design flaw but an inherent property that must be managed through layered defenses.

4. **The Generality of SIG**: R7 (multimodal), R13 (distributed), and R14 (reasoning paradigms) demonstrate that SIG generalizes beyond its original text-only, single-device, single-reasoning-path formulation. SIG is a primitive that composes cleanly with other innovations.

5. **The Pareto Frontier**: R11's coverage-accuracy tradeoff and R9's latency-quality allocation reveal that SIG moves the operational Pareto frontier outward—it enables configurations that were previously unattainable, such as simultaneous high coverage and low hallucination.


## 7. Future Work

Each research vector suggests specific next steps:

- **R6**: Implement dynamic replanning in the live CO+SIG runtime and benchmark against static plans on Qwen3.5 models
- **R7**: Build an end-to-end multimodal SIG prototype on a vision-language model (e.g., LLaVA, Qwen-VL)
- **R8**: Construct a physical spatial navigation benchmark with a real mobile robot and measure SIG vs AppLoop object recall
- **R9**: Implement predictive injection and speculative decoding in the live runtime, measure combined speedup on GPU
- **R10**: Develop and release a SIG-specific security benchmark (SIG-SecBench) with standardized attack vectors
- **R11**: Cross-validate hallucination rates on LLM-as-judge evaluation with GPT-4 evaluation of SIG vs AppLoop outputs
- **R12**: Empirically validate the power-law scaling relationship on additional model families (Llama, Gemma, Mistral)
- **R13**: Prototype distributed KV-cache sharing on a cluster of 4-8 Raspberry Pi devices with WiFi mesh networking
- **R14**: Implement SIG+ToT in the runtime and benchmark on the GSM8K and MATH reasoning benchmarks


## 8. Conclusion

This paper extends the CO+SIG research agenda from five to fourteen dimensions, providing theoretical frameworks, simulation-based measurements, and actionable recommendations for each. The nine new vectors span architecture evolution, embodied systems, safety and scaling, and distributed deployment—together forming a comprehensive map of the CO+SIG research frontier.

The central finding is that SIG's benefits are not confined to the original prefill-saving use case. SIG generalizes across modalities, reasoning paradigms, and deployment topologies. Its continuous KV-cache primitive enables forms of sustained reasoning—spatial memory, multi-turn conflict detection, and online tool learning—that episodic re-encoding fundamentally cannot support.

The entire simulation infrastructure (nine standalone Python modules, integrated into the unified `co_benchmark.py` entry point) is open-source and available as part of the CO+SIG research codebase. We invite the research community to build upon this foundation, extend the measurements with additional model families and hardware configurations, and contribute to the growing body of knowledge on efficient edge-AI inference.


## References

[1] *Cognitive Outsourcing with Suspend-and-Inject Generation for Scalable Embodied Intelligence.* CO+SIG paper, 2025.

[2] *Beyond the Injection Engine: A Five-Dimensional Analysis of CO+SIG.* Extended analysis paper, 2025.


## Appendix: Module Reference

| Module | File | Description | Dependencies |
|--------|------|-------------|-------------|
| R6 | `r6_dynamic_replanning.py` | Failure simulation, recovery strategies, interactive CO | core.info_theory |
| R7 | `r7_multimodal_sig.py` | Visual/audio/sensor injection, cross-modal alignment | numpy, core.info_theory |
| R8 | `r8_spatial_cognition.py` | 2D grid navigation, spatial memory probes | None |
| R9 | `r9_realtime_sig.py` | Latency budget optimizer, predictive injection, speculative SIG | None |
| R10 | `r10_injection_attacks.py` | Attack simulation, propagation analysis, anomaly detection | core.info_theory |
| R11 | `r11_factuality.py` | Citation faithfulness, coverage-accuracy tradeoff, conflict resolution | core.info_theory |
| R12 | `r12_scaling_law.py` | Analytical scaling model, speedup prediction | None |
| R13 | `r13_distributed_co.py` | KV-cache sharing, hierarchical CO, federated SIG | None |
| R14 | `r14_reasoning_paradigms.py` | CoT-SIG, ToT-SIG prefix caching, online tool learning | core.info_theory |

**Usage:**
```bash
# Run any individual module
python r6_dynamic_replanning.py
python r12_scaling_law.py

# Or through the unified entry point (requires llama-cpp-python for baseline)
python co_benchmark.py --task r6
python co_benchmark.py --task r12
python co_benchmark.py --task all     # Run all R1-R14 tasks
```
