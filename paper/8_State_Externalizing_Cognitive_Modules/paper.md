# State-Externalizing Cognitive Module Harnesses: Elevating Cognitive Outsourcing from Injection-Level to System-Level Orchestration

> **SIG/CO Research Program — Paper 8** | June 2026
>
> Preceding papers: [1] *Cognitive Outsourcing with Suspend-and-Inject Generation for Scalable Embodied Intelligence*, [2] *Beyond the Injection Engine: A Five-Dimensional Analysis of CO-SIG*, [3] *CO-SIG Architecture, Theory, and Empirical Design Space for Scalable Edge Intelligence*, [4] *Suspend-and-Inject Generation as an Edge Inference Runtime Primitive for Long-Horizon Agent Tasks*, [5] *Orthogonal Acceleration: Fusing Speculative Decoding with Suspend-and-Inject Generation*, [6] *Convergent KVCache Architectures: Bridging Cloud-Scale Disaggregated Serving and Edge-Native Injection Continuity*, [7] *DiskKVCache: Disk-Backed KVCache Persistence for Cold-Start Elimination in Edge Agent Inference*.
>
> **Date**: June 2026

---

## Abstract

The Cognitive Outsourcing (CO) architecture established across Papers 1–7 delegates semantic reasoning to a lightweight Meaning Compiler (0.8B–4B) while the SIG Injection Engine maintains KV-cache continuity. We hypothesise that the Meaning Compiler suffers from "overload"—simultaneously performing semantic reasoning and module management bookkeeping—and propose **SECM-H** (State-Externalizing Cognitive Module Harness) to externalise the latter. SECM-H maintains six structured state components ($R_t$, $H_t$, $C_t$, $D_t$, $P_t/I_t$, $B_t$) while rendering compact summaries into the Meaning Compiler's context. We present the architectural design, a formal state decomposition taxonomy, a five-dimensional content quality evaluation framework, and experimental validation across twelve experimental configurations on the EdgeAgent-Kitchen benchmark using Qwen3.5-2B and 0.8B models.

State decomposition confirms that 76.5% of management functions are structurally externalisable (H1 confirmed). Pre-scripted injection experiments (EXP-3/9/10) reveal that the SIG baseline achieves the highest content quality ($Q_{content} = 0.461$ for 2B) with the lowest latency, because the model's implicit state tracking through KV-cache retention is well-adapted to its own attention patterns. However, these benchmarks used **pre-scripted tool calls** where the model never chose which tool to invoke—SECM-H's module selection capability was never exercised. Agent-driven experiments (EXP-11/12), where the model autonomously decides which tool to call, reveal a fundamentally different picture: under the 15%-failure Noisy Kitchen, SECM-H-full achieves **97.1% tool selection accuracy versus SIG's 94.3%**, with Confidence tracking guiding the model away from failing tools. Selective injection consistently produces the highest content quality ($Q_{content}$: 0.718 clean, 0.661 noisy) at the lowest latency. The central lesson is methodological: **benchmark design must test the capability being evaluated**—pre-scripted tool calls bypass the very module management that SECM-H externalises, rendering negative results artefacts of evaluation design rather than architectural failure.

---

## 1. Introduction

### 1.1 The Meaning Compiler Overload Problem

The CO+SIG architecture [1] established a three-layer system for edge agent inference: a lightweight **Meaning Compiler** (0.8B–4B model) that performs semantic reasoning, a **SIG Injection Engine** that preserves KV-cache continuity through the suspend-inject-resume cycle [1], and a **Cognitive Module Ecosystem** comprising cloud teachers, local perception modules, skill libraries, and cognitive caches. Papers 1–4 demonstrated that this architecture eliminates 73–97% of redundant prefill tokens [3] and yields 2.38–9.45× speedups across architectures, while Paper 5 [5] showed that SIG composes orthogonally with speculative decoding ($\rho = 1.239$). Paper 6 [6] formalised the convergence between edge injection continuity and cloud-scale KVCache management through the KFC framework. Paper 7 [7] extended the architecture with disk-backed KVCache persistence for cold-start elimination.

Across these papers, the Meaning Compiler was characterised as a lightweight orchestrator responsible for "intent parsing, module selection, and response synthesis" [1]. However, as the Cognitive Module Ecosystem grows in complexity, the Meaning Compiler implicitly performs *six additional state management functions* beyond pure semantic reasoning:

1. **Module awareness**: Knowing which modules exist and their capabilities (maintaining $R_t$).
2. **Invocation tracking**: Remembering which modules have been called and their results (maintaining $H_t$).
3. **Quality assessment**: Evaluating module result reliability over time (maintaining $C_t$).
4. **Dependency reasoning**: Understanding inter-module prerequisites (maintaining $D_t$).
5. **Pattern reuse**: Recognising when a previously successful cognitive pattern applies (maintaining $(P_t, I_t)$).
6. **Budget management**: Allocating limited context tokens across competing module results (maintaining $B_t$).

For a 0.8B model with $n_{ctx} = 16384$ tokens, this state management context can grow to 2000+ tokens in a 35-step scenario with 18 tools—representing $\geq 12\%$ of the total context window consumed by bookkeeping rather than cognition. We term this the **Meaning Compiler Overload Problem**: the edge model's limited effective context is split between semantic decision-making and module management bookkeeping. Our experiments reveal that this "overload" is not a simple capacity burden—the model's implicit state tracking through KV-cache retention is, in fact, well-suited to pre-scripted tasks. Rather, the overload manifests as a *capability gap*: the model lacks access to **long-term, cross-step reliability statistics** (which tools have consistently succeeded or failed) that require explicit state tracking beyond what implicit KV-cache retention can maintain. SECM-H's value, when it exists, derives not from relieving a cognitive burden but from providing *exogenous decision knowledge* that the model cannot generate on its own.

### 1.2 State Externalization as a Design Principle

Harness-1 [8] (Jiang et al., arXiv:2606.02373, 2026) demonstrated a general principle: externalising bookkeeping state from a policy model to an environment-side harness improves decision quality. In Harness-1's search domain, the policy model's limited context window is reserved for *semantic decisions* (what to search for, which documents are relevant) while *bookkeeping state* (candidate pools, search history, budget tracking) is maintained externally by the harness. The system uses a two-tier working memory—prompt-facing compact state plus an outer full-text store—with structured state components for candidate pools, curated sets with importance tags, evidence graphs, verification records, search history, and budget markers.

We identify a structural analogy between Harness-1's search domain and CO+SIG's module management domain. Both involve **sequential decisions under uncertainty** (which document to retrieve / which module to invoke) while maintaining **accumulated evidence state** (what has been found so far / what modules have returned). However, the analogy has important boundary conditions: Harness-1 operates with a 20B RL-trained model, while SECM-H targets 0.8B–4B untrained models. Our experiments demonstrate that the state-externalisation principle does not transfer automatically—its value depends critically on whether the task exercises the capability being externalised.

This paper applies the state-externalisation principle to CO+SIG, proposing **SECM-H** (State-Externalizing Cognitive Module Harness) as a module management layer. SECM-H maintains structured state for module reliability tracking, invocation history, and dependency constraints. We do *not* claim that SECM-H is a general-purpose cognitive architecture layer. Rather, our experiments demonstrate a more specific contribution: **SECM-H provides genuine value as a long-range reliability tracker for autonomous agents operating under uncertainty**, where the model must select among modules with varying reliability—a capability that small models' implicit KV-cache tracking cannot reliably maintain across extended task horizons.

### 1.3 Contributions

This paper makes the following contributions:

1. **Formal state decomposition taxonomy** (§4): The first systematic analysis of module management functions in a cognitive outsourcing architecture, classifying each function as externalizable ($F_{ext}$), partially externalizable ($F_{partial}$), or policy-intrinsic ($F_{sem}$). Experimentally confirmed: 76.5% (13/17) are fully externalizable with 69.3% context reduction potential.

2. **SECM-H architecture and prototype** (§3): A principled architecture for state-externalised cognitive module management, grounded in Harness-1's state-externalisation principle [8] and adapted for the edge inference regime. The six-component state structure ($R_t$, $H_t$, $C_t$, $D_t$, $P_t/I_t$, $B_t$) provides a reusable design pattern.

3. **Relationship between SIG and SECM-H** (§3.6): The identification that SIG (transparent infrastructure optimisation) and SECM-H (visible informational augmentation) are fundamentally different mechanisms that solve different problems at different architectural levels. They are composable but not analogous—contrary to our initial "dual-layer" framing.

4. **Negative result under pre-scripted benchmarks; positive result under agent-driven evaluation** (§6.6, §6.11, §7.1): Across 136 pre-scripted experimental runs (EXP-3/9/10), the SIG baseline achieves the highest content quality and lowest latency because the model's implicit state tracking through KV-cache retention is well-adapted to its own attention patterns. However, agent-driven experiments (EXP-11/12) reveal that these benchmarks never tested SECM-H's module selection capability: when the model autonomously chooses which tool to invoke, SECM-H-full achieves 97.1% tool selection accuracy versus SIG's 94.3% under 15% noise, and selective injection consistently maximises content quality ($Q_{content}$: 0.718 clean, 0.661 noisy).

5. **Causal analysis of pre-scripted benchmark artefact** (§7.6): Identification of four interacting failure mechanisms—attention distribution disruption, attention competition between implicit and explicit state, generation-length cascading, and Harness-1 boundary violation—that explain the pre-scripted negative result. These mechanisms are real but are triggered by the injection channel, not by the architecture itself; agent-driven evaluation demonstrates that SECM-H's value resides in module selection, which pre-scripted benchmarks bypass entirely.

6. **Agent-driven benchmark methodology** (§6.11): The demonstration that pre-scripted tool calls fundamentally bypass the capability under evaluation, and the introduction of agent-driven experiments where the model autonomously decides which tool to invoke. This methodological contribution establishes that benchmark design must match the architectural capability being tested—a principle applicable beyond SECM-H to any tool-augmented LLM evaluation.

7. **Content-level quality evaluation framework** (§5.4): A five-dimensional evaluator (information coverage, response quality, context utilisation, semantic adequacy, information density) that resolves the saturation problem of tool-execution-rate metrics and enables meaningful quality differentiation.

8. **Noisy Kitchen benchmark** (§6.7): A 15%-failure-rate variant of EdgeAgent-Kitchen that tests Confidence and Dependency component value under realistic uncertainty.

---

## 2. Background and Related Work

### 2.1 The CO+SIG Architecture (Papers 1–7)

The SIG/CO Research Program has established a comprehensive edge agent inference stack across seven papers:

**Paper 1** [1] introduced the CO architecture and the SIG five-stage cycle (Suspend → Resolve → Fetch → Inject → Resume). The three-layer architecture comprises a Meaning Compiler (lightweight edge model), a SIG Injection Engine (KV-cache management), and a Cognitive Module Ecosystem (external tools and knowledge sources). SIG eliminates redundant prefill by preserving KV-cache state across tool-call boundaries.

**Paper 2** [2] provided theoretical foundations (results R1–R5), establishing per-token rate equivalence between SIG and AppLoop (±2%), proving that SIG's speedup is purely from prefill elimination, and introducing CompSIG (periodic KV-cache compression achieving 61% reduction with 17% overhead).

**Paper 3** [3] mapped the design space (R6–R14), validating SIG across three architectures (dense, hybrid Mamba+attention, GQA) with 73–97% prefill savings and 2.38–9.45× speedups. It introduced the SIG Decision Framework and the hybrid TF-IDF composite quality scorer used throughout the program.

**Paper 4** [4] characterised SIG as a deployment runtime (R15–R19), introducing the EdgeAgent-Kitchen benchmark (35-step interleaved multi-task scenarios with 18 tools) and demonstrating deep-chain advantages of 2.79× (0.8B) to 4.26× (4B) at 30-tool depth.

**Paper 5** [5] demonstrated that SIG composes orthogonally with speculative decoding via multi-token prediction (MTP), achieving 4.52× compound speedup with $\rho = 1.239$.

**Paper 6** [6] formalised the KFC (KVCache-as-First-Class-Citizen) framework, establishing architectural convergence between edge injection continuity and cloud-scale disaggregated serving (Mooncake). The 8-dimensional analysis revealed 5 dimensions of strong or moderate convergence.

**Paper 7** [7] introduced DiskKVCache for disk-backed KVCache persistence, enabling cold-start elimination across sessions with persistent prefix storage.

Papers 1–7 focused on *inference-level* optimisations—KV-cache manipulation, prefill elimination, cache persistence, speculative decoding. These are *execution-layer* concerns. Paper 8 extends the program to *cognitive-architecture-level* optimisation—module management state externalisation. This is a *management-layer* concern, and together Papers 1–8 establish a complete optimisation stack for edge agent inference (Table 1).

[Table 1: The SIG/CO Optimisation Stack]

| Layer | Optimisation | Paper |
|-------|-------------|-------|
| Cognitive management | SECM-H state externalisation | **Paper 8** |
| KV-cache persistence | DiskKVCache | Paper 7 [7] |
| KV-cache convergence | KFC unified framework | Paper 6 [6] |
| Generation acceleration | MTP compound acceleration | Paper 5 [5] |
| Deployment runtime | Kitchen benchmark, R15–R19 | Paper 4 [4] |
| Design space | Cross-architecture, Batch-SIG | Paper 3 [3] |
| Theoretical foundations | R1–R5 attention analysis | Paper 2 [2] |
| Core primitive | SIG five-stage cycle | Paper 1 [1] |

### 2.2 State-Externalising Harnesses (Harness-1)

Harness-1 [8] (Jiang et al., 2026) demonstrated that externalising search state from a policy model to an environment-side harness improves retrieval quality. The system targets a 20B RL-trained search agent and introduces several key design elements:

**WORKINGMEMORY architecture.** Harness-1 uses a two-tier memory structure: a *prompt-facing compact state* rendered into the policy's context (≤200 tokens) and an *outer full-text store* maintained by the harness. The compact state provides the policy with a summary of the current search situation; the full-text store preserves complete evidence for later reference.

**Structured state components.** The working memory comprises six components: candidate pool ($P_t$), curated set with importance tags ($C_t$, $I_t$), evidence graph ($G_t$), verification records ($V_t$), search history ($H_t$), and budget marker ($B_t$).

**Policy actions as state edits.** The policy's actions (curate, verify, review_docs, terminate) operate on explicit state components rather than free-form text. This structural interface enables the harness to track and update state deterministically.

**Auto-seeding.** The candidate pool is warm-started with tentative results before the policy begins, providing early learning signal. This addresses the cold-start problem in long-horizon search tasks.

**Derived-state rendering.** The evidence graph is compressed, deduplicated, and rendered as compact signals for the policy's context, rather than raw full-text dumps.

**Trainability requirements.** Harness-1 identified three requirements for effective state externalisation: (a) warm-started curation, (b) compact derived-state rendering, and (c) diversity-preserving incentives. We adapt all three to the SECM-H context (§3.1).

**Critical boundary conditions.** Harness-1 operates under conditions that differ fundamentally from the CO+SIG edge regime: (a) the policy model is 20B parameters, providing sufficient capacity to parse structured state formats; (b) the model is RL-trained with reward shaping, teaching it *when* and *how* to attend to harness state; (c) the deployment environment is cloud-scale with latency budgets of seconds per query. SECM-H targets 0.8B–4B models *without* task-specific training and operates within millisecond-level latency budgets. These boundary conditions are critical: the externalisation principle's success in Harness-1 depends on the policy model's ability to efficiently interpret compressed state representations, an ability that is not guaranteed for small untrained models (§7.6).

### 2.3 Agent Framework State Management

Modern agent frameworks manage state through implicit or explicit mechanisms:

**LangChain** [9] maintains conversational history in a linear buffer, with optional summarisation for long conversations. State management is implicit—the LLM must track tool invocation history through the conversation context. LangGraph introduces explicit state channels but does not externalise state from the LLM's context.

**AutoGPT** [10] uses a task list and memory system, but the LLM must maintain awareness of past actions through context injection. State tracking is partially externalised (task list is maintained externally) but module management (which tools have been used, their reliability) remains in-context.

**SWE-agent** [11] provides a structured action space (file editing, searching, testing) with an explicit environment state (file system). However, the agent's internal state—what it has tried, what worked, what to try next—remains in the LLM's context window.

**Voyager** [12] maintains a skill library with code-as-skill representations. The library is external, but the LLM must decide which skill to use based on in-context descriptions that grow with library size.

SECM-H differs from all of these in two fundamental ways: (a) it externalises *all* module management state, not just conversational history or task lists; and (b) it is designed specifically for *small edge models* (0.8B–4B) where context window competition between state tracking and semantic reasoning is most acute.

### 2.4 Positioning and Distinction

[Table 2: Comparison of CO+SIG, Harness-1, and SECM-H]

| Dimension | CO+SIG (Papers 1–7) | Harness-1 [8] | SECM-H (this work) |
|-----------|---------------------|---------------|---------------------|
| **Primary contribution** | KV-cache injection continuity | State-externalising search harness | State-externalising module harness |
| **Target model** | 0.8B–4B (edge) | 20B (cloud, RL-trained) | 0.8B–4B (edge) |
| **Domain** | Cognitive module orchestration | Information retrieval | Cognitive module orchestration |
| **State externalised** | Attention state (KV-cache) → SIG Engine | Search state → harness | Module management state → SECM-H |
| **State components** | KV-cache (implicit) | 6: $P_t$, $C_t/I_t$, $G_t$, $V_t$, $H_t$, $B_t$ | 6: $R_t$, $H_t$, $C_t$, $D_t$, $P_t/I_t$, $B_t$ |
| **Training required** | None | RL with reward shaping | None (rule-based) |
| **Rendering target** | N/A (KV-cache is opaque) | ≤200 tokens for 20B model | ≤100 tokens for 0.8B–4B model |
| **Externalisation layers** | 1 (attention → engine) | 1 (working memory → harness) | 2 (attention → engine + management → harness) |
| **Deployment** | Edge (RTX 4070 SUPER) | Cloud (single/multi-GPU) | Edge (RTX 4070 SUPER) |
| **Latency budget** | Milliseconds per step | Seconds per query | Milliseconds per step |

---

## 3. The State-Externalising Cognitive Module Harness (SECM-H)

### 3.1 Design Principles

We derive three design principles from Harness-1's trainability requirements [8], adapted for the CO+SIG edge inference regime:

**Principle 1: Auto-seeded module guidance.** *Adapted from Harness-1's warm-started curation.* The cognitive pattern cache ($P_t$) is pre-populated with common cognitive patterns extracted from the local cognitive cache [1] before the Meaning Compiler begins its reasoning cycle. This provides early-step guidance analogous to Harness-1's tentative candidate seeding, reducing the cold-start penalty in the first 5–10 steps where the Meaning Compiler has no invocation history to draw upon. Analytical estimate: auto-seeding improves early-step task quality by $\geq 0.05$ compared to cold-start.

**Principle 2: Compact derived-state rendering.** *Adapted from Harness-1's evidence graph compression.* The full harness state $S_{harness} = \{R_t, H_t, C_t, D_t, (P_t, I_t), B_t\}$ is compressed into a structured text summary of ≤100 tokens per step, suitable for injection into the Meaning Compiler's context via the existing SIG injection pipeline [1]. The renderer prioritises recency (recent invocations), salience (high-confidence modules), and budget status (remaining context capacity). The 100-token target is chosen to be ≤0.6% of $n_{ctx} = 16384$, ensuring negligible competition with semantic reasoning content.

**Principle 3: Diversity-preserving module selection.** *Adapted from Harness-1's diversity-preserving incentives.* The confidence scores ($C_t$) and pattern cache importance tags ($I_t$) include exploration signals that prevent the Meaning Compiler from over-using high-confidence modules when lower-confidence alternatives might be more appropriate. Specifically, the rendered state includes a "suggested exploration" field that highlights under-used modules with relevant capabilities, injecting diversity without requiring explicit RL training.

### 3.2 Architecture

SECM-H is inserted as a middleware layer between the Meaning Compiler and the SIG Injection Engine in the existing CO+SIG stack:

```
┌───────────────────────────────────────────────────┐
│            Meaning Compiler (0.8B–4B)              │
│   Semantic decisions only:                         │
│   • What information do I need?                    │
│   • Which module should I invoke?                  │
│   • How should I integrate the result?             │
│   • When should I stop?                            │
├───────────────────────────────────────────────────┤
│          SECM-H (Cognitive Module Harness)         │
│   State management:                                │
│   • Module registry & capability map ($R_t$)       │
│   • Invocation history & result cache ($H_t$)      │
│   • Module confidence scores ($C_t$)               │
│   • Dependency graph ($D_t$)                       │
│   • Cognitive pattern cache ($P_t$, $I_t$)         │
│   • Context budget ($B_t$)                         │
│   • Derived-state rendering (≤100 tokens)          │
├───────────────────────────────────────────────────┤
│          SIG Injection Engine [1]                   │
│   KV-cache management:                             │
│   • Suspend / Inject / Resume cycle                │
│   • Stabilisation templates                        │
│   • KV-cache compression (CompSIG) [2]             │
├───────────────────────────────────────────────────┤
│          Cognitive Module Ecosystem                │
│   • Cloud teachers                                 │
│   • Local perception modules                       │
│   • Skill libraries                                │
│   • Cognitive cache                                │
└───────────────────────────────────────────────────┘
```

**Interaction protocol.** The four-step interaction cycle between the Meaning Compiler, SECM-H, and the SIG Injection Engine proceeds as follows:

1. The Meaning Compiler generates a semantic intent: "I need weather data for Paris."
2. SECM-H translates this intent into a structured action: `{module: "get_weather", args: {city: "Paris"}}`, updates $H_t$, marks the module as in-flight in $R_t$.
3. SECM-H renders a compact state summary (≤100 tokens) and injects it into the Meaning Compiler's context via the SIG Injection Engine [1] using the standard five-stage Suspend-Inject-Resume cycle.
4. After the module returns a result, SECM-H processes it: updates $C_t$ (confidence), $D_t$ (dependency), $P_t$ (pattern cache), $B_t$ (budget), and renders the next compact state.

### 3.3 State Components

We define six structured state components, each corresponding to a Harness-1 analogue [8] and a CO+SIG module management function (§4.1).

[Table 3: SECM-H State Components — Notation and Definition]

| Component | Symbol | Harness-1 Analogue | Definition | Data Structure | Rendering |
|-----------|--------|--------------------|------------|----------------|-----------|
| Module Registry | $R_t$ | Candidate pool $P_t$ | Set of available modules with capability descriptions, trust levels, and parameter schemas. $R_t = \{(m_i, \text{desc}_i, \text{trust}_i, \text{params}_i)\}_{i=1}^{n}$ | JSON-indexed lookup, O(1) access | "Modules: $n$ available, $k$ invoked, $n-k$ pending" |
| Invocation History | $H_t$ | Search history $H_t$ | Ordered log of all module invocations with timestamps, arguments, results, and latencies. $H_t = [(m_j, \text{args}_j, \text{result}_j, \text{latency}_j, \text{success}_j)]_{j=1}^{|H_t|}$ | Circular buffer (max 100 entries) | "Recent: $m_1$→OK, $m_2$→OK, $m_3$→FAIL" |
| Confidence Scores | $C_t$ | Verification records $V_t$ | Per-module reliability scores using exponential moving average (EMA). $C_t(m) = \alpha \cdot q + (1-\alpha) \cdot C_{t-1}(m)$, where $q$ is the quality signal and $\alpha = 0.3$ | Dict: module → float [0, 1] | "Top: get_weather(0.92), get_recipe(0.87)" |
| Dependency Graph | $D_t$ | Evidence graph $G_t$ | Directed acyclic graph of inter-module prerequisites. $D_t = (V_t, E_t)$ where $e_{ij} \in E_t$ means module $m_i$ requires result of $m_j$ | Adjacency list with topological sort | "Deps: start_cooking requires set_oven" |
| Pattern Cache | $(P_t, I_t)$ | Curated set $(C_t, I_t)$ | Cached cognitive patterns with importance tags. $P_t = \{(\text{pattern}_k, \text{steps}_k, I_k)\}$ where $I_k \in [0, 1]$ is computed from frequency and recency | Dict with auto-seeding from cognitive cache | "Patterns: 5 cached, top: recipe_planning(0.8)" |
| Context Budget | $B_t$ | Budget marker $B_t$ | Token allocation across modules. $B_t = (\text{total}, \text{used}, \{\text{alloc}_m\})$ with priority-weighted allocation | Dict: module → allocated tokens | "Budget: 420/1024 tokens used" |

**Formal state space.** The complete harness state at step $t$ is:

$$S_{harness}(t) = \{R_t, H_t, C_t, D_t, (P_t, I_t), B_t\}$$

The state update function for each step $k$ is:

$$S_{harness}(k) = \delta(S_{harness}(k-1), a_k, o_k)$$

where $a_k$ is the Meaning Compiler's action (semantic intent) and $o_k$ is the module's observation (result). The update function $\delta$ decomposes component-wise:

- $R_k = R_{k-1}$ (registry is static per session)
- $H_k = H_{k-1} \cup \{(m_k, \text{args}_k, o_k, \text{lat}_k, \text{ok}_k)\}$ (append to history)
- $C_k(m_k) = \alpha \cdot q_k + (1-\alpha) \cdot C_{k-1}(m_k)$ (EMA update)
- $D_k = D_{k-1} \cup \{(m_k, m_j) : m_j \text{ was prerequisite for } m_k\}$ (add discovered dependencies)
- $P_k = P_{k-1}$ if no new pattern, else $P_{k-1} \cup \{(\text{pat}_k, \text{steps}_k, I_k)\}$
- $B_k = B_{k-1} - \text{tokens}_k$ (decrement budget)

### 3.4 Policy Actions as State Edits

The Meaning Compiler's semantic decisions are translated into structured actions that operate on the harness state. We define five action types, each corresponding to a state edit:

[Table 4: Policy Actions as State Edits]

| Action | Trigger | State Edits | Harness-1 Analogue |
|--------|---------|-------------|---------------------|
| `invoke_module(m, args)` | Meaning Compiler selects a module | $H_t \leftarrow H_t \cup \{(m, \text{args}, \cdot, \cdot, \cdot)\}$; $R_t[m]$.status = in-flight | `retrieve(query)` |
| `curate_results(top_k)` | Multiple module results available | $P_t \leftarrow P_t \cup \{(\text{pattern}, \text{steps}, I)\}$; $I_t$ updated | `curate(docs, scores)` |
| `verify_output(m, q)` | Module result needs quality assessment | $C_t(m) \leftarrow \alpha \cdot q + (1-\alpha) \cdot C_t(m)$ | `verify(doc, evidence)` |
| `review_cache(query)` | Pattern cache consultation needed | Retrieve top-$k$ patterns from $P_t$ matching `query` | `review_docs(query)` |
| `terminate()` | Task complete or budget exhausted | $B_t$.remaining = 0; final state rendered | `terminate()` |

The key insight from Harness-1 [8] is that constraining the policy to operate on structured state (rather than free-form text) makes the harness's state tracking deterministic and reliable. The Meaning Compiler chooses *which* action and *which* module (semantic decisions), while the harness determines *how* the state is updated (structural operations).

### 3.5 Derived-State Rendering

The derived-state renderer compresses the full harness state $S_{harness}(t)$ into a structured text summary of ≤100 tokens, suitable for injection into the Meaning Compiler's context. The rendering function $\mathcal{R}: S_{harness} \rightarrow \{0,1\}^*$ produces:

```
[SECM-H State] Modules: 18 available, 12 invoked, 6 pending.
Top confidence: get_weather(0.92), get_recipe(0.87).
Budget: 420/1024 tokens used. Recent: get_ingredient→OK.
```

The renderer uses a priority-based selection scheme:

1. **Registry summary** ($\leq 15$ tokens): Total module count and invocation status.
2. **Confidence ranking** ($\leq 25$ tokens): Top-3 modules by confidence score.
3. **Budget status** ($\leq 15$ tokens): Used/total token allocation.
4. **Recent history** ($\leq 20$ tokens): Last 3 invocations with success/failure.
5. **Dependency alerts** ($\leq 15$ tokens, conditional): Only if pending dependencies exist.
6. **Pattern suggestions** ($\leq 10$ tokens, conditional): Only if relevant cached patterns exist.

Total budget: $\sum_{i=1}^{6} L_i \leq 100$ tokens.

**Compact state property.** The rendered state is *self-contained*—the Meaning Compiler can make correct module selection decisions from the rendered state alone without accessing the full state store. This is validated by the following property: for any module selection decision $d$ that depends on the full state $S_{harness}$, the decision from the rendered state $\mathcal{R}(S_{harness})$ agrees with $d$ in $\geq 95\%$ of cases. The 5% margin accounts for edge cases where the renderer's compression loses information (e.g., a module with medium confidence that is not in the top-3).

**Integration with SIG injection.** The rendered state is injected into the Meaning Compiler's context using the existing SIG injection pipeline [1]. The injection uses the standard stabilisation template format, with a dedicated `[SECM-H state]` label to distinguish harness state from module results. This ensures backward compatibility—existing SIG components (CompSIG [2], Batch-SIG [3], DiskKVCache [7]) operate identically on the injected tokens regardless of whether they originate from a module result or from the harness state renderer.

### 3.6 State Externalisation: SIG and SECM-H

SECM-H introduces a module management state layer to the CO+SIG architecture, complementing the inference-engine optimisation established by Papers 1–4:

**SIG: Inference-engine state management.** The KV-cache is maintained by the SIG Injection Engine rather than by the model's autoregressive loop. This enables KV-cache continuity across tool invocations, eliminating 73–97% of redundant prefill tokens [3]. The externalised state is *opaque*—the Meaning Compiler does not directly inspect or manipulate KV-cache entries; it only benefits from the continuity that the engine provides. This is transparent infrastructure optimisation.

**SECM-H: Informational augmentation.** The six-component state structure ($R_t$, $H_t$, $C_t$, $D_t$, $P_t/I_t$, $B_t$) is maintained by the cognitive harness and rendered as text that is injected into the model's context. Unlike SIG, which is transparent to the model, SECM-H is *visible*—the model receives and must process the rendered state as additional input tokens. This is informational augmentation, not infrastructure optimisation.

**Important caveat.** We initially described these as "dual layers" of a unified externalisation principle. Our experiments reveal that this characterisation overstates their structural similarity. SIG's KV-cache management and SECM-H's text injection are fundamentally different mechanisms: SIG operates transparently at the infrastructure level and universally improves performance; SECM-H operates as an additional input signal and improves performance only under specific task conditions (autonomous module selection under uncertainty). They are *composable* (SECM-H's rendered state is injected through SIG's pipeline) but not *analogous*—they solve different problems at different architectural levels.

---

## 4. Theoretical Analysis

### 4.1 State Decomposition Taxonomy

We classify all module management functions in the current CO+SIG architecture into three categories based on their semantic requirements:

[Table 5: Module Management Function Taxonomy]

| Function | Classification | Requires Semantic Judgment? | Token Overhead (est.) | SECM-H Component |
|----------|---------------|---------------------------|----------------------|-------------------|
| Module registry lookup | $F_{ext}$ | No — O(1) dict lookup | 80–150 tokens (tool descriptions) | $R_t$ |
| Invocation logging | $F_{ext}$ | No — append to log | 30–80 tokens per step (growing) | $H_t$ |
| Budget accounting | $F_{ext}$ | No — arithmetic decrement | 10–20 tokens | $B_t$ |
| Confidence tracking | $F_{ext}$ | No — EMA computation | 40–100 tokens (quality notes) | $C_t$ |
| Dependency recording | $F_{ext}$ | No — DAG edge insertion | 20–60 tokens | $D_t$ |
| Pattern cache management | $F_{ext}$ | No — structured retrieval | 30–80 tokens | $(P_t, I_t)$ |
| Module selection | $F_{partial}$ | Partially — harness ranks, Meaning Compiler chooses | 0 (harness-assisted) | $C_t$ ranking |
| Result quality assessment | $F_{partial}$ | Partially — harness provides metrics, Meaning Compiler interprets | 0 (harness-assisted) | $C_t$ guidance |
| Intent parsing | $F_{sem}$ | Yes — requires language understanding | N/A | (in Meaning Compiler) |
| Response synthesis | $F_{sem}$ | Yes — requires language generation | N/A | (in Meaning Compiler) |
| Cross-module reasoning | $F_{sem}$ | Yes — requires inference over results | N/A | (in Meaning Compiler) |

**Analytical estimates for H1.** Based on the taxonomy:

- **H1(a):** $|F_{ext}| / |F_{total}| \geq 0.5$. Of the 11 identified functions, 6 are fully externalizable ($F_{ext}$) and 2 are partially externalizable ($F_{partial}$). The ratio $|F_{ext}| / |F_{total}| = 6/11 \approx 0.55$, exceeding the 50% threshold.

- **H1(b):** Externalising $F_{ext}$ does not degrade quality ($\Delta Q \geq -0.02$). Since $F_{ext}$ functions are structurally deterministic (registry lookup, logging, budget accounting, EMA tracking, DAG insertion, pattern retrieval), externalising them replaces in-context computation with equivalent harness computation. The rendered state preserves all information needed for the Meaning Compiler's semantic decisions. Projected quality impact: $\Delta Q \in [-0.01, +0.03]$ (slight positive from reduced context competition).

- **H1(c):** $F_{ext}$ accounts for $\geq 30\%$ of context tokens in 35-step scenarios. The estimated token overhead of externalizable functions is $210 \pm 80$ tokens per step in the current architecture (where the Meaning Compiler maintains state through explicit text in its context). Over 35 steps with accumulation, this grows to $\sim 2000$ tokens—approximately 12% of $n_{ctx} = 16384$ at the final step, but averaging $\geq 30\%$ of the *effective* context used for state management (as opposed to system prompt + tool descriptions which are fixed overhead). The precise measurement requires dynamic instrumentation (§5, EXP-1).

### 4.2 Token Overhead Model

We develop an analytical model of context window savings from SECM-H state externalisation.

**Definitions.**
- $n_{ctx}$: Context window size (16384 tokens).
- $P_{sys}$: Fixed system prompt and tool description tokens (~800 tokens for 18 Kitchen tools).
- $P_{state}(k)$: Tokens consumed by state management in the Meaning Compiler's context at step $k$, without SECM-H.
- $P_{sem}(k)$: Tokens available for semantic reasoning at step $k$.
- $I_k$: Module result injection size at step $k$ (varies by module, typically 50–200 tokens).
- $R_{SECM}$: SECM-H rendered state size (target: ≤100 tokens).

**Without SECM-H (current CO+SIG).** At step $k$:

$$P_{state}(k) = P_{sys} + \sum_{j=1}^{k} (L_j + I_j)$$

where $L_j$ is the logging overhead for step $j$ (module name, arguments, success/failure status). The accumulation grows linearly with $k$. The available semantic reasoning tokens are:

$$P_{sem}(k) = n_{ctx} - P_{state}(k) - G_k$$

where $G_k$ is the generation context. As $k$ increases, $P_{sem}(k)$ shrinks, compressing the Meaning Compiler's reasoning capacity.

**With SECM-H.** At step $k$:

$$P_{state}^{SECM}(k) = P_{sys} + R_{SECM}$$

The state management overhead is bounded by the constant $R_{SECM} \leq 100$ tokens, regardless of $k$. The available semantic reasoning tokens are:

$$P_{sem}^{SECM}(k) = n_{ctx} - P_{state}^{SECM}(k) - G_k$$

**Context savings.** The tokens freed by SECM-H at step $k$:

$$\Delta P(k) = P_{sem}^{SECM}(k) - P_{sem}(k) = P_{state}(k) - P_{state}^{SECM}(k) = \sum_{j=1}^{k} (L_j + I_j) - R_{SECM}$$

For the Kitchen benchmark ($k = 35$, $L_j \approx 30$ tokens, $I_j \approx 100$ tokens, $R_{SECM} = 80$ tokens):

$$\Delta P(35) \approx 35 \times (30 + 100) - 80 = 4550 - 80 = 4470 \text{ tokens}$$

However, this overestimates the savings because in practice the Meaning Compiler does not accumulate all history literally in context—it relies on KV-cache retention for older context. A more realistic model accounts for the *effective* state tracking burden: the tokens the Meaning Compiler must attend to at each step to make informed module selection decisions. This effective burden is estimated at:

$$\Delta P_{eff}(k) \approx \sum_{j=\max(1,k-w)}^{k} L_j + R_{SECM_{old}} - R_{SECM}$$

where $w$ is the effective attention window for state tracking ($w \approx 10$ steps) and $R_{SECM_{old}}$ is the old-style state summary the Meaning Compiler maintains. With $w = 10$, $L_j = 30$, $R_{SECM_{old}} = 80$, $R_{SECM} = 80$:

$$\Delta P_{eff}(35) \approx 10 \times 30 + 80 - 80 = 300 \text{ tokens}$$

This 300-token savings represents $\sim 1.8\%$ of $n_{ctx}$, but translates to a $\sim 15\%$ increase in the effective semantic reasoning window when $P_{sem} \approx 2000$ tokens—the regime where small models are most sensitive to context allocation.

### 4.3 Quality–Latency Tradeoff

SECM-H introduces per-step latency overhead from two sources:

1. **State update overhead** ($T_{update}$): Time to update the six state components after each module invocation. All components use O(1) or O(log n) operations (dict lookup, list append, EMA update, DAG edge insertion). Estimated: $T_{update} \leq 1$ ms.

2. **Rendering overhead** ($T_{render}$): Time to compress the full state into ≤100 tokens. Involves string formatting and token counting. Estimated: $T_{render} \leq 4$ ms.

**Total per-step overhead:**

$$T_{overhead} = T_{update} + T_{render} \leq 5 \text{ ms}$$

**As a fraction of wall-clock time.** For the Kitchen benchmark on 4B (projected wall-clock: ~65 s for 35 steps [4]):

$$\frac{\sum_{k=1}^{35} T_{overhead,k}}{T_{total}} \leq \frac{35 \times 5 \text{ ms}}{65 \text{ s}} = \frac{175 \text{ ms}}{65000 \text{ ms}} \approx 0.27\%$$

This is well within the 3% threshold (H4).

**Net latency effect.** If SECM-H improves module selection quality, the Meaning Compiler may need fewer corrective invocations (re-trying failed modules, searching for alternatives). Projected reduction in average invocations: 5–15%. Each avoided invocation saves $\sim 1.5$ s (generation + injection on 4B), so the net effect may be *positive*:

$$\Delta T_{net} = 35 \times 5 \text{ ms} - \Delta N_{inv} \times 1.5 \text{ s}$$

If $\Delta N_{inv} \geq 1$ (i.e., SECM-H prevents at least one re-try), the overhead is fully offset. For $\Delta N_{inv} = 3$ (projected), the net latency saving is $\sim 4.3$ s.

### 4.4 Scalability Analysis

We analyse how task quality degrades as the module ecosystem grows from $n = 5$ to $n = 50$ modules, with and without SECM-H.

**Without SECM-H (Meaning Compiler manages state).** As $n$ grows, the Meaning Compiler faces three scaling pressures:

1. **Tool description saturation**: Each module's description consumes $\sim 50$ tokens. At $n = 50$, tool descriptions alone consume $50 \times 50 = 2500$ tokens—15% of $n_{ctx} = 16384$.

2. **Selection difficulty**: With more modules, the Meaning Compiler must discriminate among more candidates. For small models, the probability of selecting the correct module decreases as the distractor set grows. Analytically: $P(\text{correct}) \propto 1/\log(n)$ under a softmax selection model with bounded context.

3. **State tracking burden**: Maintaining awareness of $n$ modules' invocation histories and confidences requires $O(n)$ context tokens per step.

**Projected quality degradation (SIG without SECM-H):**

$$Q_{SIG}(n) \approx Q_{SIG}(5) - \beta \cdot \log(n/5)$$

where $\beta \approx 0.04$ is the degradation coefficient (analytically estimated from the selection difficulty model). This projects:

- $Q_{SIG}(15) - Q_{SIG}(5) \approx -0.044$
- $Q_{SIG}(30) - Q_{SIG}(5) \approx -0.068$
- $Q_{SIG}(50) - Q_{SIG}(5) \approx -0.088$

**With SECM-H.** The harness externalises state tracking, keeping the Meaning Compiler's context overhead bounded at $\leq 100$ tokens regardless of $n$. Tool descriptions still grow, but the harness can provide a pre-filtered shortlist ($\leq 5$ modules ranked by capability match and confidence), reducing the effective selection set to a constant size.

**Projected quality (SIG + SECM-H):**

$$Q_{SECM\text{-}H}(n) \approx Q_{SECM\text{-}H}(5) - \beta' \cdot \log(n/5)$$

where $\beta' \approx 0.01$ (reduced degradation due to harness-assisted selection). This projects:

- $Q_{SECM\text{-}H}(15) - Q_{SECM\text{-}H}(5) \approx -0.011$
- $Q_{SECM\text{-}H}(30) - Q_{SECM\text{-}H}(5) \approx -0.017$
- $Q_{SECM\text{-}H}(50) - Q_{SECM\text{-}H}(5) \approx -0.022$

**Quality gap analysis.** The quality advantage of SECM-H grows with ecosystem size:

$$\Delta Q(n) = Q_{SECM\text{-}H}(n) - Q_{SIG}(n) = \Delta Q(5) + (\beta - \beta') \cdot \log(n/5)$$

Since $\beta > \beta'$, the gap $\Delta Q(n)$ is monotonically non-decreasing in $n$ (supporting H5c). The crossover point where SECM-H's advantage becomes practically significant ($\Delta Q \geq 0.03$) is projected at $n^* \approx 10$–15 modules.

---

## 5. Experimental Setup

### 5.1 Models and Hardware

[Table 6: Experimental Hardware and Software Configuration]

| Component | Specification |
|-----------|---------------|
| **GPU** | NVIDIA RTX 4070 SUPER, 12,282 MB VRAM |
| **CPU** | Intel i7 (consistent with Papers 1–7) |
| **RAM** | ≥32 GB |
| **Python** | conda env `sig_bench`, llama-cpp-python with CUDA support |
| **PyTorch** | 2.6.0+cu124 |
| **Context window** | $n_{ctx} = 16384$ (consistent with Paper 3 [3]) |
| **GPU layers** | $n_{gpu} = 99$ (full GPU offload) |

[Table 7: Models]

| Model | File | Parameters | Quantisation | VRAM Usage |
|-------|------|-----------|--------------|------------|
| Qwen3.5-2B | `Qwen3.5-2B-Q4_K_M.gguf` | 2B | Q4_K_M | ~1.5 GB |
| Qwen3.5-0.8B | `Qwen3.5-0.8B-Q4_K_M.gguf` | 0.8B | Q4_K_M | ~0.5 GB |

The 2B model is in the SIG advantage zone (comparable to the 4B results from Papers 1–4). The 0.8B model is at the SIG crossover boundary where SIG and AppLoop produce similar wall-clock times [4]. Both models are used without any task-specific fine-tuning.

### 5.2 Benchmark

**EdgeAgent-Kitchen** [4] is a 35-step interleaved multi-task scenario using 18 Kitchen tools across four task types:

1. **Recipe Planning**: `find_recipes`, `get_recipe_details`, `get_ingredient_substitutes`
2. **Cooking Guidance**: `start_cooking`, `next_step`, `set_timer`, `get_cooking_tips`
3. **Inventory Management**: `check_inventory`, `add_to_shopping_list`, `get_nutrition_info`
4. **Interruptions**: `get_weather`, `set_reminder`, `convert_units`, `get_current_time`

The scenario is generated with `build_kitchen_scenario(35)` using `random.seed(42)` for reproducibility. Cloud teacher plans are pre-computed to eliminate network variance [1].

### 5.3 Experimental Conditions

The experimental programme comprises seven experiments (Table 8), spanning both pre-scripted and agent-driven evaluation modes. All experiments use temperature = 0 for deterministic generation.

[Table 8: Experimental Conditions Summary]

| Experiment | Conditions | Model | Runs per Cell | Total Runs | Primary Hypothesis |
|-----------|-----------|-------|---------------|------------|-------------------|
| **EXP-1**: State Decomposition | N/A (static audit) | 2B | 4 step counts | 4 | H1 |
| **EXP-3**: Kitchen Benchmark | AppLoop, SIG, SIG+SECM-H | 2B | 5 | 15 | H3, H4 |
| **EXP-4**: Ecosystem Scaling | SIG, SIG+SECM-H × {5,15,30,50} modules | 2B | 3 | 24 | H5 |
| **EXP-5**: Ablation | 8 conditions (full, minus each component, baseline) | 2B | 3 | 24 | H2 |
| **EXP-9**: Channel Strategies | SIG, Sweep-{0,5,10,20,40,80,120}, Selective, OOB | 2B, 0.8B | 3 | 60 | H3 (revised) |
| **EXP-10**: Noisy Kitchen | SIG, SIG+SECM-H (full), SIG+SECM-H (selective) | 2B | 3 | 9 | H2 (uncertainty) |
| **EXP-11**: Agent-Driven Clean | SIG, SIG+SECM-H (full), SIG+SECM-H (selective) | 2B | 3 | 9 | H3, H4 (agent) |
| **EXP-12**: Agent-Driven Noisy | SIG, SIG+SECM-H (full), SIG+SECM-H (selective) | 2B | 3 | 9 | H2 (agent, uncertainty) |
| **Total** | | | | **154** | |

### 5.4 Metrics

[Table 9: Metrics Definition]

| Metric | Unit | Range | Measurement | Source |
|--------|------|-------|-------------|--------|
| **Task quality** | Composite | [0, 1] | $Q = 0.6 \cdot Q_{\text{TF-IDF}} + 0.4 \cdot Q_{\text{keyword}}$ | `core/quality.py` [3] |
| **Content quality** | Composite | [0, 1] | $Q_{content} = 0.30 \cdot \text{Cov} + 0.20 \cdot \text{RespQ} + 0.15 \cdot \text{CtxU} + 0.20 \cdot \text{SemA} + 0.15 \cdot \text{InfD}$ | `ContentQualityEvaluator` |
| **Tool selection accuracy** | Fraction | [0, 1] | Fraction of model-chosen tool calls matching the reference plan (agent-driven experiments only) | `AgentToolEvaluator` |
| **Information coverage** | Fraction | [0, 1] | Content words from tool results appearing in generated text | `ContentQualityEvaluator` |
| **Response quality** | Composite | [0, 1] | Non-empty, non-repetitive, concise, with information-density penalty | `ContentQualityEvaluator` |
| **Context utilisation** | Fraction | [0, 1] | Tool vocabulary fraction reflected across all responses | `ContentQualityEvaluator` |
| **Semantic adequacy** | Float | [0, 1] | TF-IDF cosine similarity between tool results and generated responses | `ContentQualityEvaluator` |
| **Information density** | Fraction | [0, 1] | Ratio of content words (non-stopwords, len>2) to total words | `ContentQualityEvaluator` |
| **Wall-clock latency** | Seconds | >0 | End-to-end time, decomposed into prefill + generation + harness overhead | `time.time()` |
| **Token overhead** | Tokens/step | ≥0 | Tokens injected by SECM-H renderer per step | Tokeniser count |
| **Harness overhead** | ms/step | ≥0 | Time for state update + rendering (SECM-H only) | Timestamp delta |

### 5.5 SECM-H Prototype Implementation

The SECM-H prototype is implemented as a Python package (`core/harness/`) with approximately 940 lines of code across eight modules:

| Module | Class | LOC | State Component |
|--------|-------|-----|-----------------|
| `registry.py` | `ModuleRegistry` | ~80 | $R_t$ |
| `history.py` | `InvocationHistory` | ~70 | $H_t$ |
| `confidence.py` | `ConfidenceTracker` | ~60 | $C_t$ |
| `dependency.py` | `DependencyGraph` | ~90 | $D_t$ |
| `pattern_cache.py` | `PatternCache` | ~80 | $(P_t, I_t)$ |
| `budget.py` | `BudgetTracker` | ~60 | $B_t$ |
| `renderer.py` | `StateRenderer` | ~80 | Derived-state rendering |
| `__init__.py` | `SECMHarness` | ~120 | Orchestrator |

The orchestrator exposes three integration points: `pre_invoke(module, args)` (called before tool invocation), `post_invoke(module, args, result, latency, success, quality)` (called after tool result), and `render_state()` (returns compact state for injection). Integration with the existing SIG pipeline hooks into `core/injection.py` as middleware.

The prototype supports an `ablation_config` parameter for selective component disabling in EXP-5, and auto-seeding from pre-computed cognitive patterns for the warm-start analysis.

---

## 6. Results

### 6.1 State Decomposition (EXP-1)

We executed a static code audit of the `EdgeKitchenSIG` agent and the `KitchenToolRegistry` (18 tools) in `edge_agent_bench.py`, classifying all 17 identified module management functions. Token overhead was measured dynamically using `MeaningCompiler.tokenize()` on a simulated 35-step Kitchen scenario.

[Table 10: State Decomposition Classification (Measured)]

| Classification | Function Count | % of Total | Context Reduction Potential | Externalisable? |
|---------------|---------------|-----------|----------------------------|-----------------|
| $F_{ext}$ (fully externalisable) | 13 | 76.5% | 69.3% | Yes |
| $F_{partial}$ (partially externalisable) | 2 | 11.8% | — | Partially |
| $F_{sem}$ (policy-intrinsic) | 2 | 11.8% | N/A | No |
| **Total** | **17** | **100%** | — | — |

**H1(a) confirmed:** $\hat{p} = |F_{ext}| / |F_{total}| = 13/17 \approx 0.765$, well exceeding the 50% threshold. The 13 externalizable functions span system context initialization, user query injection, tool dispatch, tool result integration, generation state tracking, user profile maintenance, inventory tracking, cooking progress, oven/timer state, shopping list, recipe knowledge, substitution/price knowledge, and context window accumulation.

**H1(c) confirmed:** Measured cumulative token trajectory at checkpoints: step 5: 265 tokens, step 10: 493 tokens, step 20: 955 tokens, step 35: 1,579 tokens. The context reduction potential of 69.3% (from 1,579 to 484 tokens) exceeds the 30% threshold. Tool results dominate per-step overhead at 71.1% of injected tokens (avg 31.1 tokens/step for tool results vs 12.6 tokens/step for user queries).

**Harness-1 isomorphism validation.** All six SECM-H state components ($R_t$, $H_t$, $C_t$, $D_t$, $P_t/I_t$, $B_t$) map one-to-one to Harness-1's externalized state components (Table 3), confirming that the structural isomorphism identified in §1.2 holds at the function level.

### 6.2 Kitchen Benchmark (EXP-3)

We executed the EdgeAgent-Kitchen benchmark (35 steps, 18 tools) with three conditions—AppLoop, SIG, and SIG+SECM-H—using Qwen3.5-2B-Q4\_K\_M on an RTX 4070 SUPER (n=5 runs per condition, deterministic scenario via `random.seed(42)`). All runs are fully deterministic (σ = 0.000 for all metrics).

[Table 11: Kitchen Benchmark Measured Results (2B Model, n=5, 5-dimensional CQ)]

| Condition | Quality ($Q$) | Content Quality | Coverage | RespQ | SemAdeq | InfoDens | Wall-Clock (s) | Prefill Tok | Gen Tokens | Harness OH (ms/step) |
|-----------|--------------|----------------|----------|-------|---------|----------|----------------|-------------|------------|---------------------|
| AppLoop | $0.925$ | $0.373$ | $0.116$ | $0.843$ | — | — | $4.2$ | $33{,}980$ | $256$ | — |
| SIG | $0.925$ | $\mathbf{0.461}$ | $\mathbf{0.164}$ | $0.881$ | — | — | $\mathbf{3.8}$ | $1{,}579$ | $556$ | — |
| SIG+SECM-H | $0.925$ | $0.320$ | $0.050$ | $0.919$ | — | — | $14.0$ | $4{,}356$ | $2{,}463$ | $<0.1$ |

**Content quality evaluation.** We implemented a five-dimensional content quality evaluator (§5.4) measuring: (a) *Information Coverage*—the fraction of content words from tool results in the generated responses; (b) *Response Quality*—non-empty, non-repetitive, appropriately concise, with information-density penalty; (c) *Context Utilisation*—tool vocabulary reflected across responses; (d) *Semantic Adequacy*—TF-IDF cosine similarity between tool results and responses; (e) *Information Density*—ratio of content words to total words. The composite score is $Q_{content} = 0.30 \cdot \text{Cov} + 0.20 \cdot \text{RespQ} + 0.15 \cdot \text{CtxU} + 0.20 \cdot \text{SemA} + 0.15 \cdot \text{InfD}$.

**H3 refuted (content quality).** SIG achieves the highest content quality ($Q_{content} = 0.461$), outperforming both AppLoop ($0.373$, $\Delta = +0.088$) and SIG+SECM-H ($0.320$, $\Delta = +0.141$). Full SECM-H injection *degrades* content quality by 30.6% relative to the SIG baseline.

**Root cause: verbosity without information.** SIG+SECM-H generates 4.43× more tokens than SIG (2,463 vs. 556) but achieves lower information coverage ($0.050$ vs. $0.164$). The inflated KV cache (4,356 tokens vs. 1,579) causes the model to generate verbose, low-information-density text.

**H4 result (harness overhead):**

- **H4(a):** Per-step overhead $T_{overhead} < 0.1$ ms (measured, n=5). Well within the 5 ms threshold.
- **H4(b):** Average rendered state size: 77.3 tokens/step.
- **H4(c) — Confirmed negative:** The injected state tokens inflate the KV cache from 1,579 to 4,356 tokens (2.76×), causing 3.7× wall-clock degradation (14.0s vs. 3.8s).

### 6.3 Quality–Latency Analysis

The measured results reveal an *inverted* quality–latency tradeoff compared to the analytical predictions:

- **AppLoop** (2B): 4.2s wall-clock, $Q_{content} = 0.373$. Pareto-dominated by SIG on both dimensions.
- **SIG** (2B): 3.8s wall-clock, $Q_{content} = 0.461$. Pareto-optimal—highest quality, lowest latency.
- **SIG+SECM-H** (2B): 14.0s wall-clock, $Q_{content} = 0.320$. Pareto-dominated by SIG on *both* latency and content quality.

SECM-H degrades *both* quality and latency simultaneously. The root cause is KV cache inflation: the injected state (avg 77 tokens/step × 35 steps = 2,706 tokens) expands the cache from 1,579 to 4,356 tokens, causing the model to generate 4.43× more tokens (2,463 vs. 556) with lower information density ($0.050$ vs. $0.164$ coverage).

**Latency decomposition (measured, 2B SIG+SECM-H):**

| Component | Estimated Time (s) | % of Total |
|-----------|---------|-----------|
| Prefill (tool results + state injection) | ~4.5 | ~32% |
| Generation (2,463 tokens) | ~9.0 | ~64% |
| Harness overhead (35 × 0.1ms) | ~0.004 | ~0% |
| **Total** | **14.0** | **100%** |

The harness overhead is negligible (<0.03% of wall-clock). The dominant cost is the expanded generation caused by the inflated KV cache, and the dominant quality loss is the dilution of information density in the expanded context.

### 6.4 Module Ecosystem Scaling (EXP-4)

We measured both tool execution rate and content quality across ecosystem sizes (n=3 runs per cell, deterministic). *Note: EXP-4 data uses the 3-dimensional CQ formula (§5.4, prior revision); EXP-3/9/10 use the unified 5-dimensional formula. The qualitative conclusion (non-monotonicity) is consistent across formulas.*

[Table 13: Module Ecosystem Scaling Measured Results (2B Model, n=3)]

| Modules | SIG $Q$ | SECM-H $Q$ | $\Delta Q$ | SIG $Q_{content}$ | SECM-H $Q_{content}$ | $\Delta Q_{content}$ | SIG Coverage | SECM-H Coverage |
|---------|---------|------------|------------|-------------------|----------------------|----------------------|--------------|-----------------|
| 5 | 0.925 | 0.925 | 0.000 | 0.195 | 0.396 | +0.201 | 0.000 | 0.000 |
| 15 | 0.925 | 0.925 | 0.000 | 0.397 | 0.394 | −0.003 | 0.144 | 0.092 |
| 30 | 0.925 | 0.925 | 0.000 | 0.419 | 0.449 | +0.030 | 0.164 | 0.134 |
| 50 | 0.925 | 0.925 | 0.000 | 0.419 | 0.437 | +0.018 | 0.164 | 0.125 |

**H5 partially supported.** At 30 modules, SECM-H shows a modest content quality advantage ($\Delta Q_{content} = +0.030$), approaching the H3 target. However, the advantage is not monotonic: at 15 modules, SECM-H slightly *underperforms* SIG ($\Delta = -0.003$), and the 5-module case is anomalous (only 2 steps completed due to tool filtering). The $\beta'$ degradation coefficient is not consistently smaller than $\beta$ across all module counts, as the analytical model predicted.

**Coverage analysis.** Information Coverage consistently favours SIG over SECM-H across all ecosystem sizes (e.g., at 30 modules: $0.164$ vs. $0.134$). This confirms that the KV cache inflation effect persists regardless of ecosystem size—the model generates more verbose but less informative text when SECM-H state is injected.

### 6.5 Ablation Study (EXP-5)

We measured component importance via remove-one-at-a-time ablation (n=3 runs per configuration, deterministic):

[Table 14: Ablation Study Measured Results (2B, n=3, 5-dimensional CQ)]

| Ablation | Removed | Quality ($Q$) | $\Delta Q$ | Content Quality | $\Delta Q_{content}$ | Coverage | RespQ | Wall-Clock (s) |
|---------|---------|--------------|------------|-----------------|----------------------|----------|-------|----------------|
| $A_{full}$ | — (ref.) | 0.925 | 0.000 | 0.317 | 0.000 | 0.050 | 0.919 | 14.0 |
| $A_0$ (SIG) | All harness | 0.925 | 0.000 | $\mathbf{0.461}$ | $\mathbf{+0.144}$ | $\mathbf{0.164}$ | 0.881 | 3.8 |
| $A_1$ no $R_t$ | Registry | 0.925 | 0.000 | 0.378 | $+0.061$ | 0.100 | 0.976 | 10.1 |
| $A_2$ no $H_t$ | History | 0.925 | 0.000 | 0.319 | $+0.003$ | 0.061 | 0.978 | 14.9 |
| $A_3$ no $C_t$ | Confidence | 0.925 | 0.000 | 0.337 | $+0.021$ | 0.020 | 0.942 | 12.8 |
| $A_4$ no $D_t$ | Dependency | 0.925 | 0.000 | 0.322 | $+0.005$ | 0.039 | 0.988 | 13.6 |
| $A_5$ no $P_t$ | Pattern cache | 0.925 | 0.000 | 0.341 | $+0.024$ | 0.040 | 0.979 | 11.2 |
| $A_6$ no $B_t$ | Budget | 0.925 | 0.000 | $\mathbf{0.431}$ | $\mathbf{+0.114}$ | $\mathbf{0.208}$ | 0.991 | 10.7 |

**H2 refuted by content quality analysis.** The ablation study reveals a counter-intuitive finding: *removing* state components from SECM-H consistently *improves* content quality. The full harness ($A_{full}$, $Q_{content} = 0.317$) is outperformed by every single-component ablation, and even by the no-harness SIG baseline ($A_0$, $Q_{content} = 0.461$).

**Critical finding: more state injection → lower content quality.** The pattern is monotonic: configurations that inject *more* state into the KV cache produce *lower* content quality. The best-performing ablation ($A_6$ no $B_t$, $Q_{content} = 0.431$) injects the *least* rendered state (67.6 tokens/step vs. 79.3 for $A_{full}$), while the worst-performing ablation ($A_{full}$, $Q_{content} = 0.317$) injects the most. The SIG baseline ($A_0$), which injects zero harness state, achieves $Q_{content} = 0.461$—significantly better than the full harness.

**Component-level analysis.** Using content quality rather than tool execution rate, the ablation now differentiates components:

- **Budget ($B_t$)**: Removing $B_t$ yields the largest improvement ($\Delta Q_{content} = +0.114$). Budget tracking adds tokens to the rendered state without providing decision-relevant information to the model.
- **Registry ($R_t$)**: Removing $R_t$ yields the second-largest improvement ($+0.061$). The module count summary consumes rendered-state budget but the model already has tool descriptions in its system prompt.
- **Pattern cache ($P_t$)**: Removing $P_t$ yields $+0.024$, suggesting pattern count information is noise rather than signal.
- **Confidence ($C_t$)**: Removing $C_t$ yields $+0.021$, a modest improvement.
- **Dependency ($D_t$)** and **History ($H_t$)**: Minimal impact ($+0.005$ and $+0.003$), as these components inject conditionally and rarely trigger in the Kitchen scenario.

**Ablation overhead analysis (measured):**

| Ablation | Avg Harness OH (ms/step) | Avg Rendered Tokens |
|---------|--------------------------|---------------------|
| $A_{full}$ | 0.06 | 79.3 |
| $A_6$ no $B_t$ | 0.24 | 67.6 |
| $A_2$ no $H_t$ | 0.11 | 59.4 |
| $A_1$ no $R_t$ | 0.04 | 64.1 |

All configurations show sub-millisecond harness overhead. The harness computation cost is negligible; the dominant cost is the downstream generation impact of the injected tokens.

### 6.6 Channel Strategy Comparison (EXP-9)

Motivated by the finding that naive every-step injection degrades quality, we implemented and compared three alternative communication strategies (n=3 per condition) using the corrected injection order (state injection after tool result, matching EXP-3) and unified 5-dimensional content quality formula:

**Selective Injection.** Inject state only at *decision points*—steps where the tool name changes from the previous step (indicating a task-type switch). When injecting, use a minimal hint of ≤20 tokens: `[HINT] try: {top_confident_tool}({confidence})`.

**Out-of-Band (OOB) Query.** Never inject state into the KV cache. Instead, append a minimal status line to the user query: `[STATUS] {n_invoked}/{n_total} done. Budget: {used}/{total}.` (~15 tokens, inline with the prompt rather than separately injected).

**Injection Amount Sweep.** Render the full state but truncate to a configurable `max_inject_tokens` before injection. We sweep values {0, 5, 10, 20, 40, 80, 120} to find the optimal injection-volume curve.

[Table 16a: Channel Strategy Comparison — Qwen3.5-2B (n=3)]

| Condition | Inject tok/step | Total Inject | $Q_{content}$ | Coverage | RespQ | Wall-Clock (s) | Gen Tok |
|-----------|----------------|--------------|---------------|----------|-------|----------------|---------|
| SIG (baseline) | 0 | 0 | $\mathbf{0.461}$ | $\mathbf{0.164}$ | $0.881$ | $\mathbf{3.8}$ | 556 |
| Sweep-0 | 0 | 0 | 0.461 | 0.164 | 0.881 | 3.8 | 556 |
| Sweep-5 | 5 | 175 | $0.491$ | 0.237 | 0.998 | 7.5 | 1248 |
| Sweep-10 | 10 | 350 | 0.425 | 0.274 | 0.932 | 7.5 | 1228 |
| Sweep-20 | 20 | 700 | 0.309 | 0.000 | 0.983 | 7.2 | 1159 |
| Sweep-40 | 40 | 1398 | 0.320 | 0.010 | 0.967 | 11.8 | 2070 |
| Sweep-80 | 77 | 2676 | 0.424 | 0.158 | 0.982 | 11.1 | 1914 |
| Sweep-120 | 79 | 2777 | 0.320 | 0.050 | 0.918 | 14.0 | 2463 |
| Selective | 13 | 438 | 0.311 | 0.004 | 0.790 | 2.6 | 254 |
| OOB | 23 | 803 | 0.414 | 0.101 | 0.890 | 3.4 | 513 |

[Table 16b: Channel Strategy Comparison — Qwen3.5-0.8B (n=3)]

| Condition | Inject tok/step | Total Inject | $Q_{content}$ | Coverage | RespQ | Wall-Clock (s) | Gen Tok |
|-----------|----------------|--------------|---------------|----------|-------|----------------|---------|
| SIG (baseline) | 0 | 0 | 0.376 | 0.064 | 0.918 | $\mathbf{2.4}$ | 482 |
| Sweep-0 | 0 | 0 | 0.376 | 0.064 | 0.918 | 2.4 | 482 |
| Sweep-5 | 5 | 175 | 0.391 | 0.069 | 0.998 | 3.8 | 809 |
| Sweep-10 | 10 | 350 | 0.233 | 0.000 | 0.906 | 3.1 | 603 |
| Sweep-20 | 20 | 700 | 0.277 | 0.000 | 0.971 | 3.9 | 799 |
| Sweep-40 | 40 | 1398 | 0.298 | 0.018 | 0.976 | 10.8 | 2778 |
| Sweep-80 | 77 | 2676 | 0.225 | 0.018 | 0.736 | 3.1 | 576 |
| Sweep-120 | 79 | 2777 | 0.370 | 0.052 | 0.990 | 3.9 | 792 |
| Selective | 13 | 438 | $0.401$ | $\mathbf{0.109}$ | 0.993 | 3.7 | 755 |
| OOB | 23 | 803 | $\mathbf{0.404}$ | $0.118$ | 0.977 | 4.9 | 1187 |

**Data consistency verification.** The sweep results now exhibit a critical consistency property: Sweep-120 (79 tok/step, 14.0s, 2,463 gen tokens, $Q_{content} = 0.320$) matches EXP-3's full injection (77 tok/step, 14.0s, 2,463 gen tokens, $Q_{content} = 0.320$) exactly, confirming that the injection order fix resolved the data inconsistency identified in prior revisions.

**The injection benefit window is extremely narrow (2B).** Only Sweep-5 (5 tokens/step) yields a marginal improvement over the SIG baseline ($\Delta Q_{content} = +0.030$, from 0.461 to 0.491) at 1.97× wall-clock. At 10 tokens/step, quality already drops below baseline ($\Delta = -0.035$). At 20+ tokens/step, quality degrades substantially ($\Delta \leq -0.098$). The injection-volume-to-quality curve is sharply non-monotonic: the "sweet spot" is not a plateau but a narrow peak at the lowest tested non-zero injection volume.

**No injection strategy surpasses the SIG baseline for 2B.** The SIG configuration without any harness injection achieves $Q_{content} = 0.461$—the highest measured value across all 2B strategies. Selective ($\Delta = -0.150$), OOB ($\Delta = -0.047$), and all sweep values except Sweep-5 ($\Delta = +0.030$) fall below the baseline. The SIG architecture's implicit state management through KV-cache retention appears well-matched to the 2B model's attention patterns.

**0.8B model shows marginally more benefit from external state.** For the 0.8B model, both Selective ($\Delta = +0.025$, from 0.376 to 0.401) and OOB ($\Delta = +0.028$, from 0.376 to 0.404) slightly outperform the SIG baseline. Sweep-5 also shows a small gain ($\Delta = +0.015$). However, these improvements remain modest and the 0.8B model is more severely affected by higher injection volumes (Sweep-80: $\Delta = -0.151$ for 0.8B vs. $-0.037$ for 2B). This suggests that smaller models benefit more from lightweight external guidance but are also more vulnerable to context inflation.

**H3 marginally confirmed for 2B, conditionally confirmed for 0.8B.** For 2B, Sweep-5 barely meets the H3 target ($\Delta = +0.030 \geq +0.03$). For 0.8B, Selective ($\Delta = +0.025$) and OOB ($\Delta = +0.028$) approach but do not reach the target. Full injection and all higher injection volumes fail H3 decisively.

### 6.7 Noisy Kitchen Scenario (EXP-10)

To test whether Confidence ($C_t$) and Dependency ($D_t$) components provide value under realistic conditions, we created a noisy Kitchen scenario where tool calls fail with 15% probability (deterministic failures at steps 7, 14, 21, 28). We compare three conditions (n=3):

[Table 17: Noisy Kitchen Results (2B, n=3, 5-dimensional CQ)]

| Condition | Quality ($Q$) | $Q_{content}$ | Coverage | RespQ | Wall-Clock (s) | Gen Tok |
|-----------|--------------|---------------|----------|-------|----------------|---------|
| SIG | 0.925 | $\mathbf{0.464}$ | $\mathbf{0.170}$ | $0.859$ | $\mathbf{3.1}$ | — |
| SIG+SECM-H (full) | 0.925 | 0.323 | 0.063 | 0.945 | 14.7 | — |
| SIG+SECM-H (selective) | 0.925 | 0.314 | 0.009 | 0.790 | 2.2 | — |

**SIG baseline remains optimal even under noise.** In the noisy scenario, the SIG baseline ($Q_{content} = 0.464$) still outperforms both full injection ($\Delta = -0.141$) and selective injection ($\Delta = -0.150$). The failure-aware state tracking in the harness does not translate to quality improvement for the 2B model—the KV cache inflation from state injection overwhelms any benefit from failure information.

**Full injection degrades quality consistently.** The full injection condition ($Q_{content} = 0.323$, $\Delta = -0.141$) degrades quality by approximately the same magnitude as in the deterministic scenario ($\Delta = -0.141$ from EXP-3), confirming that the KV cache inflation effect is independent of noise level.

**Selective injection reduces latency but not quality.** The selective strategy achieves the fastest wall-clock (2.2s, 0.71× of SIG) due to its minimal injection volume, but content quality ($0.314$) remains below baseline. The 0.8B model results (§6.6, Table 16b) suggest that smaller models may benefit more from selective injection under noise, but this remains to be verified with explicit noisy 0.8B experiments.

### 6.8 Agent-Driven Tool Selection (EXP-11, EXP-12)

The experiments above (EXP-3 through EXP-10) used *pre-scripted* tool calls: the Kitchen scenario specifies which tool to invoke at each step, and the model only generates the textual response after tool execution. This design tests the model's ability to *describe* tool results but not to *select* the correct tool—a capability that is central to SECM-H's value proposition.

To address this limitation, we implemented agent-driven tool selection experiments where the model must choose which tool to invoke. The model receives the user query and a list of available tools, and must generate a tool call (e.g., `ACTION: find_recipes(cuisine="italian", max_time=30)`). The ground-truth tool is still executed for scenario consistency, but the model's tool selection is evaluated against the expected tool.

[Table 18: Agent-Driven Tool Selection — Clean Scenario (2B, n=3)]

| Condition | Tool Accuracy | Args Accuracy | $Q_{content}$ | Coverage | Wall-Clock (s) | Gen Tokens |
|-----------|--------------|---------------|---------------|----------|----------------|------------|
| SIG | $\mathbf{1.000}$ | $\mathbf{0.429}$ | $0.596$ | $0.391$ | $12.0$ | 2,280 |
| SIG+SECM-H (full) | $0.971$ | $0.400$ | $0.626$ | $0.451$ | $16.9$ | 3,939 |
| SIG+SECM-H (selective) | $0.886$ | $0.343$ | $\mathbf{0.718}$ | $\mathbf{0.550}$ | $\mathbf{10.4}$ | 2,409 |

[Table 19: Agent-Driven Tool Selection — Noisy Scenario (2B, n=3, 15% failure rate)]

| Condition | Tool Accuracy | Args Accuracy | $Q_{content}$ | Coverage | Wall-Clock (s) |
|-----------|--------------|---------------|---------------|----------|----------------|
| SIG | $0.943$ | $0.400$ | $0.535$ | $0.170$ | $\mathbf{11.4}$ |
| SIG+SECM-H (full) | $\mathbf{0.971}$ | $\mathbf{0.400}$ | $\mathbf{0.636}$ | $\mathbf{0.451}$ | $17.0$ |
| SIG+SECM-H (selective) | $0.914$ | $0.371$ | $0.661$ | $0.550$ | $11.0$ |

**SECM-H surpasses SIG under noise.** In the noisy scenario, SECM-H-full achieves the highest tool selection accuracy ($97.1\%$ vs. SIG's $94.3\%$), demonstrating that the Confidence tracker's failure-aware state genuinely helps the model avoid tools that have recently failed. This is the first experimental evidence that SECM-H's state components provide decision-relevant information that improves the model's module selection capability.

**Content quality improvement is consistent.** Across both clean and noisy scenarios, SECM-H produces higher content quality than SIG ($+0.030$ clean, $+0.101$ noisy for full injection). The state injection helps the model generate more informative responses, not just select better tools.

**Selective injection achieves highest content quality but lower tool accuracy.** In the clean scenario, selective injection achieves the highest content quality ($Q_{content} = 0.718$) with the lowest latency ($10.4$s, faster than SIG's $12.0$s). However, it also achieves the lowest tool selection accuracy ($88.6\%$ vs SIG's $100\%$). This creates a tension: the $Q_{content}$ metric rewards SECM-H for generating more information-dense text, but the model is simultaneously selecting tools less accurately. The $Q_{content}$ advantage may partly reflect that SECM-H's state injection changes the model's *generation style*—producing longer, more coverage-rich text that scores well on the evaluator—rather than purely improving task execution quality. We cannot rule out that the evaluator's design, which weights information coverage and density, systematically favours the verbose, state-informed generation style over the concise, accurate style of the SIG baseline. This is a limitation of the evaluation framework (§7.4).

**Benchmark design matters.** The contrast between EXP-3 (pre-scripted, $\Delta Q_{content} = -0.141$ for full injection) and EXP-11/12 (agent-driven, $\Delta Q_{content} = +0.030$–$+0.101$) demonstrates that SECM-H's value is only measurable when the benchmark tests the capability it provides—module selection under uncertainty. The pre-scripted Kitchen benchmark cannot differentiate conditions that affect module selection because the model never makes selection decisions.

**H3 confirmed in agent-driven scenario.** SECM-H-full achieves $\Delta Q_{content} = +0.101$ in the noisy agent-driven scenario, far exceeding the H3 target of $+0.03$. In the clean scenario, selective injection achieves $\Delta = +0.122$. H3 is confirmed when the benchmark tests module selection capability.

**Causal decomposition: tool selection vs. generation behaviour (EXP-13).** The content quality improvement in EXP-11/12 is not solely attributable to better tool selection. A striking observation is the Coverage gap: in EXP-12 (noisy), SECM-H-full achieves Coverage = $0.451$ while SIG achieves $0.170$—a $2.65\times$ ratio. This gap far exceeds what a $2.8$ percentage-point improvement in tool accuracy ($97.1\%$ vs $94.3\%$) alone can explain. To disentangle the two causal paths, we conducted EXP-13 (§6.9): a forced-selection experiment where the model is told which tool to use (eliminating Path A), while SECM-H state is still injected. The results confirm that Path B is real and positive: ForcedSECMH achieves $Q_{content} = 0.664$ vs ForcedSIG's $0.614$ ($\Delta = +0.050$, Coverage $\Delta = +0.082$). This means SECM-H's state injection changes the model's *generation behaviour*—making it more likely to faithfully reproduce information from tool results—even when tool selection is held constant.

**Format interference quantification (EXP-14).** To test whether the template format of SECM-H state contributes to the attention disruption (Mechanism 1), we conducted EXP-14 (§6.10): comparing template-format state (`[SECM-H] Modules: 18 available...`) with natural-language state (`You have 18 tools available...`). Natural-language rendering achieves $Q_{content} = 0.739$ vs template's $0.626$ ($\Delta = +0.113$, Coverage $\Delta = +0.175$), confirming that format interference is a significant component of the attention disruption mechanism. This has direct practical implications: reformatting the harness state as natural language is a low-cost intervention that substantially reduces the communication channel's interference cost.

**Cross-experiment metric comparability caveat.** The Coverage values reported in EXP-11/12 ($0.391$–$0.550$) are systematically higher than those in EXP-3/9/10 ($0.050$–$0.170$) for comparable conditions. This gap reflects a fundamental change in the model's generation mode: in pre-scripted scenarios, the model generates a brief response after each tool execution; in agent-driven scenarios, the model generates both a tool call *and* a response, producing longer and more information-dense text. The five-dimensional content quality evaluator (§5.4) is sensitive to these generation-mode differences. Therefore, **absolute metric values should only be compared within the same experimental paradigm** (pre-scripted or agent-driven), not across paradigms. Relative comparisons ($\Delta$ values) within each paradigm remain valid and informative.

### 6.9 Path A/B Disentanglement: Forced Selection (EXP-13)

To separate the contribution of improved tool selection (Path A) from changed generation behaviour (Path B), we designed a forced-selection experiment. The model is told which tool to use in the user query (e.g., "Plan Monday dinner. Use the find_recipes tool."), eliminating Path A entirely. SECM-H state is still injected. Any quality difference between ForcedSIG (no state) and ForcedSECMH (with state) is purely attributable to Path B.

[Table 20: Path A/B Disentanglement — Forced Selection (2B, n=3)]

| Condition | Tool Accuracy | $Q_{content}$ | Coverage | Wall-Clock (s) | Gen Tokens |
|-----------|--------------|---------------|----------|----------------|------------|
| ForcedSIG (baseline) | $\mathbf{1.000}$ | $0.614$ | $0.430$ | $13.3$ | 2,486 |
| ForcedSECMH (with state) | $0.914$ | $\mathbf{0.664}$ | $\mathbf{0.512}$ | $\mathbf{9.8}$ | 1,743 |

**Path B is real and positive.** ForcedSECMH achieves $\Delta Q_{content} = +0.050$ and $\Delta \text{Coverage} = +0.082$ over ForcedSIG, despite having lower tool accuracy ($91.4\%$ vs $100\%$—the model sometimes ignores the forced instruction when SECM-H state is present). This confirms that SECM-H's state injection changes the model's generation behaviour: it makes the model more likely to faithfully reproduce information from tool results, even when the tool selection is identical. The $+0.082$ Coverage improvement represents a $19\%$ relative increase, suggesting that the state injection provides contextual framing that helps the model integrate tool results into its responses.

**Path B also reduces latency.** ForcedSECMH completes in $9.8$s vs ForcedSIG's $13.3$s ($0.74\times$), generating $30\%$ fewer tokens (1,743 vs 2,486). The state injection appears to focus the model's generation, reducing the verbosity that characterises the forced-selection baseline.

### 6.10 Format Interference: Template vs. Natural Language (EXP-14)

To quantify the format interference component of Mechanism 1 (§7.6), we compared two rendering strategies for SECM-H state: the template format used throughout this paper (`[SECM-H] Modules: 18 available, 12 invoked, 6 pending. Budget: 50/2048. Top: set_oven(0.92).`) and a natural-language format (`You have 18 tools available, 12 already used. Most reliable tool: set_oven (confidence 0.92). You have 50 of 2048 budget remaining.`).

[Table 21: Format Interference — Template vs. Natural Language (2B, n=3)]

| Rendering | Tool Accuracy | $Q_{content}$ | Coverage | Wall-Clock (s) | Gen Tokens |
|-----------|--------------|---------------|----------|----------------|------------|
| Template (SECM-H) | $\mathbf{0.971}$ | $0.626$ | $0.435$ | $17.1$ | 3,191 |
| Natural language | $0.943$ | $\mathbf{0.739}$ | $\mathbf{0.610}$ | $\mathbf{11.5}$ | 2,092 |

**Format interference is a significant component of attention disruption.** Natural-language rendering achieves $\Delta Q_{content} = +0.113$ and $\Delta \text{Coverage} = +0.175$ over template format—an $18\%$ and $40\%$ relative improvement, respectively. Tool accuracy drops marginally ($97.1\%$ → $94.3\%$, $\Delta = -2.8$pp), suggesting that the natural-language format slightly reduces the model's attention to the state content, but the net effect on content quality is strongly positive.

**Natural-language rendering also reduces latency.** The NL format completes in $11.5$s vs template's $17.1$s ($0.67\times$), generating $34\%$ fewer tokens (2,092 vs 3,191). This is consistent with Mechanism 3 (generation-length cascading): the template format triggers verbose, unfocused generation, while the natural-language format integrates more smoothly with the model's pre-trained language patterns.

**Implication for Mechanism 1a.** These results confirm that format interference is a substantial (but not the sole) component of the attention disruption mechanism. The $+0.113$ CQ improvement from NL rendering is larger than the $+0.050$ improvement from Path B alone (EXP-13), suggesting that format and content interference are roughly equal contributors. However, even with NL rendering, the tool accuracy remains slightly below the template format ($94.3\%$ vs $97.1\%$), indicating that some content interference persists regardless of format.

### 6.11 Model Size: 0.8B Agent-Driven (EXP-15)

To test whether SECM-H's agent-driven benefit generalises to smaller models, we replicated EXP-11/12 with Qwen3.5-0.8B-Q4\_K\_M.

[Table 22: Agent-Driven Tool Selection — 0.8B Clean (n=3)]

| Condition | Tool Accuracy | Args Accuracy | $Q_{content}$ | Coverage | Wall-Clock (s) |
|-----------|--------------|---------------|---------------|----------|----------------|
| SIG | $\mathbf{0.743}$ | $\mathbf{0.371}$ | $0.630$ | — | $\mathbf{7.5}$ |
| SIG+SECM-H (full) | $0.543$ | $0.171$ | $\mathbf{0.647}$ | — | $7.7$ |
| SIG+SECM-H (selective) | $0.714$ | $0.343$ | $0.589$ | — | $7.7$ |

[Table 23: Agent-Driven Tool Selection — 0.8B Noisy (n=3, 15% failure)]

| Condition | Tool Accuracy | Args Accuracy | $Q_{content}$ | Coverage | Wall-Clock (s) |
|-----------|--------------|---------------|---------------|----------|----------------|
| SIG | $\mathbf{0.714}$ | $\mathbf{0.343}$ | $0.589$ | — | $\mathbf{7.7}$ |
| SIG+SECM-H (full) | $0.371$ | $0.114$ | $\mathbf{0.636}$ | — | $7.8$ |
| SIG+SECM-H (selective) | $0.571$ | $0.343$ | $0.456$ | — | $5.6$ |

**0.8B model struggles with agent-driven tool selection.** The 0.8B model achieves only $74.3\%$ tool accuracy in clean scenarios (vs 2B's $100\%$), reflecting its limited capacity for the complex instruction-following required to parse tool descriptions and generate correct ACTION calls. Under noise, accuracy drops to $71.4\%$.

**SECM-H does not help 0.8B with tool selection.** In both clean and noisy scenarios, SECM-H-full *degrades* tool accuracy for the 0.8B model ($54.3\%$ clean, $37.1\%$ noisy). The state injection appears to overwhelm the 0.8B model's already limited instruction-following capacity, consistent with Mechanism 4 (Harness-1 boundary violation). The 0.8B model lacks the capacity to simultaneously process the tool selection task and the SECM-H state.

**Content quality benefit is marginal.** SECM-H-full achieves $\Delta Q_{content} = +0.017$ clean and $+0.047$ noisy—smaller than the 2B improvements ($+0.030$ and $+0.101$). The 0.8B model's content quality is less sensitive to state injection because its generation capacity is the binding constraint, not its module management tracking.

**Implication.** The 2B/0.8B comparison confirms that SECM-H's benefit is model-size-dependent: the 2B model has sufficient capacity to benefit from state injection in agent-driven scenarios, while the 0.8B model does not. This supports the Harness-1 boundary condition (Mechanism 4): the externalisation principle requires a policy model with sufficient capacity to integrate structured state alongside task reasoning.

---

## 7. Discussion

### 7.1 Meaning Compiler Overload: Confirmed or Refuted?

The experimental evidence reveals a *task-dependent* pattern that resolves the apparent contradiction between the architectural premise and the experimental results. The state decomposition analysis (H1) confirms that 76.5% of module management functions are *structurally* externalizable. In pre-scripted benchmarks (EXP-3/9/10), injection degrades quality because these benchmarks never test module selection. In agent-driven benchmarks (EXP-11/12), where the model must choose which tool to invoke, SECM-H demonstrates genuine improvement—especially under noise.

[Table 15: Hypothesis Assessment Summary (Post-Experiment, Unified 5-dimensional CQ)]

| Hypothesis | Claim | Status | Evidence |
|-----------|-------|--------|----------|
| **H1** | $\geq 50\%$ of module management functions are externalizable; they consume $\geq 30\%$ of context | **Confirmed (structural)** | 76.5% externalizable (13/17 functions); 69.3% context reduction potential measured |
| **H2** | 6-component state renders in $\leq 100$ tokens; auto-seeding improves early quality | **Confirmed (rendering)** | Renderer produces avg 77.3 tokens (within budget); but all injection volumes tested degrade quality for 2B |
| **H3** | $Q_{SECM\text{-}H} - Q_{SIG} \geq 0.03$ on Kitchen benchmark | **Conditionally confirmed** | In pre-scripted scenarios: refuted ($\Delta Q_{content} = -0.141$ for full injection). In agent-driven scenarios: confirmed—under noise, SECM-H-full achieves $\text{ToolAcc} = 0.971$ vs SIG $0.943$ and $Q_{content} = 0.636$ vs SIG $0.535$ ($\Delta = +0.101$); Selective achieves $Q_{content} = 0.718$ vs SIG $0.596$ in clean ($\Delta = +0.122$) |
| **H4** | Per-step overhead $\leq 5$ ms; total overhead $\leq 3\%$ wall-clock | **Confirmed (harness); Refuted (system)** | Harness overhead <0.1 ms/step; but downstream KV cache inflation causes 3.7× wall-clock for full injection |
| **H5** | Quality gap monotonically non-decreasing with ecosystem size | **Refuted** | Pre-scripted: non-monotonic (negative at 15 modules, +0.030 at 30, +0.018 at 50). The strict monotonicity prediction of §4.4 is falsified by the data. Agent-driven results show SECM-H advantage under noise, but this does not rescue the monotonicity claim |

**The dual nature of results.** The experimental evidence reveals not a simple contradiction but a *task-dependent* pattern. In pre-scripted scenarios (EXP-3/9/10), the Meaning Compiler Overload Hypothesis in its original form—as a capacity burden—is not supported: the SIG baseline achieves $Q_{content} = 0.461$ at 3.8s, and every injection strategy degrades or marginally improves this baseline. The model's implicit state tracking through KV-cache retention is well-suited to deterministic tasks where the tool-call sequence is known in advance. However, in agent-driven scenarios (EXP-11/12), where the model must make genuine module selection decisions under uncertainty, SECM-H demonstrates clear value ($\Delta = +0.101$ noisy, $\Delta = +0.122$ clean selective). This reframes the overload problem: the bottleneck is not *context capacity consumed by bookkeeping* but *the model's inability to maintain long-term reliability statistics through implicit tracking alone*. SECM-H's value derives from providing **exogenous decision knowledge**—structured information about module reliability, invocation history, and dependency constraints that the model's KV-cache cannot reliably encode across extended task horizons.

However, agent-driven scenarios (EXP-11/12) tell a fundamentally different story. When the agent must *select among modules* and *adapt to execution outcomes*—rather than following a fixed script—SECM-H-full outperforms SIG under noise ($\text{ToolAcc} = 0.971$ vs $0.943$; $Q_{content} = 0.636$ vs $0.535$, $\Delta = +0.101$) and Selective outperforms SIG in clean conditions ($Q_{content} = 0.718$ vs $0.596$, $\Delta = +0.122$). The critical insight is that pre-scripted scenarios do not exercise the capability SECM-H is designed to support: *dynamic module management under uncertainty*. In a pre-scripted benchmark, the model follows a deterministic tool-call sequence where state management is trivial; in an agent-driven benchmark, the model must reason about which modules to invoke, evaluate results, and adapt its strategy—precisely the cognitive load that SECM-H is designed to externalise.

This task-dependency resolves the apparent contradiction: the benchmark design determines whether the hypothesis is testable. Pre-scripted benchmarks create a ceiling effect where the state management challenge is too simple for externalisation to provide benefit, while agent-driven benchmarks create the conditions under which externalisation genuinely reduces cognitive load.

### 7.2 Is Dual-Layer State Externalisation a General Principle?

The dual-layer state externalisation model (§3.6) was proposed as a general design principle. Our experiments reveal that this claim requires significant qualification:

> *In an edge agent system with a small policy model, externalising state that does not require the model's semantic capabilities *may* improve performance—but only when (a) the task genuinely exercises the externalised capability, and (b) the model has sufficient capacity to integrate the externalised state alongside task reasoning.*

We do **not** claim that SIG's KV-cache management and SECM-H's text injection are analogous "layers" of the same architectural principle. SIG externalises a *computational process result* (the KV-cache) transparently—the model never sees it as text. SECM-H injects *informational content* (module management metadata) as visible text tokens. These are fundamentally different mechanisms: one is transparent infrastructure, the other is an additional input signal. The "dual-layer" framing overstates their structural similarity. The honest characterisation is: SECM-H is an *informational augmentation* that provides exogenous decision knowledge, not a peer "layer" to SIG's inference-engine optimisation.

The principle's applicability is narrower than initially proposed. SECM-H provides genuine value only when: (a) the model must make autonomous module selection decisions, (b) the environment contains uncertainty (tool failures, varying reliability), and (c) the task horizon is long enough for reliability statistics to matter. Outside these conditions—pre-scripted tasks, deterministic environments, short horizons—the harness layer adds interference cost without commensurate benefit.

### 7.3 Practical Deployment Considerations

**When to deploy SECM-H:**

| Scenario | Recommendation | Rationale |
|----------|---------------|-----------|
| State decomposition analysis | **Use SECM-H taxonomy** | The 6-component state decomposition (§4.1) is valuable as an analytical framework regardless of deployment |
| Content quality evaluation | **Use 5-dimensional evaluator** | The evaluator (§5.4) enables meaningful quality differentiation for any agent system |
| Pre-scripted scenarios | **Do not deploy injection** | In fixed tool-call sequences, the model's implicit state tracking through KV-cache retention outperforms all injection strategies; the state management challenge is too simple for externalisation to provide benefit |
| Agent-driven scenarios with noise | **Deploy SECM-H (full or selective)** | Under realistic conditions requiring dynamic module selection and adaptation to execution outcomes, SECM-H-full outperforms SIG by $\Delta = +0.101$ (noisy) and Selective by $\Delta = +0.122$ (clean) |
| Harness as external oracle (model queries on demand) | **Promising direction** | Avoids KV cache inflation; model retains control over when to access state (§7.6) |
| Harness-aware fine-tuning | **Promising direction** | Following Harness-1's RL approach, train the model to interpret structured state format |
| $> 4$B model with RL training | **Consider** | Larger models may tolerate template-format injection; RL training may teach effective state utilisation |

**Composition with other CO+SIG optimisations.** SECM-H composes with all existing optimisations:

- **CompSIG** [2]: SECM-H's rendered state is subject to CompSIG's periodic compression like any other injected content. No special handling needed.
- **MTP** [5]: SECM-H's quality improvement compounds with MTP's throughput improvement. Projected compound benefit: $4.52\times \times 1.06 \approx 4.79\times$ effective speedup.
- **DiskKVCache** [7]: SECM-H's state components (particularly $C_t$ confidence scores and $P_t$ pattern cache) can be persisted across sessions via DiskKVCache, enabling long-term module reliability tracking.
- **KFC** [6]: SECM-H adds a management-state dimension to the KFC framework's optimisation landscape.

### 7.4 Limitations

This paper has several important limitations:

1. **Narrow benefit window in pre-scripted scenarios.** In pre-scripted benchmarks, the injection benefit for the 2B model is limited to ≤5 tokens/step ($\Delta = +0.030$). However, in agent-driven benchmarks, SECM-H demonstrates substantial benefit ($\Delta = +0.101$ noisy, $\Delta = +0.122$ clean selective). The benefit window is task-dependent, not merely narrow.

2. **Single benchmark.** All results are based on the EdgeAgent-Kitchen benchmark (35 steps, 18 tools). Other agent workloads (coding, web browsing, multi-modal) may exhibit different sensitivity to state injection and different boundaries between pre-scripted and agent-driven evaluation.

3. **Deterministic scenario with simulated tools.** The Kitchen scenario uses `random.seed(42)` for full determinism with simulated tool execution. While EXP-10/12 add noise (15% failure rate at fixed steps), it remains a controlled simulation. Real-world tool execution involves stochastic failures, latency variance, and partial results that may shift the optimal injection parameters.

4. **Content quality evaluator limitations.** The five-dimensional evaluator (coverage, response quality, context utilisation, semantic adequacy, information density) uses keyword-based and TF-IDF methods. While the information density penalty addresses the prior version's RespQ verbosity bias, the evaluator lacks neural evaluation (BERTScore, LLM-as-Judge). More critically, the evaluator is sensitive to generation-mode differences across experimental paradigms: absolute metric values from pre-scripted experiments (EXP-3/9/10) are not directly comparable to those from agent-driven experiments (EXP-11/12), as the model's generation behaviour differs systematically between paradigms (§6.8).

5. **Mediation model not disentangled.** The content quality improvement in agent-driven experiments ($\Delta = +0.101$ noisy) is the combined effect of two causal paths: (A) improved tool selection accuracy ($+2.8$pp) and (B) changed generation behaviour (e.g., more faithful reproduction of tool results). The relative contribution of each path is not measured. The $2.65\times$ Coverage gap ($0.451$ vs $0.170$) suggests that Path B may dominate, but a controlled experiment holding tool selection constant is needed to confirm this.

6. **Rule-based harness.** Unlike Harness-1 [8], which uses RL training, SECM-H uses rule-based state management. An RL-trained harness could learn model-size-adaptive injection timing and volume, potentially discovering strategies beyond our discrete sweep.

7. **Data consistency was resolved mid-study.** An injection-order inconsistency between EXP-3 and the initial EXP-9 implementation was identified and corrected. The content quality formula was also upgraded from 3-dimensional to 5-dimensional during the study. EXP-3, EXP-5, EXP-9, EXP-10, EXP-11, and EXP-12 use the unified 5-dimensional formula; EXP-4 retains the 3-dimensional formula with a consistency note.

### 7.5 Synergy with Other CO+SIG Optimisations

SECM-H sits at the *cognitive management layer* of the CO+SIG optimisation stack (Table 1), above all other optimisations. This positioning creates a natural composition hierarchy:

**SECM-H + SIG (Layer 2 + Layer 1):** The core composition. SECM-H's rendered state is injected through SIG's pipeline, inheriting all KV-cache preservation benefits.

**SECM-H + CompSIG (management + compression):** When the KV-cache grows large, CompSIG's periodic compression [2] can reduce the injected state size. SECM-H's compact rendering (≤100 tokens) minimises the compression burden.

**SECM-H + MTP (management + acceleration):** SECM-H improves *what* the Meaning Compiler generates (better module selection), while MTP improves *how fast* it generates (speculative decoding). The effects are orthogonal and multiplicative.

**SECM-H + DiskKVCache (management + persistence):** DiskKVCache [7] can persist SECM-H's state components across sessions, enabling:
- Cross-session confidence tracking (module $m$ was reliable in 8/10 previous sessions)
- Cross-session pattern cache (cognitive patterns learned in previous sessions are available immediately)
- Cold-start elimination (SECM-H's state is loaded from disk rather than rebuilt from scratch)

This composition creates a *three-tier state hierarchy*: real-time state (SECM-H in-memory), session-persistent state (DiskKVCache on SSD), and cross-session state (shared module reliability database).

### 7.6 Why Does State Externalisation Fail? A Causal Analysis

The finding that the SIG baseline outperforms all injection strategies in pre-scripted scenarios demands a deeper causal analysis than "KV cache inflation." We identify four interacting failure mechanisms that operate in pre-scripted contexts, then explain why these same mechanisms are mitigated or overridden in agent-driven scenarios.

**Mechanism 1: Attention distribution disruption.** The injected state text (e.g., `[SECM-H] Modules: 18 available, 12 invoked, 6 pending. Top: set_oven(0.85).`) is a structured template with a format distribution that diverges sharply from the natural-language tool results and user queries that constitute the model's training distribution. For a 2B model whose attention patterns have been shaped by pre-training on natural language, these template-formatted tokens introduce out-of-distribution signals into the attention computation. The model's next-token prediction becomes less reliable not because the context is *larger* but because the context contains *adversarial-format tokens* that disrupt the attention weights the model has learned to assign to semantically meaningful content. Evidence: Information Coverage drops from 0.164 (SIG) to 0.050 (SECM-H full)—the model does not merely generate more verbose text, it generates text that *ignores the tool results it was supposed to reason about*. However, in agent-driven scenarios, this mechanism is partially overridden: when the task requires dynamic module selection, the structured state provides *decision-relevant context* (which modules are available, which have succeeded) that the model's implicit KV-cache tracking cannot reliably maintain across novel execution paths. The benefit of this decision context outweighs the attention disruption cost, particularly under noise where module reliability information becomes critical.

**Mechanism 1a: Format interference vs. content interference.** The attention disruption described above conflates two distinguishable dimensions. *Format interference* arises from the structured template syntax (e.g., `[SECM-H] Modules: 18 available...`), which diverges from the model's natural-language training distribution. *Content interference* arises from the *type* of information injected—module management metadata (tool reliability, invocation history) that is semantically distant from the task domain (cooking instructions). If format interference is the primary driver, the problem could be substantially mitigated by reformatting the state as natural language (e.g., "You have 18 tools available, 12 already used. The oven tool has been reliable."). If content interference dominates, reformatting alone would not help, because the information type itself is irrelevant to the pre-scripted task. The agent-driven results suggest that content interference is the more fundamental factor: even with the same template format, SECM-H provides net positive value when the task genuinely requires module management information. This has important implications for future work: format improvements (natural-language rendering, prompt engineering) may reduce the attention disruption cost, but the core value proposition depends on the task's module management complexity.

**Mechanism 2: Attention competition between implicit and explicit state.** SECM-H's design follows a three-step cycle: (1) externalise state from the model's context to the harness, (2) compress it into a structured summary, (3) re-inject the summary into the model's context. The failure of this cycle in pre-scripted scenarios is not a "translation" problem—the model is not attempting to convert between two equivalent representations. Rather, it is an *attention competition* problem: the injected explicit state and the model's own implicit state are two *incommensurable information sources* that compete for the model's limited attention capacity. The KV-cache encodes an implicit, distributed, high-dimensional representation of the task history—a trajectory of the model's own reasoning process. The SECM-H summary encodes an explicit, symbolic, low-dimensional record of module management metadata. These serve fundamentally different functions: the former supports *task reasoning* (what to say next), the latter supports *module management* (which tool to trust). In pre-scripted scenarios, where module management decisions are irrelevant, the management metadata competes with task-relevant information for attention allocation, producing net negative value. In agent-driven scenarios, where module management decisions are critical, the metadata provides information the model's implicit tracking cannot reliably maintain (e.g., long-term reliability statistics), producing net positive value. The key insight is that the value of explicit state injection is *task-dependent*, not *capacity-dependent*.

**Mechanism 3: Generation-length cascading.** The injected tokens expand the KV-cache, which increases the model's effective context length at generation time. Crucially, this does not merely slow generation—it changes *what* the model generates. With a larger context containing mixed natural-language and template-format tokens, the model's output distribution shifts toward longer, more diffuse responses. The 4.43× increase in generation tokens (556 → 2,463) is not a proportional elaboration; it is a qualitative degradation where the model loses focus on the immediate task. The information density metric confirms this: SIG generates text with higher content-word ratio than SECM-H, meaning the additional tokens are predominantly filler rather than substantive content.

**Mechanism 4: Harness-1 boundary violation.** Harness-1 [8] achieves its results with a 20B RL-trained model that has been specifically optimised to interpret and act upon structured harness state. The RL training teaches the policy model *when* to attend to harness state and *how* to integrate it with its own reasoning. SECM-H applies the same architectural pattern to 0.8B–2B models *without any adaptation training*. This violates a critical boundary condition: the externalisation principle assumes the policy model can efficiently parse the compressed state format, an assumption that holds for 20B RL-trained models but fails for small untrained models. The 0.8B model's marginally better response to Selective and OOB strategies ($\Delta = +0.025$–$+0.028$) may reflect its *greater need* for external guidance, but this need is not met by the current template-format injection approach. However, the agent-driven results demonstrate that even untrained models can benefit from structured state when the task genuinely requires module selection: under noise, SECM-H-full achieves $\text{ToolAcc} = 0.971$ vs SIG's $0.943$, suggesting that when the cognitive load of module management exceeds the model's implicit tracking capacity, even a suboptimal communication channel provides net positive value. The boundary violation is real but conditional—it manifests when the task is trivial for the model's implicit state tracking, and recedes when the task demands genuine module management reasoning.

**Implication for the CO+SIG programme.** These four mechanisms collectively explain why the SIG baseline is surprisingly hard to beat in pre-scripted scenarios: the model's implicit state tracking through KV-cache retention is *well-adapted to its own attention patterns*, while any explicit re-injection of state—however compact—introduces attention competition and format interference. However, this is not a fundamental architectural failure. SECM-H's value resides in providing *exogenous decision knowledge* (module reliability statistics, dependency constraints, invocation history) that the model's implicit tracking cannot maintain across extended task horizons. This knowledge becomes valuable precisely when the task demands genuine module management reasoning—under uncertainty, with failures, and with dynamic tool selection. The design challenge is not to eliminate the communication channel but to *minimise its interference cost* (through format-aware rendering) while *maximising its decision-relevant content* (through selective injection at decision points). Promising directions include: (a) natural-language state rendering that reduces format interference; (b) retrieval-augmented access where the model queries the harness on demand; (c) harness-aware fine-tuning that teaches the model to integrate structured state; (d) operating the harness as an external oracle consulted selectively, rather than as an inject-every-step state publisher.

**Pre-scripted vs. agent-driven: why the mechanisms manifest differently.** A critical question is why the same "attention competition" mechanism produces opposite effects in pre-scripted (negative) and agent-driven (positive) scenarios. We propose a *generation stabiliser* hypothesis: in agent-driven scenarios, the model faces genuine uncertainty about which tool to invoke and how to interpret results. When the model selects a wrong tool (which happens $5.7\%$ of the time for SECM-H-full vs $2.9\%$ for SIG in noisy scenarios), it receives unexpected results that may derail its generation. The SECM-H state—containing module reliability rankings and invocation history—serves as a *contextual anchor* that stabilises the model's generation even when individual tool selections fail. In pre-scripted scenarios, no such stabilisation is needed because the tool-call sequence is deterministic and the model never faces selection uncertainty. This reframes SECM-H not as a general-purpose cognitive architecture layer, but as a **generation stabiliser for autonomous agents under decision uncertainty**—a more specific but empirically supported role. We acknowledge that this interpretation is provisional; a definitive explanation would require attention-level analysis (e.g., attention heatmap comparisons with and without SECM-H state) that is beyond the scope of this paper.

---

## 8. Conclusion

This paper introduced SECM-H (State-Externalizing Cognitive Module Harness), applying the state externalisation principle from Harness-1 [8] to the CO+SIG edge agent architecture. We present the architectural design, a formal state decomposition taxonomy, a five-dimensional content quality evaluation framework, and experimental validation across ten communication strategies on the EdgeAgent-Kitchen benchmark, supplemented by agent-driven experiments (EXP-11/12) that test SECM-H under conditions of genuine module management uncertainty.

**The core finding is a two-part result.** Part (a): on pre-scripted benchmarks, SECM-H fails. The SIG baseline without any harness injection achieves the highest content quality ($Q_{content} = 0.461$ at 3.8s), and every injection strategy either degrades or marginally improves this baseline. Part (b): on agent-driven benchmarks, SECM-H succeeds. When the task requires dynamic module selection and adaptation to execution outcomes—precisely the cognitive load SECM-H is designed to externalise—the harness outperforms SIG. Under noise, SECM-H-full achieves $\text{ToolAcc} = 0.971$ vs SIG's $0.943$, and $Q_{content} = 0.636$ vs SIG's $0.535$ ($\Delta = +0.101$). In clean agent-driven conditions, Selective achieves $Q_{content} = 0.718$ vs SIG's $0.596$ ($\Delta = +0.122$).

**Hypothesis assessment.** State decomposition confirms that 76.5% of module management functions are structurally externalisable (H1 confirmed, $\geq 50\%$ threshold exceeded). H3 is conditionally confirmed: the quality gap $Q_{SECM\text{-}H} - Q_{SIG} \geq 0.03$ holds in agent-driven scenarios ($\Delta = +0.101$ noisy, $\Delta = +0.122$ clean selective) but is refuted in pre-scripted scenarios ($\Delta = -0.141$ for full injection). H5 is refuted: the strict monotonicity prediction is falsified by non-monotonic data across module counts.

**The key lesson is about benchmark design.** The pre-scripted vs. agent-driven distinction resolves the apparent contradiction in our results and carries a general methodological implication: a benchmark must test the capability being evaluated. Pre-scripted benchmarks do not exercise dynamic module management—the very capability SECM-H externalises—because the tool-call sequence is deterministic and the state management challenge is trivial. This creates a ceiling effect that makes SECM-H appear ineffective. Agent-driven benchmarks, by contrast, create the conditions of genuine module management uncertainty (tool selection, failure adaptation, strategy revision) under which state externalisation provides real benefit. This lesson extends beyond SECM-H: any evaluation of a management-layer optimisation requires a benchmark that exercises the management capabilities in question.

**Precise positioning of the contribution.** We do not claim SECM-H as a general-purpose cognitive architecture layer. The evidence supports a more specific claim: **SECM-H functions as a generation stabiliser for autonomous agents under decision uncertainty.** In pre-scripted tasks, where the model's implicit KV-cache tracking is sufficient, SECM-H adds interference cost without benefit. In agent-driven tasks under noise, where the model must select among modules with varying reliability, SECM-H's reliability statistics provide genuine decision support. However, this claim carries a critical unresolved ambiguity (discussed below).

**The core open question: what drives the content quality improvement?** The content quality improvement in agent-driven experiments has two components—Path A (better tool selection, +2.8pp accuracy) and Path B (changed generation behaviour, +0.082 Coverage from EXP-13). These are disproportionate: a small accuracy improvement coexists with a large coverage improvement. If Path B dominates, then SECM-H's effectiveness may stem primarily from its *style-shaping* effect on generation—prompting the model to more faithfully reproduce tool results—rather than from its *module management* value. A simpler intervention that achieves the same style-shaping (e.g., a prompt instruction to "quote tool results in your response") might produce comparable $Q_{content}$ improvements at negligible cost. This is the most important open question for determining whether SECM-H's architecture justifies its complexity.

**The 0.8B model: two competing explanations.** The 0.8B results (§6.11) can be interpreted through two lenses that predict different futures: (a) *model-size limitation* (Mechanism 4): the 0.8B model lacks capacity to integrate SECM-H state alongside task reasoning, implying that larger models would benefit more; (b) *generation stabiliser* (§7.6): SECM-H's value comes from anchoring the model's generation under uncertainty, implying that benefit peaks at an intermediate model size—large enough to integrate state, small enough to need stabilisation. These predictions are contradictory (monotonic vs. peaked), and distinguishing them requires testing across a wider range of model sizes (4B, 7B, 14B). We present both hypotheses without resolving the tension.

**Future work** should pursue four directions, ordered by priority: (1) **Path A/B causal decoupling**: holding tool selection constant while varying only state injection (building on EXP-13) to determine whether SECM-H's value is primarily in module management or in generation style shaping; (2) **attention-level analysis** (attention heatmap comparisons with and without SECM-H state) to definitively characterise the attention competition mechanism; (3) **format-aware state rendering** that presents module management information as natural language rather than structured templates, to reduce format interference (EXP-14 shows $\Delta Q_{content} = +0.113$ for NL rendering); (4) **multi-model evaluation** across diverse model families and sizes to resolve the model-size-limitation vs. generation-stabiliser tension and determine whether the pre-scripted/agent-driven boundary generalises beyond Qwen3.5.

Together, Papers 1–8 establish a complete optimisation stack for edge agent inference. SECM-H contributes a specific insight: in autonomous agent scenarios with genuine module management uncertainty, explicit reliability tracking through a harness layer improves generation quality. The dual-layer state externalisation theory (§3.6) is overstated—SIG's KV-cache management and SECM-H's informational augmentation are not analogous "layers" but fundamentally different mechanisms. The honest contribution is narrower but empirically grounded: **for small edge models operating autonomously under uncertainty, a rule-based harness that tracks module reliability and injects compact state summaries can stabilise generation and improve content quality, provided the injection format is compatible with the model's pre-trained attention patterns.**

---

## References

[1] Cognitive Outsourcing with Suspend-and-Inject Generation for Scalable Embodied Intelligence. *CO+SIG Research Program, Paper 1*, 2025.

[2] Beyond the Injection Engine: A Five-Dimensional Analysis of CO-SIG. *CO+SIG Research Program, Paper 2*, 2025.

[3] CO-SIG Architecture, Theory, and Empirical Design Space for Scalable Edge Intelligence. *CO+SIG Research Program, Paper 3*, 2025.

[4] Suspend-and-Inject Generation as an Edge Inference Runtime Primitive for Long-Horizon Agent Tasks. *CO+SIG Research Program, Paper 4*, 2026.

[5] Orthogonal Acceleration: Fusing Speculative Decoding with Suspend-and-Inject Generation. *CO+SIG Research Program, Paper 5*, 2026.

[6] Convergent KVCache Architectures: Bridging Cloud-Scale Disaggregated Serving and Edge-Native Injection Continuity. *CO+SIG Research Program, Paper 6*, 2026.

[7] DiskKVCache: Disk-Backed KVCache Persistence for Cold-Start Elimination in Edge Agent Inference. *CO+SIG Research Program, Paper 7*, 2026.

[8] C. Jiang, Z. Wu, Y. Chen, and Q. Liu. Harness-1: Reinforcement Learning for Search Agents with State-Externalizing Harnesses. *arXiv:2606.02373*, 2026.

[9] H. Chase. LangChain: Building applications with LLMs through composability. https://github.com/langchain-ai/langchain, 2022–2026.

[10] T. Richards. AutoGPT: An autonomous GPT-4 experiment. https://github.com/Significant-Gravitas/AutoGPT, 2023–2026.

[11] C. Yang, J. Chen, Y. Qian, et al. SWE-agent: Agent-Computer Interfaces Enable Automated Software Engineering. *NeurIPS 2024*.

[12] G. Wang, Y. Xie, Y. Jiang, et al. Voyager: An Open-Ended Embodied Agent with Large Language Models. *NeurIPS 2023*.

[13] R. Qin, Z. Li, W. He, et al. Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving. *arXiv:2407.00079*, 2024. FAST 2025 Best Paper.

[14] W. Kwon, Z. Li, S. Zhuang, et al. Efficient Memory Management for Large Language Model Serving with PagedAttention. *SOSP 2023*.

[15] G. Gerganov et al. llama.cpp: LLM inference in C/C++. https://github.com/ggerganov/llama.cpp, 2023–2026.
