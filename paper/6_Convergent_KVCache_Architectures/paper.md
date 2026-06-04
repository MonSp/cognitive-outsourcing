# Convergent KVCache Architectures: Bridging Cloud-Scale Disaggregated Serving and Edge-Native Injection Continuity

> **SIG/CO Research Program — Paper 6** | June 2026
>
> Preceding papers: [1] *Cognitive Outsourcing with Suspend-and-Inject Generation for Scalable Embodied Intelligence*, [2] *Beyond the Injection Engine: A Five-Dimensional Analysis of CO-SIG*, [3] *CO-SIG Architecture, Theory, and Empirical Design Space for Scalable Edge Intelligence*, [4] *Suspend-and-Inject Generation as an Edge Inference Runtime Primitive for Long-Horizon Agent Tasks*, [5] *Orthogonal Acceleration: Fusing Speculative Decoding with Suspend-and-Inject Generation*.
>
> **Date**: June 2026

---

## Abstract

Modern LLM inference systems have independently converged on a single organising principle: treating the key-value cache (KVCache) as a first-class resource rather than a transient byproduct of computation. Mooncake, a cloud-scale disaggregated serving architecture (FAST 2025 Best Paper), achieves high throughput by separating prefill and decode into independent clusters connected through a global KVCache pool with RDMA transfer, prefix-hash matching, and multi-tier storage. Suspend-and-Inject Generation (SIG), an edge-native inference primitive, achieves low latency by preserving the KVCache across tool-call boundaries through injection continuity, eliminating 73–97% of redundant prefill computation. Despite targeting opposite deployment regimes—multi-tenant cloud clusters versus single-user edge devices—both systems embody the same design philosophy: "trading storage for computation" (以存换算). This paper formalises the KVCache-as-First-Class-Citizen (KFC) framework, a unified optimisation objective that subsumes both Mooncake and SIG as regime-specific instantiations. We present an 8-dimensional convergence analysis revealing 5 dimensions of strong or moderate convergence, a transferability assessment of four Mooncake design patterns for edge deployment, and a conditional feasibility analysis of composing edge injection continuity with cloud prefix caching. An analytical model demonstrates that SIG captures 96–99.8% of achievable prefill reduction in edge agent workloads as a standalone mechanism; prefix caching provides conditional incremental benefit (0.23–3.82% of SIG savings, depending on shared prefix size) that is material only for multi-session scenarios with extensive shared tool descriptions ($P_s + P_a \geq 500$ tokens, $N \geq 5$ sessions), with an interaction term of $(N-1)(P_s + P_a)$ tokens. We characterise the KVCache format translation overhead between llama.cpp and vLLM PagedAttention (measured at ~2.7 ms on GPU for 1024-token sessions with FP16→FP16, confirming the ~1–7 ms estimate for GPU execution) and derive break-even conditions for hybrid superiority over pure-edge cold-start inference (measured at ~4.1 Gbps for GQA models, close to the original 3.6 Gbps estimate). An end-to-end migration PoC confirms the five-step pipeline is functional, with serialisation (not network bandwidth) as the bottleneck at 256+ tokens. Monte Carlo jitter analysis reveals no jitter level achieves 95% hybrid reliability at mean 3 Gbps. Entropy-based privacy grading (three tiers: public/internal/sensitive) retains 65.9% of KVCache locally, confirming the binary 67.6% estimate. Extended interaction validation confirms the interaction term $(N-1)(P_s + P_a)$ across $N = 1$–$50$ sessions on Qwen3.5-0.8B and 4B (GQA), with FP16→FP8 compression shown to be content-invariant (50% across code, JSON, and natural language). This paper extends Paper 1's positioning of CO+SIG from non-competition with cloud frameworks to architectural convergence with them, providing an empirically grounded analysis of when hybrid composition is feasible and when SIG alone is sufficient.

---

## 1. Introduction

### 1.1 The KVCache Convergence Observation

A remarkable pattern has emerged in the independent evolution of LLM inference systems. Two research communities—cloud-scale serving and edge-native inference—have arrived at the same foundational insight without cross-pollination: the key-value cache (KVCache) is the central resource to optimise, not a transient byproduct of the transformer forward pass.

On the cloud side, Mooncake [6] (arXiv:2407.00079, FAST 2025 Best Paper) demonstrated that reorganising the entire serving architecture around the KVCache lifecycle—its creation, storage, transfer, reuse, and eviction—yields dramatic throughput improvements. By disaggregating prefill and decode into independent clusters, connecting them through a global KVCache pool with zero-copy RDMA transfer, and implementing prefix-hash matching for cross-request cache reuse, Mooncake achieved 525% throughput increase under overload and serves production workloads at Kimi K2 scale (128 H200 GPUs, 224k tok/s prefill, 288k tok/s decode). The organising principle is explicit: "trading storage for computation" (以存换算)—expand KVCache storage from VRAM to DRAM to SSD, increasing cache hit rates and reducing redundant GPU computation.

On the edge side, Suspend-and-Inject Generation (SIG) [1] demonstrated that preserving the KVCache across tool-call boundaries—rather than discarding and recomputing it—eliminates 73–97% of redundant prefill tokens [3] and yields 2.38–9.45× speedups across architectures. SIG's injection continuity operates on the same principle: rather than recompute the KVCache from scratch at each tool invocation, preserve it and extend it incrementally. The cost savings are substantial—on a 4B model running a 30-step tool chain, SIG reduces prefill from 5.5 seconds to 0.1 seconds, a 55-fold reduction [4].

Both systems, developed independently and targeting opposite deployment regimes, converged on the same insight: **KVCache preservation is preferred over KVCache recomputation**. This convergence is not superficial—it extends to the core architectural decisions: both treat KVCache transfer as a first-class operation, both expand storage beyond VRAM, and both make scheduling decisions that are KVCache-centric.

### 1.2 Problem Statement

Despite this convergence, three gaps prevent the realisation of a unified KVCache architecture:

1. **Convergence without formalisation.** The shared design philosophy between Mooncake and SIG is implicit—neither system was designed with awareness of the other, and no formal framework captures their commonalities and differences. Without formalisation, the convergence remains an observation rather than a design principle that can guide future systems.

2. **Orthogonal but asymmetric mechanisms.** Mooncake's prefix-hash matching excels at cross-request KVCache reuse (shared system prompts across users) but provides negligible benefit in the deep-chain regime where Paper 4 proved prefix reuse falls below 3% [4]. SIG's injection continuity excels at within-request KVCache preservation (deep tool chains) and captures 96–99.8% of achievable prefill reduction as a standalone mechanism. The mechanisms operate on orthogonal axes (cross-session vs. within-session), but their contributions are highly asymmetric: SIG is the dominant mechanism, with prefix caching providing marginal additive benefit ($\leq 4\%$) conditional on multi-session scenarios with large shared prefixes.

3. **Hybrid architecture gap.** No existing system provides a unified KVCache architecture that spans the edge-cloud boundary. Edge devices running SIG cannot share KVCache with cloud clusters running Mooncake; cloud clusters cannot delegate injection-continuous workloads to edge devices. This gap prevents the realisation of a seamless edge-cloud LLM inference stack where KVCache flows across regimes as fluidly as computation does today.

### 1.3 Contributions

This paper makes the following contributions:

1. **The KFC framework** (§3): A formal "KVCache-as-First-Class-Citizen" optimisation framework that subsumes both Mooncake and SIG as regime-specific instantiations of a unified objective—minimise redundant KVCache computation subject to latency and storage constraints, where the deployment regime determines the feasible solution space. We provide a parameterised objective function, a regime parameterisation, an architectural decision tree, and a subsumption proof.

2. **8-dimensional convergence analysis** (§4): A systematic comparison of Mooncake and SIG across eight architectural dimensions—KVCache creation, computation-storage tradeoff, scheduling, cache sharing scope, transport, deployment scale, optimisation objective, and failure handling—revealing 5 dimensions of strong or moderate convergence and 3 dimensions of regime-determined divergence.

3. **Transferability analysis** (§5): An assessment of four Mooncake design patterns (prefix-hash matching, hot/cold scheduling, multi-tier storage, PD disaggregation) for transferability to SIG's edge deployment, identifying hard constraints (PD disaggregation is fundamentally incompatible with single-device deployment) and soft constraints (prefix-hash matching requires engineering effort but is feasible). We also identify reverse transferability: SIG's injection continuity model can improve Mooncake's handling of long-running agent sessions.

4. **Hybrid architecture feasibility analysis** (§6): A conditional feasibility analysis of a three-component architecture (Edge SIG Node, Cloud Mooncake Cluster, Edge-Cloud Gateway), demonstrating that composing SIG with cloud prefix caching is technically feasible but yields marginal benefit (0.23–3.82% of SIG savings) at substantial engineering cost (~9,000 lines of code, 16–23 person-months). We characterise the KVCache format translation overhead (measured at ~2.7 ms on GPU for 1024-token sessions), derive break-even bandwidth (~4.1 Gbps for GQA models), and identify the feasible domain: hybrid outperforms pure-edge at $\geq 5$ Gbps with $\geq 256$-token prefixes. We recommend local KVCache persistence as the more practical path for cold-start elimination.

5. **Asymmetric interaction analysis** (§7): An analytical model quantifying the marginal benefit of adding prefix caching to SIG: an interaction term of $(N-1)(P_s + P_a)$ tokens representing cold-start elimination for sessions 2 through $N$. The model reveals that SIG captures 96–99.8% of achievable savings as a standalone mechanism, with the interaction term contributing 0.23–3.82% in typical configurations. Empirical validation confirms 18–21% savings at $N=10$ with medium prefixes (60 tokens), translating to ~100 ms absolute improvement. The interaction is material only in the narrow sweet spot of $N \geq 5$, $P_s + P_a \geq 500$.

6. **KFC generative validation** (§6.8): Application of the KFC decision tree to the connected-edge scenario ($C=1, B>0$), predicting a novel "SIG + Cloud Prefix Prefetch" mode not present in either Mooncake or SIG, with entropy-based privacy grading (three tiers: public/internal/sensitive) showing 65.9% of KVCache can be retained locally.

7. **End-to-end migration PoC** (§6.7b–d): A five-step KVCache migration pipeline measuring extract, translate, transfer, translate-back, and restore latency on Qwen3.5-0.8B and 4B. The CPU-baseline PoC reveals serialisation as the bottleneck at 256+ tokens; a CUDA-optimised projection (using GPU D2H measurements at 28–94 GB/s) shifts the bottleneck to network transfer and identifies the feasible domain: hybrid wins at $\geq 5$ Gbps with $\geq 256$-token prefixes. Monte Carlo jitter analysis at 10 Gbps shows P(hybrid wins) $\geq 99.67\%$ with CUDA optimisation.

8. **Roadmap** (§8): A three-horizon roadmap for SIG's evolution from edge-only to distributed with Mooncake transport, identifying the technical requirements and dependencies at each stage.

### 1.4 Positioning Within the Research Program

Paper 1 [1] established CO+SIG as an edge-only architecture, explicitly stating: "CO+SIG is not intended to compete with cloud-scale serving frameworks (vLLM, TensorRT-LLM) that already exploit FlashAttention and RadixAttention to reduce prefill to near-negligible levels; rather, CO+SIG targets the qualitatively different regime of single-user, single-instance edge inference where these server-side optimisations are inapplicable." Papers 2–5 [2, 3, 4, 5] deepened the understanding of SIG within the edge regime—theoretical foundations [2], empirical design space [3], runtime primitive characterisation [4], and composability with speculative decoding [5].

This paper is the first in the program that looks *outward* from the edge regime. It extends Paper 1's positioning from *non-competition* to *architectural convergence*: SIG and Mooncake share a common KVCache-first design philosophy, formalised as the KFC framework. However, the practical implications of this convergence are asymmetric—SIG captures 96–99.8% of achievable edge-side savings as a standalone mechanism, and the composition of SIG with cloud prefix caching is technically feasible (§6) but yields marginal benefit (0.23–3.82%) at substantial engineering cost. The contribution of this paper is the *formalisation of convergence* (KFC framework, 8-dimensional analysis), not the *advocacy of hybrid composition*. We demonstrate that a hybrid is feasible under specific conditions ($\geq 5$ Gbps, $\geq 256$-token prefixes) but not generally necessary, and recommend local KVCache persistence as the practical path for most edge deployments.

---

## 2. Background

### 2.1 Mooncake: A KVCache-Centric Disaggregated Architecture

Mooncake [6] is a cloud-scale LLM serving system developed by Moonshot AI and Tsinghua University that reorganises the entire serving architecture around the KVCache lifecycle. Its core insight is that in modern LLM serving, memory (KVCache) is the binding constraint, not computation (FLOPs), and that the architecture should be designed accordingly.

**PD disaggregation.** Mooncake separates the prefill phase (compute-intensive, processing all input tokens in parallel) from the decode phase (memory-intensive, generating one token at a time with the full KVCache in memory) into independent clusters. Prefill nodes are optimised for throughput (maximising cache reuse across requests with shared prefixes), whilst decode nodes are optimised for memory efficiency (maximising the number of concurrent sessions per GPU). This spatial separation eliminates the prefill-decode interference that plagues co-located designs, where a long prefill request blocks decode requests and violates time-between-tokens (TBT) service-level objectives.

**Global KVCache pool.** Mooncake leverages underutilised CPU, DRAM, and SSD resources across the GPU cluster to form a global KVCache pool. KVCache is stored as paged blocks with prefix-hash matching, enabling cross-request, cross-node cache sharing. The pool spans three storage tiers: VRAM (fastest, smallest), DRAM (medium speed, medium size), and SSD (slowest, largest). The "trading storage for computation" principle holds because the cost of storing more KVCache is less than the cost of recomputing it—expanding storage from VRAM-only to VRAM + DRAM + SSD increases cache hit rates from approximately 50% to much higher.

**Transfer Engine.** Mooncake implements zero-copy RDMA data transfer for KVCache migration between nodes, supporting GPUDirect, NVLink, CXL, and TCP protocols. Multi-NIC aggregation achieves 8×400 Gbps bandwidth. The Transfer Engine is topology-aware, dynamically selecting the optimal transfer path based on network conditions. This enables fast KVCache transfer from prefill nodes to decode nodes (typically 10–100 μs within a rack via RDMA).

**Conductor scheduler.** A global, KVCache-centric scheduler that makes three types of decisions: (a) cache-aware prefill scheduling—placing new requests on prefill nodes that maximise prefix reuse; (b) KVCache balance scheduling—replicating hot KVCache blocks across nodes and evicting cold blocks; and (c) load-balance decode scheduling—distributing decode requests across decode nodes to meet TBT SLOs. Under overload, the Conductor implements prediction-based early rejection, achieving 525% throughput increase in simulated scenarios and handling 75% more requests under real workloads.

**Ecosystem integration.** As of 2026, Mooncake has been integrated into the major LLM serving frameworks: vLLM (Mooncake Store as distributed KVCache pool backend, Transfer Engine as KV Connector for PD disaggregation), SGLang (HiCache backend, EPD disaggregation), TensorRT-LLM (Transfer Engine for KVCache transfer), and LMDeploy (PD disaggregation backend). Mooncake is also an official member of the PyTorch ecosystem. The Kimi K2 deployment uses 128 H200 GPUs to achieve 224k tok/s prefill and 288k tok/s decode.

### 2.2 SIG: Suspend-and-Inject Generation for Edge Inference

Suspend-and-Inject Generation [1] is an inference-engine-level primitive that preserves KVCache continuity across tool-call boundaries. Rather than discarding the KVCache and recomputing it from scratch at each tool invocation (the standard application-loop approach), SIG retains the KVCache and extends it incrementally with the tool result.

**Five-stage cycle.** SIG operates through a suspend-inject-resume cycle: (1) Suspend—pause autoregressive decoding when a tool-call marker is detected; (2) Resolve—parse the tool request; (3) Fetch—invoke the external module; (4) Inject—tokenise the tool result, wrap it in a stabilisation template, and execute a forward pass with the suspended KVCache as prefix, extending the cache with the injected tokens; (5) Resume—continue autoregressive decoding from the extended cache. Because only the injected tokens undergo prefill, the cost is linear in the injection size and independent of the total conversation length.

**Key results from the CO+SIG research program.** Paper 1 [1] demonstrated that SIG eliminates up to 96% of prefill tokens, yielding 3.85× end-to-end speedup on a 4B GPU. Paper 2 [2] established that SIG and AppLoop generate tokens at nearly identical per-token rates (108 vs. 103 tok/s), confirming that SIG's speedup is purely from KVCache preservation, and introduced CompSIG (periodic KVCache compression achieving 61% reduction with 17% overhead). Paper 3 [3] validated SIG across architectures (Qwen3.5 dense, Nemotron-3-Nano hybrid Mamba+attention, Gemma 4 GQA), demonstrating 73–97% prefill savings and 2.38–9.45× speedups, with Batch-SIG proving architecture-independent (4.24–6.82× vs. AppLoop-PC). Paper 4 [4] characterised SIG as a runtime primitive, demonstrating deep-chain advantage of 2.79× (0.8B) to 4.26× (4B) at 30-tool depth, and proving that prefix caching provides less than 3% token reuse in the deep-chain regime. Paper 5 [5] demonstrated that SIG composes with speculative decoding via multi-token prediction (MTP), achieving 4.52× compound speedup with an orthogonality ratio ρ = 1.239.

**Edge-only positioning.** Paper 1 explicitly positioned CO+SIG as targeting the "qualitatively different regime of single-user, single-instance edge inference" where "FlashAttention is frequently unavailable due to hardware, driver, or framework constraints" and "the inference engine is a lightweight library such as llama.cpp rather than a full serving stack." This positioning is maintained and extended in this paper—we do not argue that SIG should replace cloud serving systems; rather, we show that SIG and cloud systems converge on the same design philosophy, with SIG capturing 96–99.8% of achievable edge-side savings as a standalone mechanism.

---

## 3. The KFC Framework: Unified KVCache-Centric Optimisation

### 3.1 Core Principle

The KVCache-as-First-Class-Citizen (KFC) framework posits that LLM inference system design should be organised around the lifecycle of the KVCache—its creation, storage, transfer, reuse, and eviction—rather than around the computation graph of the transformer forward pass. This inversion of the traditional compute-centric design paradigm reflects the empirical reality that in modern LLM serving, memory (KVCache) is the binding constraint, not computation (FLOPs).

### 3.2 Formal Objective Function

We formalise the KFC principle as a constrained optimisation:

$$\min_{\mathcal{A}} \Phi(\mathcal{A}) = \sum_{r \in \mathcal{R}} \alpha \cdot C_{\text{prefill}}(r, \mathcal{A}) + \beta \cdot C_{\text{transfer}}(r, \mathcal{A}) + \gamma \cdot C_{\text{storage}}(r, \mathcal{A})$$

Subject to:
- $\text{TTFT}(r) \leq L_{\text{TTFT},r}$ (time-to-first-token latency bound)
- $\text{TBT}(r) \leq L_{\text{TBT},r}$ (time-between-tokens latency bound)
- $\sum_{r} \text{KVCache}(r) \leq S_{\max}$ (storage budget)
- $\text{Throughput} \geq T_{\min}$ (minimum throughput, cloud regime only)

Where:
- $\mathcal{A}$ = set of architectural decisions (PD disaggregation, prefix caching, injection continuity, multi-tier storage, scheduling policy, etc.)
- $\mathcal{R}$ = request set
- $C_{\text{prefill}}(r, \mathcal{A})$ = cost of computing KVCache entries from input tokens for request $r$ under architecture $\mathcal{A}$
- $C_{\text{transfer}}(r, \mathcal{A})$ = cost of moving KVCache between storage tiers or nodes for request $r$
- $C_{\text{storage}}(r, \mathcal{A})$ = cost of storing KVCache (memory/SSD allocation, compression overhead)
- $\alpha, \beta, \gamma$ = regime-dependent weights

**Interpretation.** The objective minimises the total cost of KVCache management—computation to create it, transfer to move it, and storage to hold it—subject to latency and capacity constraints. The deployment regime determines which cost term dominates and which constraints are binding.

### 3.3 Regime Parameterisation

The deployment regime determines the feasible solution space through the following parameter vector $\mathbf{p} = (C, \mathcal{N}, \mathcal{S}, \mathcal{T}, D, L, B, \sigma)$:

[Table 1: KFC Regime Parameterisation]

| Parameter | Symbol | Cloud Regime (Mooncake) | Edge Regime (SIG) |
|-----------|--------|------------------------|-------------------|
| Concurrency model | $C$ | Multi-tenant ($10^3$–$10^4$ requests) | Single-user (1 request) |
| Deployment scale | $\mathcal{N}$ | $10^2$–$10^3$ nodes | 1 device |
| KVCache sharing scope | $\mathcal{S}$ | Global (cross-node, cross-request) | Local (in-process, within-request) |
| Interconnect | $\mathcal{T}$ | RDMA/NVLink/CXL/TCP (100+ Gbps) | None (in-process memory) |
| Storage depth | $D$ | VRAM → DRAM → SSD (3 tiers) | VRAM → DRAM (2 tiers) |
| Latency orientation | $L$ | Throughput-oriented (TBT SLO) | Latency-oriented (wall-clock) |
| Edge-cloud bandwidth | $B$ | N/A (within cluster) | Variable (0–100 Mbps) |
| Scheduling complexity | $\sigma$ | Global (Conductor) | None (direct execution) |

**Weight assignment by regime:**

- **Cloud regime**: $\gamma \gg \alpha \gg \beta$ — storage cost dominates (KVCache pool is the binding constraint), prefill cost is secondary (amortised across many requests), transfer cost is minimised by RDMA.
- **Edge regime**: $\alpha \gg \gamma \gg \beta$ — prefill cost dominates (wall-clock time, Paper 1: 36% of total on 4B GPU), storage cost is secondary (single-session KVCache fits in VRAM for typical models), transfer cost is zero (in-process).

### 3.4 Architectural Decision Tree

Given regime parameters, the KFC framework produces architectural decisions through a deterministic mapping:

[Figure 1: KFC Architectural Decision Tree]

```
KFC Framework
├── Is KVCache sharing across requests required? (C > 1)
│   ├── YES (multi-tenant) → Global KVCache Pool (Mooncake path)
│   │   ├── Is network transfer available? (T ≠ ∅)
│   │   │   ├── YES → PD Disaggregation + Transfer Engine
│   │   │   └── NO → Co-located Prefill/Decode with Shared Memory
│   │   └── Is storage hierarchy deep? (D ≥ 3)
│   │       ├── YES → Multi-tier Storage (VRAM → DRAM → SSD)
│   │       └── NO → VRAM-only with Eviction
│   └── NO (single-user) → Local KVCache Preservation (SIG path)
│       ├── Are there tool-call boundaries?
│       │   ├── YES → Injection Continuity (SIG core)
│       │   └── NO → Standard Autoregressive Decoding
│       └── Is KVCache growth bounded?
│           ├── YES → No compression needed
│           └── NO → Periodic Compression (CompSIG)
├── Is prefix reuse across sessions beneficial? (P_shared / P_total ≥ threshold)
│   ├── YES (large shared prefix) → Prefix-Hash Matching
│   └── NO (small/unique prefix) → Skip prefix caching
└── Is edge-cloud handoff required? (B > 0 AND hybrid benefit exists)
    ├── YES → Hybrid Architecture (SIG + Mooncake)
    │   ├── KVCache Translation Layer
    │   └── Edge-Cloud Scheduling Policy
    └── NO → Standalone deployment
```

### 3.5 Subsumption Proof

**Theorem.** Mooncake and SIG are both optimal solutions to the KFC optimisation objective under their respective regime constraints.

*Proof (by construction).*

**Part A: Mooncake as optimal cloud solution.** Under cloud regime parameters $\mathbf{p}_{\text{cloud}} = (C=10^3, \mathcal{N}=10^2, \mathcal{S}=\text{global}, \mathcal{T}=\text{RDMA}, D=3, L=\text{throughput}, B=\text{N/A}, \sigma=\text{global})$:

1. *PD disaggregation is optimal.* With $C > 1$ and $\mathcal{T} \neq \emptyset$, the decision tree selects PD disaggregation. Separating prefill (compute-intensive) from decode (memory-intensive) eliminates interference, allowing each cluster to be independently scaled. The transfer cost $\beta \cdot C_{\text{transfer}}$ is minimised by RDMA (zero-copy, ~100 Gbps). Any co-located design would suffer from prefill-decode interference, increasing both TTFT and TBT beyond their SLO bounds.

2. *Global KVCache pool is optimal.* With $\mathcal{S} = \text{global}$ and $\mathcal{N} > 1$, a shared pool enables cross-request prefix reuse. The storage cost $\gamma \cdot C_{\text{storage}}$ is amortised across all requests sharing a prefix. Any per-node isolation would forfeit this amortisation, increasing total $C_{\text{prefill}}$.

3. *Multi-tier storage is optimal.* With $D = 3$ (VRAM → DRAM → SSD), expanding storage beyond VRAM increases cache hit rates. The "trading storage for computation" principle holds because $\gamma \cdot \Delta C_{\text{storage}} < \alpha \cdot \Delta C_{\text{prefill avoided}}$.

4. *Prefix-hash matching is optimal.* With $C > 1$ (multi-tenant), many requests share system prompts. Hash-based matching enables O(1) prefix lookup with storage overhead bounded by the number of unique prefixes.

5. *Conductor scheduling is optimal.* With $\sigma = \text{global}$ and $C > 1$, a global scheduler makes cache-aware placement decisions, with overhead amortised across thousands of requests.

**Part B: SIG as optimal edge solution.** Under edge regime parameters $\mathbf{p}_{\text{edge}} = (C=1, \mathcal{N}=1, \mathcal{S}=\text{local}, \mathcal{T}=\emptyset, D=2, L=\text{wall-clock}, B=0, \sigma=\text{none})$:

1. *Injection continuity is optimal.* With $C = 1$ and tool-call boundaries present, the decision tree selects injection continuity. For a single-user device with no network, the only way to minimise $C_{\text{prefill}}$ is to avoid recomputing KVCache entries that are already available. Injection continuity achieves this by preserving the KVCache across tool calls, reducing $C_{\text{prefill}}$ by 73–97% [3]. Any alternative that re-encodes the full prefix incurs $O(P_{\text{total}})$ prefill cost per step versus SIG's $O(I_k)$ incremental cost.

2. *No PD disaggregation.* With $\mathcal{T} = \emptyset$ (no network), PD disaggregation is infeasible—there is no interconnect for KVCache transfer.

3. *No global pool.* With $\mathcal{N} = 1$ and $\mathcal{S} = \text{local}$, there are no other nodes to share KVCache with.

4. *No scheduling.* With $C = 1$ and $\sigma = \text{none}$, there is only one request at a time. Scheduling overhead would increase latency without improving throughput.

5. *CompSIG as edge analog of multi-tier storage.* When KVCache growth exceeds VRAM ($D = 2$, DRAM overflow), CompSIG provides periodic compression (61% reduction, 17% overhead [2]). This is the edge analog of Mooncake's VRAM → DRAM → SSD tiering, adapted for the single-device constraint.

**Conclusion.** The KFC framework subsumes both systems as regime-specific instantiations of the same optimisation principle: minimise redundant KVCache computation subject to latency and storage constraints, where the deployment regime determines which constraints are binding and which cost terms dominate. $\blacksquare$

---

## 4. Convergence Analysis

### 4.1 Overview

We systematically compare Mooncake and SIG across eight architectural dimensions derived from the KFC framework's KVCache lifecycle. For each dimension, we identify: (a) the design choice in each system, (b) the regime constraint motivating it, (c) the convergence principle unifying them, and (d) divergence points where regime constraints produce fundamentally different solutions.

### 4.2 Dimension-by-Dimension Analysis

#### D1: KVCache Creation

| Aspect | Mooncake | SIG |
|--------|----------|-----|
| **Creation mechanism** | Dedicated prefill cluster computes KVCache for full input | Inject operation extends existing KVCache with incremental tokens |
| **Spatial organisation** | Spatially separated: prefill nodes → Transfer Engine → decode nodes | Temporally separated: suspend → inject → resume within same process |
| **Cost model** | $C_{\text{prefill}} = O(n_{\text{tokens}} \cdot d_{\text{model}}^2)$ per request, amortised across prefix-sharing requests | $C_{\text{prefill}} = O(I_k \cdot d_{\text{model}}^2)$ per step, where $I_k \ll P_{\text{total}}$ |

**Convergence point.** Both systems minimise redundant KVCache computation. Mooncake does so by sharing KVCache across requests (cross-request amortisation); SIG does so by preserving KVCache within a request (within-request continuity). Both instantiate the KFC principle: "trading storage for computation."

**Divergence point.** Mooncake's creation is a *batch* operation (full prefill of all input tokens), whilst SIG's creation is an *incremental* operation (inject only new tokens). This reflects the fundamental difference between cross-request reuse (where the full prefix must be computed once) and within-request continuity (where the prefix is already in cache).

#### D2: Computation-Storage Tradeoff

| Aspect | Mooncake | SIG |
|--------|----------|-----|
| **Tradeoff direction** | Expand storage (VRAM → DRAM → SSD) to increase cache hit rate, reducing GPU re-computation | Preserve storage (keep KVCache in VRAM/DRAM) to avoid re-computation entirely |
| **Computation avoided** | Prefix hit rate × prefill cost per request | 73–97% of prefill tokens [3] |
| **Quantified benefit** | Cache hit rate increases from ~50% (VRAM-only) to much higher with multi-tier | 2.38–9.45× speedup [3], 55× prefill reduction [4] |

**Convergence point.** Both systems embody "trading storage for computation" (以存换算). Mooncake trades cluster-scale storage (DRAM + SSD across nodes) for reduced GPU computation. SIG trades device-scale storage (VRAM + DRAM on one device) for eliminated re-prefill. The tradeoff coefficient $\lambda = \Delta \text{Storage} / \Delta \text{Computation avoided}$ is favourable in both regimes.

**Divergence point.** The storage medium differs fundamentally. Mooncake's multi-tier storage spans VRAM → DRAM → SSD across a cluster, enabling terabyte-scale KVCache pools. SIG's storage is limited to a single device's VRAM + DRAM (typically 12–24 GB for edge GPUs). This difference in scale (TB vs. GB) determines the feasible eviction policies: Mooncake can afford to retain cold KVCache on SSD; SIG must compress or evict when VRAM fills (CompSIG).

#### D3: Scheduling Philosophy

| Aspect | Mooncake | SIG |
|--------|----------|-----|
| **Scheduler** | Conductor: global, cache-aware, prediction-based | None: direct execution |
| **Scheduling granularity** | Request-level: which prefill node, which decode node, cache-aware placement | Step-level: automatic (inject → resume) |
| **Optimisation target** | Maximise throughput subject to SLO (TTFT, TBT) | Minimise wall-clock latency |

**Convergence point.** Both systems make scheduling decisions that are KVCache-centric. Mooncake's Conductor places requests to maximise prefix reuse (cache-aware scheduling). SIG's injection implicitly schedules computation to minimise re-encoding (cache-preserving scheduling). Both treat the KVCache state as the primary scheduling signal.

**Divergence point.** Mooncake requires explicit, global scheduling because of multi-tenant resource contention. SIG requires no scheduling because there is no contention. In the hybrid architecture, edge-cloud scheduling would be needed—this is a key design challenge (§6).

#### D4: Cache Sharing Scope

| Aspect | Mooncake | SIG |
|--------|----------|-----|
| **Sharing scope** | Cross-request, cross-node (global pool) | Within-request, within-process (local cache) |
| **Sharing mechanism** | Prefix-hash matching: hash(system_prompt) → KVCache blocks | Injection continuity: preserved KVCache across tool calls |
| **Effective when** | Multiple requests share the same prefix (system prompts, tool descriptions) | A single request has multiple tool calls (deep chains) |

**Convergence point.** Both systems reuse KVCache to avoid re-computation. The reuse mechanism differs in scope (cross-request vs. within-request) but shares the same principle: cached KVCache entries should be reused rather than recomputed.

**Divergence point.** The sharing scope is fundamentally determined by the concurrency model ($C$). In the multi-tenant cloud ($C > 1$), cross-request sharing is possible and beneficial. In the single-user edge ($C = 1$), within-request continuity is the only available reuse mechanism. Paper 4's R6 data confirms that these operate on orthogonal axes: prefix caching provides less than 3% benefit in the deep-chain regime where SIG provides 2.79–4.26× speedup [4].

#### D5: Transport Mechanism

| Aspect | Mooncake | SIG |
|--------|----------|-----|
| **Transport** | Transfer Engine: zero-copy RDMA, GPUDirect, NVLink, CXL, TCP | In-process memory: no transport needed |
| **Bandwidth** | 8×400 Gbps (multi-NIC aggregation) | N/A (memory bandwidth only) |
| **Latency** | ~10–100 μs (RDMA within rack), ~1–10 ms (TCP across racks) | ~0 (in-process) |

**Convergence point.** Both systems treat KVCache transfer as a first-class operation. Mooncake's Transfer Engine and SIG's injection both move KVCache from one state to another (prefill node → decode node, or pre-injection → post-injection). The key insight is that KVCache is a *portable object*, not a transient byproduct.

**Divergence point.** Mooncake's transport is network-based (inter-node), whilst SIG's transport is memory-based (intra-process). This is the most fundamental divergence: it determines whether PD disaggregation is feasible (requires network) or whether injection continuity is the only option (no network). The hybrid architecture must bridge this gap with an edge-cloud transport layer.

#### D6: Deployment Scale

| Aspect | Mooncake | SIG |
|--------|----------|-----|
| **Scale** | Cluster: $10^2$–$10^3$ nodes, $10^2$–$10^3$ GPUs | Device: 1 node, 1 GPU |
| **Model size** | 70B+ (e.g., Kimi K2 on 128 H200 GPUs) | 0.5B–4B (edge-deployable) |
| **KVCache size** | TB-scale (global pool across cluster) | MB-scale (single device VRAM) |

**Convergence point.** Both systems face the same fundamental constraint: KVCache memory is the binding resource, not FLOPs. At cloud scale, this manifests as GPU memory fragmentation across thousands of requests. At edge scale, this manifests as VRAM capacity limits for a single session.

**Divergence point.** The scale difference (3+ orders of magnitude) produces qualitatively different design choices. Cloud-scale requires distributed coordination (Conductor, global pool), whilst edge-scale requires local optimisation (injection, compression).

#### D7: Optimisation Objective

| Aspect | Mooncake | SIG |
|--------|----------|-----|
| **Primary objective** | Maximise throughput subject to SLO | Minimise wall-clock latency |
| **KFC weight mapping** | $\gamma \gg \alpha \gg \beta$ (storage-dominant) | $\alpha \gg \gamma \gg \beta$ (prefill-dominant) |
| **SLO** | TTFT ≤ 5s, TBT ≤ 50ms (typical cloud SLOs) | No SLO; minimise total time |

**Convergence point.** Both objectives are special cases of the KFC framework's general objective. The KFC objective $\min \Phi = \alpha \cdot C_{\text{prefill}} + \beta \cdot C_{\text{transfer}} + \gamma \cdot C_{\text{storage}}$ reduces to throughput maximisation when $\gamma$ dominates (cloud) and to latency minimisation when $\alpha$ dominates (edge).

**Divergence point.** The optimisation objectives are inversely related in the general case: maximising throughput may increase per-request latency (batching more requests), and minimising latency may reduce throughput (serving fewer requests). This tension is resolved by the deployment regime: cloud systems optimise for the many (throughput), edge systems optimise for the one (latency).

#### D8: Failure Handling

| Aspect | Mooncake | SIG |
|--------|----------|-----|
| **Overload** | Prediction-based early rejection; 525% throughput increase | N/A (single user) |
| **Node failure** | KVCache replication (hot replicas); failover to backup decode node | N/A (single device) |
| **Cache overflow** | Evict to DRAM/SSD; recompute on cache miss | CompSIG compression (61% reduction [2]) |

**Convergence point.** Both systems need mechanisms to handle KVCache unavailability. Mooncake uses replication and multi-tier fallback; SIG uses compression and re-encoding fallback. Both treat KVCache loss as a performance degradation (not a correctness failure)—the system can always recompute from source tokens.

**Divergence point.** Mooncake's failure handling is distributed (replication, failover) whilst SIG's is local (compression, re-encoding). The hybrid architecture would need to handle failures at the edge-cloud boundary (network partition, translation errors).

### 4.3 Convergence Summary

[Table 2: Convergence Strength Summary]

| Dimension | Convergence Strength | Convergence Principle | Key Divergence |
|-----------|---------------------|----------------------|----------------|
| D1: KVCache creation | **Strong** | Minimise redundant computation | Batch vs. incremental creation |
| D2: Computation-storage tradeoff | **Strong** | Trading storage for computation | TB-scale vs. GB-scale storage |
| D3: Scheduling philosophy | **Moderate** | KVCache-centric scheduling | Global vs. no scheduling |
| D4: Cache sharing scope | **Moderate** | Reuse cached KVCache | Cross-request vs. within-request |
| D5: Transport mechanism | **Weak** | KVCache as portable object | Network vs. in-process |
| D6: Deployment scale | **Weak** | Memory is the binding constraint | Cluster vs. device |
| D7: Optimisation objective | **Strong** | KFC objective special cases | Throughput vs. latency |
| D8: Failure handling | **Moderate** | Graceful degradation | Distributed vs. local |

Five of eight dimensions show strong or moderate convergence, confirming that Mooncake and SIG share a common design philosophy. The three weak-convergence dimensions (D5, D6, D8) are precisely those where regime constraints (network availability, scale, fault tolerance) produce fundamentally different solutions—confirming the KFC framework's prediction that regime parameters determine architectural choices.

---

## 5. Transferability Analysis

### 5.1 Methodology

We assess four Mooncake design patterns for transferability to SIG's edge deployment along two axes: **technical feasibility** ($f \in [0, 1]$) and **expected benefit** ($b \in [0, 1]$). The transferability score is $\tau = f \times b$. We distinguish between soft constraints (engineering effort, can be overcome with development) and hard constraints (fundamental incompatibilities, cannot be overcome).

### 5.2 Transferability Matrix

[Table 3: Transferability Matrix]

| Mooncake Pattern | Technical Feasibility ($f$) | Expected Benefit ($b$) | Transferability ($\tau$) | Constraint Type |
|-----------------|---------------------------|----------------------|------------------------|----------------|
| **P1: Prefix-hash matching** | 0.7 | 0.4 | **0.28** | Soft (implementation) + Hard (deep-chain) |
| **P2: Hot/cold KVCache scheduling** | 0.3 | 0.5 | **0.15** | Hard (no multi-tenant contention) |
| **P3: Multi-tier storage** | 0.6 | 0.6 | **0.36** | Soft (CompSIG adaptation) |
| **P4: PD disaggregation** | 0.1 | 0.3 | **0.03** | Hard (no network interconnect) |

### 5.3 Pattern-by-Pattern Analysis

**P1: Prefix-Hash Matching → Edge-Adapted Prefix Caching.** Mooncake hashes the system prompt and shared context to create a lookup key; when a new request arrives with a matching prefix, the pre-computed KVCache blocks are reused. For edge adaptation, this requires: (a) implementing a persistent KVCache store on disk (beyond llama.cpp's current in-memory-only model), (b) adding hash-based lookup for system prompt and tool description prefixes, (c) implementing KVCache block serialisation/deserialisation, and (d) handling position encoding continuity when restoring cached blocks. These are engineering tasks, not fundamental impossibilities.

The expected benefit is scenario-dependent. Paper 4's R6 data shows prefix caching provides less than 3% benefit when $P_s + P_a \approx 80$ tokens and total context exceeds 2,500 tokens. However, for multi-agent scenarios with extensive shared tool descriptions ($P_s + P_a \geq 500$ tokens), the benefit becomes material. For $N = 5$ agents sharing a 2,000-token prefix, prefix caching saves $(N-1) \times 2,000 = 8,000$ tokens of first-turn prefill—an 80% reduction in first-turn cost.

**Hard constraint.** Prefix caching provides negligible benefit in the deep-chain regime where per-step incremental tokens dominate (Paper 4). The hypothesis applies specifically to first-turn prefill and cross-session reuse, not to within-request chain continuity.

**P2: Hot/Cold KVCache Scheduling → Edge-Adapted Cache Priority.** Mooncake's Conductor classifies KVCache blocks as "hot" (frequently accessed, replicated across nodes) or "cold" (rarely accessed, stored on SSD, evicted when space is needed). The fundamental challenge for edge transfer is that SIG's single-user, single-request model has no contention—there is no "scheduling" in the traditional sense because there is only one active request. The benefit comes not from scheduling (choosing which request to serve) but from cache management (choosing which KVCache segments to retain). CompSIG already provides a partial solution (periodic compression); the remaining gap is importance-based selective retention, which requires attention-score tracking not currently available in llama.cpp.

**P3: Multi-Tier Storage → CompSIG + DRAM Offloading.** This pattern maps most directly to SIG's existing CompSIG mechanism, extended with DRAM offloading. Tier 1 (VRAM) holds active KVCache; Tier 2 (DRAM) holds compressed KVCache (CompSIG-processed segments). For a 4B model on RTX 4070 SUPER (12 GB VRAM), with approximately 4 GB available for KVCache after model weights, the maximum KVCache in VRAM is approximately 8,000 tokens (FP16). With CompSIG + DRAM offloading, approximately 8,000 tokens remain in VRAM plus approximately 20,000 tokens compressed in DRAM (8 GB DRAM budget), yielding an effective context of approximately 28,000 tokens—a 3.5× extension. The main engineering challenge is implementing efficient VRAM ↔ DRAM transfer without blocking generation.

**P4: PD Disaggregation → Not Transferable.** PD disaggregation requires network interconnect between prefill and decode nodes, multiple GPU nodes, and global scheduling—none of which are available on edge devices. The only viable adaptation is the hybrid architecture: use the cloud as the "prefill node" and the edge as the "decode node," with KVCache migration between them (§6). This is not a direct transfer of PD disaggregation but a novel composition enabled by the edge-cloud boundary.

### 5.4 Reverse Transferability: SIG → Mooncake

[Table 4: Reverse Transferability]

| SIG Pattern | Mooncake Adaptation | Feasibility | Expected Benefit |
|-------------|--------------------|----|-----------------|
| **Injection continuity** | Long-running agent sessions in cloud: suspend session, store KVCache in global pool, resume on different decode node with injection of new tool results | High (Mooncake already supports KVCache transfer between decode nodes) | Reduced re-encoding for agent workloads; principled framework for handling tool-call boundaries |
| **CompSIG compression** | KVCache compression for cold blocks in SSD tier: compress using importance-based pruning instead of storing full-precision | Medium (requires integration with PagedAttention block format) | 2–3× SSD storage efficiency; more KVCache retained per GB of SSD |
| **Stabilisation templates** | Template-based injection for structured tool results in cloud agent workloads | High (no architectural change needed) | Reduced distribution shift when injecting tool results; improved generation quality after injection |

**Key insight.** The most impactful reverse transfer is SIG's injection continuity model for cloud agent workloads. As LLM agents become more prevalent in cloud serving (AutoGPT, Devin-style coding agents), Mooncake will need to handle long-running sessions with many tool calls—exactly the regime where SIG excels. SIG's formalisation of the suspend-inject-resume cycle provides a principled framework that Mooncake's current KVCache transfer mechanism lacks. For a 100-step agent session on a 70B model, SIG-inspired injection + CompSIG compression could enable serving 2.5× more concurrent long-running agent sessions per decode node by keeping each session's active KVCache within a smaller VRAM footprint.

---

## 6. Hybrid Architecture Feasibility Analysis: When Does Prefix Caching Add Value on Top of SIG?

### 6.1 Component Model

The hybrid SIG-Mooncake architecture comprises three components:

[Figure 2: Hybrid Architecture Component Diagram]

[Table 5: Hybrid Architecture Components]

| Component | Responsibilities | Technology Stack |
|-----------|-----------------|------------------|
| **Edge SIG Node** | Local inference with injection continuity, local KVCache store, edge scheduler, prefix cache client | llama.cpp + SIG extensions, local SSD/DRAM for KVCache persistence |
| **Cloud Mooncake Cluster** | Global KVCache pool, PD-disaggregated serving, Transfer Engine, Conductor scheduler | vLLM/SGLang + Mooncake Store + Transfer Engine |
| **Edge-Cloud Gateway** | KVCache translation, migration scheduling, protocol adaptation, privacy filtering | Translation layer (new), Mooncake Transfer Engine client, policy engine |

### 6.2 Interface Contracts

**Interface 1: Edge → Cloud (KVCache_Migrate).**

```
KVCache_Migrate(
    session_id: str,
    kv_blocks: List[KVBlock],
    metadata: KVCacheMetadata  // model_id, quantization, position_range, layer_count
) → MigrationHandle
```

*When:* Session handoff (edge → cloud), KVCache overflow (edge VRAM full), cloud prefill request.
*Feasibility condition:* Migration must complete before the next inference step is needed.

**Interface 2: Cloud → Edge (KVCache_Prefill).**

```
KVCache_Prefill(
    session_id: str,
    input_tokens: List[int],
    model_id: str,
    quantization: str
) → List[KVBlock]
```

*When:* Edge cold-start with cloud prefill, cloud pre-computation of shared prefix.
*Feasibility condition:* Cloud prefill + transfer must be faster than local edge prefill.

**Interface 3: Cloud → Cloud (existing Mooncake Transfer Engine).** RDMA-based zero-copy transfer—no new design needed.

### 6.3 KVCache Format Translation

The primary technical challenge for the hybrid architecture is KVCache format compatibility between llama.cpp (SIG's runtime) and vLLM/SGLang (Mooncake's backends).

[Table 6: KVCache Format Comparison]

| Dimension | llama.cpp | vLLM/PagedAttention |
|-----------|-----------|---------------------|
| **Block structure** | Contiguous per layer | Fixed-size blocks (16 tokens) |
| **Attention head layout** | [n_kv_heads × head_dim × max_seq_len] | [block_size × n_kv_heads × head_dim] |
| **Quantization** | FP16 (KVCache activations, even for Q4_K_M model weights) | FP16 (default) or FP8 (optional) |
| **Positional encoding** | RoPE applied during attention; positions tracked by token index | RoPE applied during attention; positions tracked in block metadata |
| **GQA group mapping** | Implicit in model weights | Explicit in block table |

[Figure 3: KVCache Format Translation Pipeline]

The translation proceeds in five steps:

1. **Block alignment.** Chunk the contiguous array into 16-token blocks; pad the last block. Overhead: ≤1 block per layer (≤2.8 MB for a 4B model with 36 layers).

2. **Head mapping.** Reshape from [n_kv_heads × head_dim × seq_len] to [n_blocks × block_size × n_kv_heads × head_dim]. Information loss: zero—this is a memory layout transformation. Overhead: O(n_layers × n_kv_heads × seq_len × head_dim) memory copies.

3. **Quantization reconciliation.** FP16 → FP16: zero overhead. FP16 → FP8: 2× storage reduction, less than 1% perplexity increase. Q4 → FP16 (if future llama.cpp supports quantised KVCache): 4× storage increase.

4. **Positional encoding.** Record the position range [start_pos, end_pos] for each block. The receiving system must not re-apply RoPE to the transferred KVCache. Overhead: metadata only.

5. **GQA mapping.** Both systems store only the KV heads (not the full attention heads). No translation needed.

[Table 7: Translation Overhead Summary]

| Step | Operation | Time (1024 tokens, 4B model, GPU) | Time (1024 tokens, 4B model, CPU) | Storage Overhead | Information Loss |
|------|-----------|-----------------------------------|-----------------------------------|-----------------|-----------------|
| 1. Block alignment | Chunk + pad | <0.1 ms | <0.1 ms | ≤2.8 MB | None |
| 2. Head mapping | Reshape + copy | 2.71 ms | 49.63 ms | None | None |
| 3. Quantization | FP16→FP16 (copy) | <0.01 ms | 18.45 ms | 0 | 0% |
| | FP16→FP8 (quantize) | 3.85 ms | 263.44 ms | −50% | <1% |
| 4. Position encoding | Metadata update | <0.1 ms | <0.1 ms | Negligible | None |
| 5. GQA mapping | No-op | 0 | 0 | None | None |
| **Total (FP16→FP16)** | | **2.71 ms** | **68.08 ms** | **≤2.8 MB** | **0%** |
| **Total (FP16→FP8)** | | **6.56 ms** | **313.08 ms** | **≤1.4 MB** | **<1%** |

*Measured on NVIDIA RTX 4070 SUPER, Qwen3.5-4B-Q4_K_M (32 layers, GQA with n_kv_heads=4, head_dim=160). GPU measurements use PyTorch CUDA with torch.cuda.synchronize() for accurate timing. CPU measurements use numpy for comparison.*

**Feasibility verdict.** KVCache format translation is feasible for session-level migration on GPU hardware: ~2.7 ms for FP16→FP16 and ~6.6 ms for FP16→FP8 at 1024 tokens, confirming the original ~1–7 ms estimate for the FP16→FP16 path. However, two important qualifications apply: (a) the ~1–7 ms range holds for GPU execution only—CPU-based translation is 10–50× slower due to sequential memory operations versus parallel CUDA kernels; and (b) the overhead scales with token count, reaching ~28 ms at 4096 tokens (FP16→FP16) on GPU. For per-step edge-cloud synchronisation, even GPU-based translation is too expensive; the hybrid architecture must minimise the frequency of translation events.

### 6.4 Scheduling Policy

[Table 8: Hybrid Architecture Scheduling Policy]

| Condition | Decision | Rationale |
|-----------|----------|-----------|
| Cold-start AND $B_{\text{edge-cloud}} > B_{\text{break-even}}$ AND prefix available in cloud | Fetch prefix from cloud | $T_{\text{prefix\_fetch}} < T_{\text{cold}}$ |
| Cold-start AND ($B_{\text{edge-cloud}} < B_{\text{break-even}}$ OR prefix not in cloud) | Local prefill | Cloud fetch is slower than local computation |
| Warm-start (local prefix cache available) | Use local prefix cache | $T_{\text{restore}} \ll T_{\text{prefix\_fetch}}$ |
| Deep chain ($K > 10$) AND privacy allows cloud | Consider cloud delegation | Cloud per-step latency may be lower |
| Deep chain AND privacy constrains to edge | SIG injection locally | Only feasible option |
| Network partition | Fall back to pure-edge SIG | Graceful degradation |

### 6.5 Performance Model

For a hybrid inference session initiated at the edge with optional cloud delegation:

$$T_{\text{hybrid}} = T_{\text{prefix\_cache}} + K \cdot T_{\text{SIG\_inject}} + \mathbb{1}_{\text{cloud}} \cdot T_{\text{transfer}}$$

Where:
- $T_{\text{prefix\_cache}}$ = time to obtain the shared prefix KVCache (local prefill, cloud fetch, or local cache restore)
- $K$ = chain depth (number of tool calls)
- $T_{\text{SIG\_inject}}$ = per-step SIG injection time
- $\mathbb{1}_{\text{cloud}}$ = indicator for cloud involvement
- $T_{\text{transfer}}$ = KVCache migration time (edge ↔ cloud), including format translation

**Component breakdown for a 4B model (Qwen3.5-4B, GQA) on RTX 4070 SUPER:**

| Component | Formula | Typical Value |
|-----------|---------|--------|
| $T_{\text{prefix\_cache}}$ (local cold-start) | $(P_s + P_d + P_r) \times t_{\text{prefill\_per\_token}}$ | 5000 tokens × 0.158 ms/token = 0.79 s |
| $T_{\text{prefix\_cache}}$ (cloud fetch, 3 Gbps, 4× compressed) | $S_{\text{KVCache}} / B_{\text{edge-cloud}} + T_{\text{translate}}$ | ~0.27 s (GQA: 80 KB/token) |
| $T_{\text{prefix\_cache}}$ (local prefix cache hit) | $T_{\text{restore}}$ (memcpy from DRAM) | ~1 ms |
| $T_{\text{SIG\_inject}}$ | $I_k \times t_{\text{prefill\_per\_token}}$ | 80 tokens × 0.158 ms/token = 12.6 ms |

### 6.6 Break-Even Analysis

**Hybrid vs. pure-edge cold-start.** The hybrid outperforms pure-edge when:

$$\frac{S_{\text{KVCache}}(P)}{B} + T_{\text{translate}} < P \times t_{\text{prefill}}$$

Solving for the break-even bandwidth:

$$B > \frac{S_{\text{KVCache}}(P)}{P \times t_{\text{prefill}} - T_{\text{translate}}}$$

For large $P$ (where $T_{\text{translate}} \ll P \times t_{\text{prefill}}$), this simplifies to a constant:

$$B > \frac{s_{\text{per\_token}}}{t_{\text{prefill}}}$$

**GQA correction.** The original analysis assumed a non-GQA architecture where $n_{\text{kv\_heads}} = n_{\text{heads}}$, yielding KVCache per token ≈ 360 KB for a 4B model. Empirical measurement on Qwen3.5-4B (which uses GQA with $n_{\text{kv\_heads}} = 4$ vs. $n_{\text{heads}} = 16$) reveals KVCache per token = 80 KB—4× smaller than the non-GQA assumption. This has significant implications for the break-even bandwidth:

[Table 7b: Break-Even Bandwidth by Architecture]

| Architecture | KVCache/token | $t_{\text{prefill}}$/token | Break-even (raw) | Break-even (4× compressed) |
|-------------|-------------|---------------------------|-----------------|---------------------------|
| Non-GQA ($n_{\text{kv}} = n_{\text{heads}} = 16$) | 320 KB | 0.158 ms | 16.2 Gbps | 4.1 Gbps |
| GQA ($n_{\text{kv}} = 4$, Qwen3.5-4B actual) | 80 KB | 0.158 ms | 4.1 Gbps | 1.0 Gbps |
| Original estimate (§6.6 draft) | 360 KB | 0.8 ms | 3.6 Gbps | 0.9 Gbps |

*Measured on NVIDIA RTX 4070 SUPER, Qwen3.5-4B-Q4_K_M. Prefill per-token time measured at steady-state (256–4096 tokens): 0.142–0.158 ms/token (6336–7029 tok/s).*

**Key finding.** The original 3.6 Gbps estimate was close to the GQA-corrected value (4.1 Gbps) despite using different assumptions. The original analysis used a higher prefill per-token time (0.8 ms/token) that reflected cold-start conditions with small batch sizes, whilst our steady-state measurement yields 0.158 ms/token. The near-agreement arises because the original estimate's higher numerator (360 KB vs. 80 KB) was offset by its higher denominator (0.8 ms vs. 0.158 ms). For GQA models—the dominant architecture for modern edge-deployable LLMs—the break-even bandwidth is approximately **4.1 Gbps (raw) / 1.0 Gbps (4× compressed)**, achievable on Gigabit Ethernet with compression.

**Cold-start scenario with GQA-corrected values.** For a 5000-token prefix with GQA (80 KB/token), the total KVCache is 390.6 MB (vs. 1562.5 MB for non-GQA). At 1 Gbps with 4× compression, cloud fetch takes ~0.82 s vs. local prefill of ~0.79 s—near the break-even point. At 3 Gbps with compression, cloud fetch takes ~0.27 s, providing 3× speedup over local prefill. This confirms the hybrid architecture's viability at Ethernet-grade bandwidth with compression.

The hybrid architecture's cold-start benefit depends critically on: (a) KVCache compression—without compression, the transfer time exceeds local prefill time for large prefixes; (b) network bandwidth—at 3+ Gbps with compression, hybrid cold-start becomes clearly advantageous; and (c) pre-loading—the strongest benefit comes from pre-loading known prefixes during idle time, reducing cold-start to near-zero.

### 6.8 KFC Framework Generative Validation

The KFC framework's value depends not only on describing existing systems but on predicting architectures for unexplored regime combinations. We test this by applying the decision tree (§3.4) to a scenario not covered by either Mooncake or SIG: the *connected edge* ($C = 1, B > 0$)—a single-user device with network connectivity.

**KFC decision tree output for connected edge:**

```
C = 1 → Local KVCache Preservation (SIG path)
B > 0 AND prefix available → Cloud Prefix Prefetch (new component)
Tool-call boundaries → Injection Continuity (SIG)
Privacy filter → Upload only Ps+Pa, retain Pr+Ik locally
```

The framework predicts a novel hybrid mode: **SIG + Cloud Prefix Prefetch**—neither PD disaggregation (which requires multi-node) nor pure SIG (which assumes no network). This mode operates as follows:

1. **First cold-start:** Fetch the shared prefix KVCache ($P_s + P_a$) from the cloud, avoiding local prefill of the full prefix.
2. **Subsequent sessions:** Restore prefix from local PrefixCache (SIG + PC, as validated in §7.6).
3. **Network disconnection:** Gracefully degrade to pure SIG (local prefill + injection continuity).

This prediction is not a trivial interpolation between Mooncake and SIG—it identifies a qualitatively new architectural component (Cloud Prefix Prefetch) that neither system possesses. The framework's ability to generate this prediction from regime parameters alone demonstrates its generative capacity beyond post-hoc description.

**Privacy filtering analysis.** The connected-edge scenario raises a critical privacy question: what KVCache can be safely shared with the cloud? We first present a binary analysis, then refine it with entropy-based grading.

*Binary analysis.* For a typical agent workload on Qwen3.5-4B:

| KVCache Segment | Tokens | Size (GQA) | Upload to Cloud? |
|----------------|--------|-----------|-----------------|
| Shared prefix ($P_s + P_a$) | 60 | 4.8 MB | Yes (non-sensitive) |
| Per-request ($P_r$) + Injection ($I_k$) | 125 | 10.0 MB | No (may contain user data) |
| **Total** | 185 | 14.8 MB | **67.6% retained locally** |

*Entropy-based grading.* The binary $P_s + P_a$ / $P_r + I_k$ dichotomy treats all shared-prefix content as equally safe and all private content as equally sensitive. We refine this with Shannon entropy as a quantitative privacy proxy, measuring six content categories:

| Content Category | Shannon Entropy (bits/char) | Privacy Level | Uploadable |
|-----------------|---------------------------|---------------|------------|
| System prompt | 3.81 | L0 (public) | Yes |
| Tool descriptions | 4.38 | L1 (internal) | Auth only |
| User query | 4.33 | L1 (internal) | Auth only |
| Tool result (weather) | 4.51 | L2 (sensitive) | No |
| Tool result (flights) | 4.83 | L2 (sensitive) | No |
| Code snippet | 3.64 | L0 (public) | Yes |

*Thresholds: L0 (H < 4.03), L1 (4.03 ≤ H < 4.43), L2 (H ≥ 4.43). Measured on Qwen3.5-4B KVCache (80 KB/token).*

The three-tier grading retains **65.9% locally**—closely matching the binary estimate (67.6%)—but provides finer-grained control: system prompts and code snippets (L0, deterministic content with low entropy) are safely uploadable; tool descriptions and user queries (L1, moderate entropy) require authorisation; tool results containing personal or financial data (L2, high entropy) are never uploaded.

**Limitations of entropy-based grading.** We emphasise that the entropy-based privacy grading is a *heuristic engineering guideline*, not a security guarantee. Three important limitations apply:

1. **KVCache reconstruction risk.** Recent work has demonstrated that KVCache activations can be used to partially reconstruct the original text, even without access to the model weights. Shannon entropy of the source text does not capture this risk—low-entropy code snippets may be more easily reconstructed than high-entropy natural language, yet our grading would classify the code snippet as L0 (uploadable). A formal threat model accounting for adversarial inference from KVCache activations is needed before deployment.

2. **Entropy-sensitivity mismatch.** The correlation between text entropy and privacy sensitivity is imperfect. A short, low-entropy string containing a password or API key (e.g., `sk-abc123...`) would be classified as L0 despite being highly sensitive. Conversely, a high-entropy natural language passage about public weather data would be classified as L2 despite being non-sensitive.

3. **Defence-in-depth recommendation.** The entropy grading should be one layer in a multi-layer defence strategy, not the sole privacy mechanism. Complementary defences include: (a) end-to-end encryption of KVCache during transfer, (b) secure enclave processing for sensitive tokens on the edge device, (c) differential privacy noise injection before cloud upload, and (d) access control policies that restrict which cloud nodes can receive KVCache. We recommend the following revised deployment guidance: L0 content is uploadable with encryption in transit; L1 content requires both encryption and access control; L2 content must remain on-device regardless of entropy score. Future work should develop a formal security model that quantifies the actual information leakage risk from KVCache transfer under adversarial assumptions.

1. **Position encoding compatibility.** llama.cpp stores pre-RoPE K/V values in the KVCache; RoPE is applied dynamically during attention computation. Therefore, the shared prefix's KVCache can be transferred to the cloud and re-used without position encoding conflicts—the receiving system applies RoPE at its own position indices.

2. **Semantic integrity.** The shared prefix contains only system prompts and tool descriptions—non-sensitive, deterministic content that is identical across sessions and users. Stripping $P_r$ and $I_k$ (which may contain user queries and tool results with sensitive data) ensures privacy.

3. **Position continuity.** After restoring the shared prefix at positions $[0, |P_s + P_a|)$, local injection continues at position $|P_s + P_a|$ and beyond. The position offset is a simple integer parameter, requiring no KVCache modification.

**Hybrid vs. pure-cloud.** For deep chains ($K \gg 1$), the hybrid outperforms pure-cloud when $T_{\text{inject}} < T_{\text{RTT}} + T_{\text{cloud\_step}}$. For $T_{\text{inject}} = 64$ ms (4B edge), $T_{\text{RTT}} = 50$ ms, $T_{\text{cloud\_step}} = 10$ ms: $64 < 60$—the hybrid does not outperform pure-cloud for this scenario. However, the hybrid has advantages that pure-cloud lacks: (a) privacy—sensitive data stays on-device, (b) offline capability—SIG continues locally if the network is lost, (c) latency tail—no network jitter, and (d) cost—no cloud compute charges for per-step injection. When privacy constraints require edge-only processing of $P_r$ and $I_k$, the hybrid is the only architecture that combines cloud prefix caching with edge injection.

### 6.7 Cold-Start Scenario Analysis

[Table 9: Cold-Start Scenario Comparison]

| Scenario | Total Time | vs. Pure-Edge |
|----------|-----------|-------------|
| Pure-edge SIG (cold-start) | 1.05 s | Baseline |
| Hybrid (raw KVCache transfer, 1 Gbps) | 3.28 s | −212% (slower) |
| Hybrid (compressed transfer, 1 Gbps) | 0.82 s | +22% |
| Hybrid (compressed transfer, 3 Gbps) | 0.27 s | +74% |
| Hybrid (pre-loaded prefix) | 0.02 s | +98% |

*Setup: 5000-token prefix, 20-step chain, Qwen3.5-4B (GQA) on RTX 4070 SUPER. Local prefill: 0.158 ms/token. KVCache: 80 KB/token (GQA). SIG inject: 12.6 ms/step.*

The hybrid architecture's cold-start benefit depends critically on: (a) KVCache compression—without compression, the transfer time exceeds local prefill time for large prefixes; (b) network bandwidth—at 3+ Gbps with compression, hybrid cold-start becomes clearly advantageous; and (c) pre-loading—the strongest benefit comes from pre-loading known prefixes during idle time, reducing cold-start to near-zero.

### 6.7b End-to-End Migration PoC

We implement a proof-of-concept end-to-end KVCache migration pipeline on Qwen3.5-4B (GQA, n_kv_heads=4, head_dim=160, 80 KB/token) on RTX 4070 SUPER, measuring the full five-step chain: (1) KVCache extract from llama.cpp, (2) format translation to PagedAttention blocks, (3) simulated network transfer at 3 Gbps, (4) reverse translation, (5) restore. The extract step uses CPU numpy allocation as a proxy for the actual CUDA D2H transfer cost.

[Table 9b: End-to-End Migration Latency (4B, 3 Gbps)]

| Tokens | Extract (ms) | Translate (ms) | Transfer (ms) | Translate₂ (ms) | Restore (ms) | Total (ms) | Local Prefill (ms) | Verdict |
|--------|-------------|----------------|---------------|-----------------|-------------|------------|-------------------|---------|
| 128 | 63.6 | 5.0 | 28.0 | 3.5 | 2.0 | 102.1 | 116.7 | **HYBRID** |
| 256 | 134.2 | 7.0 | 55.9 | 7.0 | 4.0 | 208.2 | 15.6 | EDGE |
| 512 | 270.4 | 14.2 | 111.9 | 14.1 | 8.1 | 418.7 | 27.2 | EDGE |
| 1024 | 539.8 | 33.6 | 223.7 | 32.5 | 18.5 | 848.1 | 113.6 | EDGE |
| 2048 | 1088.4 | 249.9 | 447.4 | 63.2 | 27.5 | 1876.3 | 259.5 | EDGE |
| 4096 | 2166.9 | 1125.1 | 894.8 | 131.3 | 52.2 | 4370.3 | 586.7 | EDGE |

**Key finding (CPU-baseline PoC).** At 128 tokens, the hybrid path (102 ms) is marginally faster than local prefill (117 ms)—the only scale where hybrid wins. For 256+ tokens, the extract step dominates: CPU-numpy allocation of the KVCache tensor array costs 134–2167 ms, far exceeding the network transfer time (29–895 ms). This confirms that the bottleneck is not network bandwidth but the KVCache serialisation path.

**Multi-bandwidth break-even, CPU-baseline PoC (1024 tokens, 4B):**

| BW (Gbps) | Transfer (ms) | Total Hybrid (ms) | Local Prefill (ms) | Verdict |
|-----------|--------------|-------------------|-------------------|---------|
| 0.5 | 1342.2 | 1967.7 | 113.6 | EDGE |
| 1.0 | 671.1 | 1296.6 | 113.6 | EDGE |
| 3.0 | 223.7 | 849.3 | 113.6 | EDGE |
| 10.0 | 67.1 | 692.7 | 113.6 | EDGE |

Even at 10 Gbps, the CPU-baseline hybrid total (693 ms) exceeds local prefill (114 ms) due to the extract+translate overhead.

#### 6.7b-revised: CUDA-Optimised End-to-End Migration PoC

The CPU-baseline PoC uses numpy for tensor allocation and manipulation, which introduces 10–75× overhead compared to direct GPU operations. We address the reviewer's concern by constructing a CUDA-optimised projection using our existing Exp1 GPU measurements (§6.3): GPU reshape (0–25.68 ms), GPU D2H transfer (3.01–28.56 ms at 28–94 GB/s), and GPU FP16 translation (2.71–28.19 ms for 1024–4096 tokens). The network transfer time is unchanged; only the serialisation path is replaced.

[Table 9b-revised: CUDA-Optimised vs. CPU-Baseline PoC (4B, 3 Gbps)]

| Tokens | CPU Total (ms) | CUDA Extract (ms) | CUDA Translate (ms) | Transfer (ms) | CUDA Total (ms) | Local Prefill (ms) | Speedup vs CPU | Verdict |
|--------|---------------|-------------------|---------------------|---------------|----------------|-------------------|---------------|---------|
| 128 | 102.1 | 17.5 | — | 26.7 | 44.6 | 19.0 | 2.3× | EDGE |
| 256 | 208.2 | 7.1 | — | 53.3 | 61.0 | 32.9 | 3.4× | EDGE |
| 512 | 418.7 | 9.3 | — | 106.7 | 116.4 | 61.0 | 3.6× | EDGE |
| 1024 | 848.1 | 12.3 | 2.7 | 213.3 | 226.1 | 160.3 | 3.7× | EDGE |
| 2048 | 1876.3 | 27.7 | 8.4 | 426.7 | 454.9 | 309.1 | 4.1× | EDGE |
| 4096 | 4370.3 | 56.8 | 25.7 | 853.3 | 910.6 | 651.0 | 4.8× | EDGE |

*CUDA Extract = GPU D2H (torch.cuda.synchronize) + GPU reshape overhead. CUDA Translate = GPU FP16→FP16 reshape to PagedAttention layout (only needed for ≥1024 tokens where block alignment requires non-trivial permutation). Transfer at 3 Gbps. Restore estimated at 0.5 ms (GPU memcpy).*

At 3 Gbps, even with CUDA optimisation the hybrid path remains slower than local prefill—the bottleneck has shifted from serialisation to network transfer. However, the serialisation overhead is now negligible (12–57 ms vs. 63–2167 ms), making higher bandwidth the sole remaining variable.

**Multi-bandwidth break-even, CUDA-optimised PoC (1024 tokens, 4B):**

| BW (Gbps) | Transfer (ms) | CUDA Total (ms) | Local Prefill (ms) | Verdict | vs. CPU-Baseline |
|-----------|--------------|----------------|-------------------|---------|-----------------|
| 0.5 | 1280.0 | 1292.8 | 160.3 | EDGE | 1.5× faster |
| 1.0 | 640.0 | 652.8 | 160.3 | EDGE | 1.9× faster |
| 3.0 | 213.3 | 226.1 | 160.3 | EDGE | 3.7× faster |
| **5.0** | **128.0** | **140.8** | **160.3** | **HYBRID** | — |
| 7.5 | 85.3 | 98.1 | 160.3 | **HYBRID** | — |
| **10.0** | **64.0** | **76.8** | **160.3** | **HYBRID** | **2.1× faster than local** |

**Feasibility domain.** The CUDA-optimised PoC reveals a clear feasibility boundary:

[Table 9b-feasibility: Hybrid vs. Edge by Bandwidth and Token Count (CUDA-Optimised)]

| BW (Gbps) | 128 tok | 256 tok | 512 tok | 1024 tok | 2048 tok | 4096 tok |
|-----------|---------|---------|---------|----------|----------|----------|
| 0.5 | EDGE | EDGE | EDGE | EDGE | EDGE | EDGE |
| 1.0 | EDGE | EDGE | EDGE | EDGE | EDGE | EDGE |
| 3.0 | EDGE | EDGE | EDGE | EDGE | EDGE | EDGE |
| **5.0** | EDGE | EDGE | EDGE | **HYBRID** | **HYBRID** | **HYBRID** |
| **7.5** | EDGE | **HYBRID** | **HYBRID** | **HYBRID** | **HYBRID** | **HYBRID** |
| **10.0** | EDGE | **HYBRID** | **HYBRID** | **HYBRID** | **HYBRID** | **HYBRID** |

**Bandwidth thresholds for hybrid superiority:**

| Token Count | Min BW (Gbps) | KVCache Size | Practical Context |
|------------|--------------|-------------|-------------------|
| 128 | 76.2 | 10 MB | Infeasible |
| 256 | 6.3 | 20 MB | Gigabit + compression |
| 512 | 6.3 | 40 MB | Gigabit + compression |
| 1024 | 4.3 | 80 MB | Enterprise WiFi 6 / wired Ethernet |
| 2048 | 4.6 | 160 MB | Wired Ethernet |
| 4096 | 4.3 | 320 MB | Wired Ethernet |

**Honest assessment.** At ≤3 Gbps (typical mobile or home networks), the hybrid architecture cannot outperform pure-edge prefill at any token count—the network transfer is the irreducible bottleneck. At ≥5 Gbps (enterprise WiFi 6, wired Gigabit Ethernet with compression), the hybrid architecture becomes viable for ≥256-token prefixes, achieving 2.1× speedup over local prefill at 10 Gbps / 1024 tokens. This delineates the **actual feasible domain** of the hybrid architecture: enterprise or wired edge deployments with reliable high-bandwidth connectivity, not mobile or residential networks.

### 6.7c Network Jitter Sensitivity Analysis

The break-even analysis (§6.6) assumes deterministic bandwidth. In practice, edge-cloud networks exhibit jitter. We analyse the impact via Monte Carlo simulation (1000 samples per configuration) on Qwen3.5-4B with 1024-token prefix (80 MB KVCache):

[Table 9c: Jitter Sensitivity — P(hybrid wins) vs. σ]

| σ (Gbps) | P(hybrid wins) | Mean Savings (ms) | Mean Hybrid (s) | Std (s) |
|----------|---------------|-------------------|-----------------|---------|
| 0.1 | 0.0% | −30.1 | 0.234 | 0.007 |
| 0.3 | 6.2% | −32.2 | 0.236 | 0.023 |
| 0.5 | 16.6% | −36.8 | 0.241 | 0.042 |
| 1.0 | 33.8% | −286.7 | 0.490 | 3.701 |
| 1.5 | 39.0% | −1547.1 | 1.751 | 9.525 |
| 2.0 | 42.3% | −3957.5 | 4.161 | 15.412 |

*Mean bandwidth = 3 Gbps. "Hybrid wins" means total hybrid time < local prefill time (0.204 s).*

**Key finding (CPU-baseline).** Even with mean bandwidth of 3 Gbps—which exceeds the break-even threshold—the CPU-baseline hybrid architecture fails to outperform pure-edge in the majority of simulations. At σ = 0.3 Gbps (10% jitter), P(hybrid wins) = 6.2%; at σ = 1.0 Gbps (33% jitter), P(hybrid wins) = 33.8%. The heavy tail is severe: at σ = 2.0 Gbps, the mean hybrid time reaches 4.16 s (20× the local prefill) with std = 15.4 s. **No σ value achieves P ≥ 95%**, indicating that the CPU-baseline hybrid requires bandwidth stabilisation to be reliable.

**CUDA-optimised jitter analysis at 10 Gbps.** The CPU-baseline jitter analysis is dominated by serialisation overhead, not network jitter. We repeat the Monte Carlo analysis with CUDA-optimised extract/translate times and mean bandwidth of 10 Gbps (enterprise WiFi 6 or wired Ethernet):

[Table 9c-revised: CUDA-Optimised Jitter Sensitivity — P(hybrid wins) vs. σ (10 Gbps, 1024 tokens)]

| σ (Gbps) | P(hybrid wins) | Mean Total (ms) | P99 (ms) | Local Prefill (ms) |
|----------|---------------|-----------------|----------|-------------------|
| 0.1 | **100.0%** | 76.8 | 78.3 | 160.3 |
| 0.3 | **100.0%** | 76.9 | 81.7 | 160.3 |
| 0.5 | **100.0%** | 77.0 | 85.0 | 160.3 |
| 1.0 | **100.0%** | 77.4 | 96.4 | 160.3 |
| 2.0 | **99.67%** | 79.7 | 134.9 | 160.3 |

*CUDA non-transfer overhead = 12.81 ms (extract + translate + restore). Network transfer at 10 Gbps mean.*

**Contrast with CPU-baseline.** The CPU-baseline PoC at 3 Gbps showed P(hybrid wins) = 0% for all σ values. The CUDA-optimised PoC at 10 Gbps achieves P(hybrid wins) ≥ 99.67% for all tested σ values. This confirms that the original jitter analysis's pessimism was an artefact of the CPU serialisation bottleneck, not a fundamental limitation of the hybrid architecture. However, this reliability is contingent on maintaining ≥10 Gbps mean bandwidth—a condition that holds for enterprise wired networks but not for mobile or residential connections.

### 6.7d Multi-Tenant Edge Scenario

When multiple edge devices share a single cloud connection, effective per-device bandwidth decreases. We analyse break-even for a 10 Gbps shared link with 1024-token prefix on 4B, comparing the CPU-baseline and CUDA-optimised PoCs:

[Table 9d: Multi-Tenant Break-Even (1024 tokens, 4B, 10 Gbps shared)]

| N Devices | Per-Device BW | CPU-Baseline Verdict | CUDA-Optimised Verdict |
|-----------|--------------|---------------------|----------------------|
| 1 | 10.0 Gbps | HYBRID | **HYBRID** (76.8 ms vs 160.3 ms) |
| 2 | 5.0 Gbps | HYBRID | **HYBRID** (140.8 ms vs 160.3 ms) |
| 3 | 3.33 Gbps | EDGE | EDGE (204.8 ms vs 160.3 ms) |
| 5 | 2.0 Gbps | EDGE | EDGE (332.8 ms vs 160.3 ms) |
| 10 | 1.0 Gbps | EDGE | EDGE (652.8 ms vs 160.3 ms) |

**Break-even: 2 devices** sharing 10 Gbps (per-device 5.0 Gbps) with CUDA optimisation, compared to 3 devices with the CPU-baseline. The improvement is modest because network transfer—not serialisation—is the binding constraint at these bandwidths. For the 0.8B model (24 KB/token), the break-even shifts to 1 device (10 Gbps dedicated). These results assume equal bandwidth sharing across devices; in practice, QoS mechanisms and switch contention may reduce effective per-device bandwidth below the theoretical division.

### 6.9 Cost-Benefit Assessment and Recommendation

The preceding analysis establishes the hybrid architecture's technical feasibility under specific bandwidth conditions. We now assess whether this feasibility translates into practical engineering value by quantifying the cost-benefit tradeoff.

[Table 9e: Hybrid Architecture Engineering Cost]

| Component | Function | Est. Lines | Effort | Risk | Alternative |
|-----------|----------|-----------|--------|------|-------------|
| KVCache translation layer | llama.cpp ↔ vLLM layout | ~2,000 | 4–6 mo. | Medium | Standardised KVCache schema |
| Edge-cloud gateway | Migration scheduling + protocol | ~3,000 | 6–8 mo. | High | Local disk persistence |
| Prefix cache client | Hash matching + serialisation | ~1,500 | 2–3 mo. | Low | Local disk persistence |
| Privacy filter | Entropy-based grading | ~1,000 | 1–2 mo. | Low | — |
| Scheduling engine | Bandwidth-aware scheduling | ~1,500 | 3–4 mo. | Medium | Fixed local-first policy |
| **Total** | | **~9,000** | **16–23 mo.** | | |

[Table 9f: Marginal Benefit vs. Engineering Cost]

| Configuration | Interaction Benefit | Absolute Time Saved | Net After Overhead | ROI |
|--------------|--------------------|--------------------|-------------------|-----|
| $N=3$, $P_s+P_a=60$ (typical) | 120 tokens | ~15–30 ms | ~5–10 ms | **Negative** |
| $N=10$, $P_s+P_a=60$ | 540 tokens | ~56–161 ms | ~36–141 ms | **Very low** |
| $N=10$, $P_s+P_a=500$ | 4,500 tokens | ~0.5–1.5 s | ~0.5–1.5 s | **Low** |
| $N=50$, $P_s+P_a=2000$ | 98,000 tokens | ~10–30 s | ~10–30 s | **Moderate** |

*Net benefit accounts for gateway processing (5–15 ms/invocation) and translation overhead (2.7 ms at 1024 tokens, GPU).*

**Local KVCache persistence as the practical alternative.** Local disk-backed KVCache persistence (§8, H1.1) achieves equivalent cold-start elimination at approximately 1/6 the engineering cost (2–3 months vs. 16–23 months), with no network dependency, no privacy risk, and full deterministic latency. The persistence approach stores the KVCache prefix hash and blocks on local SSD, restoring them on subsequent sessions without re-encoding—functionally equivalent to the cloud prefix prefetch for the single-device case.

Our analysis reveals that the hybrid SIG-Mooncake architecture is technically feasible but economically marginal for most edge deployments. The interaction term $(N-1)(P_s + P_a)$ contributes 0.23–3.82% of SIG savings in typical configurations, while the engineering cost is substantial. We recommend **local KVCache persistence** as the practical path for cold-start elimination, relegating the full hybrid architecture to the narrow sweet spot of edge devices without local storage, $N \geq 5$ sessions with $P_s + P_a \geq 500$ tokens, reliable network connectivity ($\geq 5$ Gbps), and first-turn latency SLOs.

---

## 7. Prefix Caching Meets Injection: Asymmetric Interaction Analysis

### 7.1 Token Composition Model

We define the token composition of a multi-agent scenario as:

- **Shared prefix** $P_s$: System prompt shared across all agents and all requests
- **Per-agent prefix** $P_a$: Agent-specific context (shared across an agent's requests but not across agents)
- **Per-request private** $P_r$: Request-specific context (unique to each request)
- **Per-step incremental** $I_k$: Tool result at step $k$ within a request (unique to each step)

For a request with chain depth $K$, the total tokens needed at step $k$ are:

$$\text{Tokens at step } k = P_s + P_a + P_r + \sum_{j=1}^{k} I_j$$

### 7.2 Per-Step and Total Prefill by Mechanism

[Table 10: Tokens Prefilled at Step $k$ Under Each Mechanism]

| Mechanism | Tokens at Step $k$ | Total Prefill Tokens (K steps) |
|-----------|-------------------|-------------------------------|
| AppLoop (baseline) | $P_s + P_a + P_r + \sum_{j=1}^{k} I_j$ | $K(P_s + P_a + P_r) + \sum_{k=1}^{K}\sum_{j=1}^{k} I_j$ |
| Prefix caching only | $P_r + \sum_{j=1}^{k} I_j$ (after first request) | $K \cdot P_r + \sum_{k=1}^{K}\sum_{j=1}^{k} I_j$ |
| SIG only | $I_k$ (injection) | $\sum_{k=1}^{K} I_k$ |
| Prefix caching + SIG | $I_k$ (injection, prefix cached) | $\sum_{k=1}^{K} I_k$ |

**Key observation.** Within a single session, prefix caching adds no benefit on top of SIG, because SIG already avoids re-encoding $P_s + P_a + P_r$ through injection continuity. The interaction is *cross-session*.

### 7.3 Multi-Session Analysis

For $N$ sessions (each with chain depth $K$), total prefill tokens:

[Table 11: Multi-Session Total Prefill Tokens]

| Mechanism | Total Prefill Tokens |
|-----------|---------------------|
| AppLoop | $N \left[ K(P_s + P_a + P_r) + \sum_{k=1}^{K}\sum_{j=1}^{k} I_j \right]$ |
| Prefix caching only | $K(P_s + P_a + P_r) + \sum_{k=1}^{K}\sum_{j=1}^{k} I_j + (N-1)\left[ K \cdot P_r + \sum_{k=1}^{K}\sum_{j=1}^{k} I_j \right]$ |
| SIG only | $N(P_s + P_a + P_r) + N \sum_{k=1}^{K} I_k$ |
| Prefix caching + SIG | $(P_s + P_a + P_r) + \sum_{k=1}^{K} I_k + (N-1)\left[ P_r + \sum_{k=1}^{K} I_k \right]$ |

### 7.4 Savings Derivation

**Savings over AppLoop baseline:**

**Prefix caching alone:**
$$S_{\text{prefix}} = (N-1) \cdot K \cdot (P_s + P_a)$$

Prefix caching saves re-encoding the shared prefix $(P_s + P_a)$ for $K$ steps across $(N-1)$ subsequent sessions.

**SIG alone:**
$$S_{\text{SIG}} = N \cdot (K-1)(P_s + P_a + P_r) + N \left[ \sum_{k=1}^{K}\sum_{j=1}^{k} I_j - \sum_{k=1}^{K} I_k \right]$$

SIG saves re-encoding the growing prefix $(P_s + P_a + P_r)$ for steps 2 through $K$ within each session, plus the quadratic accumulation of incremental tokens.

**Combined (prefix caching + SIG):**
$$S_{\text{combined}} = S_{\text{SIG}} + \Delta S_{\text{prefix|SIG}}$$

Where the marginal benefit of adding prefix caching on top of SIG is:

$$\Delta S_{\text{prefix|SIG}} = (N-1)(P_s + P_a)$$

This term arises because SIG alone must re-encode $P_s + P_a$ at the start of each new session (cold-start cost), but prefix caching eliminates this cost for sessions 2 through $N$.

### 7.5 Asymmetric Interaction Characterisation

The original hypothesis (H4) posited superadditivity: $S_{\text{combined}} > S_{\text{prefix}} + S_{\text{SIG}}$. Our analysis reveals a more nuanced picture.

The combined savings are less than the sum of individual savings ($S_{\text{combined}} < S_{\text{prefix}} + S_{\text{SIG}}$) because the savings overlap—both mechanisms reduce re-encoding of $P_s + P_a$ tokens, albeit in different temporal scopes. However, the combined savings exceed either mechanism alone, and the marginal benefit of adding prefix caching to SIG is non-zero: $(N-1)(P_s + P_a)$ tokens.

We characterise the relationship as **asymmetric and conditionally additive**: SIG is the primary mechanism capturing 96–99.8% of savings; prefix caching provides a marginal additive term that is negligible for typical configurations ($P_s + P_a \leq 300$, $N \leq 5$) and modest even in extreme configurations ($P_s + P_a = 5000$, $N = 50$, yielding ~4% marginal gain):

1. **SIG** operates on the *within-session* axis—eliminating redundant prefill of the growing prefix within a session—and captures the vast majority of achievable savings.
2. **Prefix caching** operates on the *cross-session* axis—eliminating redundant prefill of the shared prefix across sessions—providing conditional incremental benefit.
3. **The interaction term** $(N-1)(P_s + P_a)$ represents the genuine additional benefit at session boundaries, where prefix caching eliminates the cold-start cost that SIG alone incurs. This term is quantitatively marginal in all tested configurations.

The interaction is *conditional*: when $P_s + P_a$ is small (as in Paper 4's deep-chain regime, where $P_s + P_a \approx 80$ tokens), the interaction term is negligible and prefix caching adds essentially no benefit on top of SIG. When $P_s + P_a$ is large (multi-agent systems with extensive shared tool descriptions, $P_s + P_a \geq 500$), the interaction term becomes modestly material—but even then, SIG alone captures $\geq 96\%$ of the benefit.

### 7.6 Validation Against Paper 4 Data

Paper 4's R6 experiment provides a critical validation point: $P_s + P_a \approx 80$ tokens, $P_r \approx 0$, $I_k \approx 80$ tokens per step, $K = 30$ steps.

**Prefix caching savings** (within a single session): $(K-1) \times (P_s + P_a) = 29 \times 80 = 2,320$ tokens.

**SIG savings** (within a single session): $(K-1)(P_s + P_a + P_r) + [\sum_{k=1}^{K}\sum_{j=1}^{k} I_j - \sum_{k=1}^{K} I_k] = 2,320 + 34,800 = 37,120$ tokens.

**Prefix caching as fraction of total savings**: $2,320 / (37,120 + 2,320) \approx 5.9\%$.

This is consistent with Paper 4's finding that prefix caching provides less than 3% token reuse in the deep-chain regime (the slight discrepancy arises because our model counts cross-session reuse, whilst Paper 4 measured within-session reuse). **The key validation**: when $P_s + P_a$ is small relative to $K \times I_k$, prefix caching contributes negligibly, confirming that SIG and prefix caching operate on orthogonal axes—SIG dominates the deep-chain regime while prefix caching provides marginal cross-session benefit.

**Empirical multi-session validation.** We validate the asymmetric interaction model with end-to-end latency measurements on RTX 4070 SUPER, running $N = 3$ sessions with $K = 5$ tool-call steps each:

[Table 11b: Multi-Session Latency Validation (N=3)]

| Prefix Size | $P_s + P_a$ (tokens) | SIG Only (s) | SIG + PrefixCache (s) | PC Savings | Savings % |
|------------|----------------------|-------------|----------------------|------------|-----------|
| Small | 5 | 0.261 | 0.195 | 0.066 s | 25.2% |
| Medium | 60 | 0.227 | 0.198 | 0.029 s | 12.8% |
| Large | 314 | 0.323 | 0.234 | 0.089 s | 27.5% |

The results confirm the model's prediction: prefix caching provides measurable benefit in multi-session scenarios, with savings proportional to $P_s + P_a$. The medium-prefix case shows lower percentage savings because the SIG-only baseline is already fast (the prefix is a smaller fraction of total computation). The large-prefix case demonstrates the strongest absolute benefit (89 ms), consistent with the interaction term $(N-1)(P_s + P_a) = 2 \times 314 = 628$ tokens of avoided re-prefill.

**Extended validation (N=10).** We extend the measurement to $N = 10$ sessions with $P_s + P_a = 60$ tokens (system prompt + tool descriptions), $K = 5$ steps:

| Model | SIG Only (s) | SIG + PrefixCache (s) | PC Savings | Savings % |
|-------|-------------|----------------------|------------|-----------|
| Qwen3.5-0.8B | 0.315 | 0.259 | 0.056 s | 17.9% |
| Qwen3.5-4B | 0.786 | 0.625 | 0.161 s | 20.5% |

The interaction term is $(N-1)(P_s + P_a) = 9 \times 60 = 540$ tokens of avoided re-prefill across 10 sessions. Both models show consistent savings (18–21%), confirming the asymmetric interaction model at higher session counts.

**Interaction term scaling analysis.** We verify the theoretical interaction term $\Delta S_{\text{prefix|SIG}} = (N-1)(P_s + P_a)$ against the model for $N = 1$–$50$ with multi-agent parameters ($P_s = 500$, $P_a = 1500$, $P_r = 200$, $I_k = 100$, $K = 20$):

| $N$ | $\Delta S_{\text{prefix|SIG}}$ (empirical) | $(N-1)(P_s + P_a)$ (predicted) | Match |
|-----|-------------------------------------------|-------------------------------|-------|
| 1 | 0 | 0 | ✓ |
| 2 | 2,000 | 2,000 | ✓ |
| 5 | 8,000 | 8,000 | ✓ |
| 10 | 18,000 | 18,000 | ✓ |
| 20 | 38,000 | 38,000 | ✓ |
| 50 | 98,000 | 98,000 | ✓ |

All predictions match exactly, confirming the analytical model across three orders of magnitude in $N$.

**Interaction term as fraction of SIG savings.** The interaction term's relative importance grows with $P_s + P_a$ but remains modest for typical configurations:

| $P_s + P_a$ (tokens) | Interaction $(N-1)(P_s + P_a)$ | SIG Savings | Interaction / SIG Savings |
|----------------------|-------------------------------|-------------|--------------------------|
| 60 | 540 | 239,400 | 0.23% |
| 300 | 2,700 | 285,000 | 0.95% |
| 1,000 | 9,000 | 418,000 | 2.15% |
| 2,000 | 18,000 | 608,000 | 2.96% |
| 5,000 | 45,000 | 1,178,000 | 3.82% |

*At $N = 10$, $K = 20$, $P_r = 200$, $I_k = 100$.*

For typical agent workloads ($P_s + P_a \leq 500$ tokens), the interaction term contributes less than 1.5% of total savings—confirming that SIG alone captures the vast majority of the benefit, with prefix caching providing a small but measurable marginal gain. For large-scale multi-agent deployments ($P_s + P_a \geq 2000$ tokens), the interaction term reaches 3–4%, making prefix caching a worthwhile optimisation.

### 7.7 Projection to Multi-Agent Scenarios

For a multi-agent system with extensive shared tool descriptions:

**Parameters**: $P_s = 500$ tokens (system prompt), $P_a = 1,500$ tokens (per-agent tool descriptions), $P_r = 200$ tokens (per-request context), $I_k = 100$ tokens per step, $K = 20$ steps, $N = 5$ agents.

[Table 12: Multi-Agent Savings Projection]

| Mechanism | Total Prefill | Savings vs. AppLoop | Savings % |
|-----------|--------------|-------------------|----------|
| AppLoop | 325,000 | 0 | 0% |
| Prefix caching only | 165,000 | 160,000 | 49.2% |
| SIG only | 21,000 | 304,000 | 93.5% |
| Prefix caching + SIG | 13,000 | 312,000 | 95.9% |

**Marginal benefit of prefix caching on top of SIG**: $312,000 - 304,000 = 8,000$ tokens = $(N-1)(P_s + P_a) = 4 \times 2,000 = 8,000$. This represents the cold-start elimination for sessions 2 through 5.

**Sensitivity analysis.** The interaction term scales linearly with both $N$ (number of sessions) and $P_s + P_a$ (shared prefix size). For a large-scale deployment with $N = 50$ agents and $P_s + P_a = 5,000$ tokens, the interaction term reaches $49 \times 5,000 = 245,000$ tokens—comparable to SIG's within-session savings, making both mechanisms equally important.

[Figure 4: Savings by Mechanism for Varying Session Counts]

[Figure 5: Interaction Term Scaling with $N$ and $P_s + P_a$]

---

## 8. Roadmap: From Edge-Only to Distributed SIG

### 8.1 Three-Horizon Roadmap

[Figure 6: Three-Horizon Roadmap]

**Horizon 1: Edge Enhancement (0–12 months).**

*Goal:* Bring the most transferable Mooncake patterns to SIG's edge deployment.

- **H1.1: Persistent KVCache store.** Implement disk-backed KVCache persistence in llama.cpp, enabling cross-session prefix reuse without network connectivity. This is the edge-adapted version of Mooncake's prefix-hash matching, operating locally on the device. Estimated engineering effort: 2–3 months for llama.cpp extension.

- **H1.2: DRAM offloading with CompSIG.** Extend CompSIG's compression pipeline with DRAM-backed KVCache offloading, implementing the two-tier storage model (VRAM → DRAM) identified in P3. This extends SIG's effective context window by approximately 3.5× without quality loss. Estimated engineering effort: 1–2 months for llama.cpp memory management extension.

- **H1.3: Multi-agent prefix sharing.** Implement hash-based KVCache block matching within llama.cpp for scenarios where multiple agents on the same device share common system prompts and tool descriptions. This is the local, single-device version of Mooncake's global prefix-hash matching. Estimated engineering effort: 1–2 months.

**Horizon 2: Framework Porting (12–24 months).**

*Goal:* Port SIG to vLLM and SGLang, enabling SIG to operate within Mooncake's ecosystem.

- **H2.1: SIG on vLLM.** Implement SIG's injection continuity as a vLLM extension, leveraging PagedAttention's block-based KVCache management. The key API surface needed: `inject_tokens(session_id, token_ids)` that extends the KVCache without re-encoding the prefix. This requires modifying vLLM's `ForwardBatch` to support incremental KVCache extension. Estimated engineering effort: 4–6 months.

- **H2.2: SIG on SGLang.** Implement SIG as an SGLang extension, leveraging RadixAttention's prefix tree for automatic prefix detection. The injection operation maps naturally to SGLang's radix tree: the injected tokens extend an existing leaf node rather than creating a new tree. Estimated engineering effort: 3–4 months.

- **H2.3: SIG as a Mooncake Store backend.** Expose SIG's injection-continuous KVCache as a specialised store within Mooncake's distributed pool. Edge SIG nodes can register their KVCache with the global pool, enabling cloud-side prefix caching and cross-node sharing. Estimated engineering effort: 3–4 months.

**Horizon 3: Distributed SIG with Mooncake Transport (24–36 months).**

*Goal:* Realise the full hybrid architecture with edge-cloud KVCache migration.

- **H3.1: KVCache translation layer.** Implement the five-step translation pipeline (§6.3) as a standalone library, supporting llama.cpp ↔ vLLM/SGLang format conversion. The translation layer must be model-aware (handling different architectures' KVCache dimensions) and quantization-aware (supporting FP16, FP8, and Q4 formats). Estimated engineering effort: 4–6 months.

- **H3.2: Edge-cloud gateway.** Implement the Edge-Cloud Gateway component with the three interface contracts (KVCache_Migrate, KVCache_Prefill, existing Transfer Engine). The gateway must handle network partitions (fallback to pure-edge SIG), privacy filtering (strip sensitive tokens before cloud transfer), and scheduling (decide when to use cloud vs. edge). Estimated engineering effort: 6–8 months.

- **H3.3: Distributed SIG scheduling.** Implement the scheduling policy (§6.4) as an extension of Mooncake's Conductor, adding edge-awareness to the global scheduler. The scheduler must account for edge-cloud bandwidth variability, privacy constraints, and the cost of KVCache translation. Estimated engineering effort: 3–4 months.

### 8.2 Ecosystem Integration Path

Mooncake's ecosystem integration provides a natural path for SIG's expansion:

| Mooncake Integration | SIG Adaptation | Horizon |
|---------------------|---------------|---------|
| Mooncake Store (distributed KVCache pool) | SIG's injection-continuous KVCache as a specialised store | H2.3 |
| Transfer Engine (RDMA/TCP) | Edge-cloud KVCache migration via TCP mode | H3.2 |
| vLLM KV Connector | SIG injection as a KV Connector extension | H2.1 |
| SGLang HiCache | SIG injection as a HiCache extension | H2.2 |
| Conductor scheduler | Edge-aware scheduling extension | H3.3 |

### 8.3 Key Dependencies and Risks

1. **llama.cpp API stability.** SIG currently depends on llama.cpp's `kv_cache_seq_rm`, `kv_cache_seq_cp`, and related APIs. Changes to these APIs (e.g., the transition from static to dynamic cell-based allocation) require SIG adaptation. Mitigation: contribute SIG extensions upstream to llama.cpp.

2. **vLLM/SGLang injection API.** Neither vLLM nor SGLang currently exposes an "inject tokens into existing KVCache" API. Implementing SIG on these frameworks requires adding this capability. Mitigation: design the API in collaboration with the vLLM and SGLang maintainers.

3. **KVCache format fragmentation.** Different frameworks (llama.cpp, vLLM, SGLang, TensorRT-LLM) use different KVCache layouts. The translation layer must handle this fragmentation. Mitigation: propose a standardised KVCache metadata schema (§6.3) to the community.

4. **Network reliability.** The hybrid architecture assumes network connectivity between edge and cloud. In practice, networks are unreliable (packet loss, congestion, outages). Mitigation: the scheduling policy includes fallback to pure-edge SIG (§6.4), ensuring graceful degradation.

---

## 9. Related Work

### 9.1 PD Disaggregation Lineage

The separation of prefill and decode into independent resources has evolved through several systems. **Orca** [7] introduced iteration-level scheduling and continuous batching, establishing the serving foundation. **DistServe** [8] (OSDI 2024) disaggregated prefill and decode for throughput, demonstrating that separation eliminates interference and improves both TTFT and TBT. **Splitwise** [9] (MLSys 2024) extended disaggregation with KVCache transfer between prefill and decode nodes. **Mooncake** [6] (FAST 2025) made the crucial shift to a KVCache-centric design: rather than treating disaggregation as a compute optimisation, Mooncake treats it as a KVCache lifecycle optimisation, with the global pool, Transfer Engine, and Conductor all organised around KVCache management. Our KFC framework formalises this shift as a general principle that applies beyond the cloud regime.

### 9.2 KVCache Memory Management

**PagedAttention** [10] (SOSP 2023) introduced virtual memory management for KVCache, using fixed-size blocks with on-demand allocation—eliminating memory fragmentation and enabling memory sharing across requests. PagedAttention is the substrate that Mooncake builds on; SIG's llama.cpp-based implementation uses a different memory model (contiguous allocation with sequence management APIs). The format translation analysis in §6.3 addresses the gap between these models.

**RadixAttention** [11] (ICML 2024, SGLang) introduced a radix tree for automatic prefix sharing, enabling fine-grained KVCache reuse across requests with shared prefixes. Mooncake's prefix-hash matching is a distributed version of RadixAttention's prefix tree. SIG's injection continuity is orthogonal to prefix-based reuse—it preserves KVCache within a request rather than sharing it across requests.

### 9.3 KVCache as a Portable Object

**CacheGen** [12] (2024) treats KVCache as a compressible, streamable object, enabling KVCache transfer between devices with bandwidth-constrained networks. CacheGen's compression approach (KVCache quantisation + selective head transmission) is directly relevant to our hybrid architecture's cold-start scenario, where KVCache must be transferred from cloud to edge over bandwidth-limited connections. Our break-even analysis (§6.6) shows that CacheGen-style 4× compression reduces the break-even bandwidth from 3.6 Gbps to 0.9 Gbps, making hybrid cold-start viable on Gigabit Ethernet.

**CacheBlend** [13] (2024) addresses KVCache reuse with partial recomputation, selectively recomputing cached KVCache entries that may have become stale. This is relevant to the hybrid architecture's session migration scenario, where KVCache transferred from edge to cloud may need partial recomputation to account for positional encoding differences.

### 9.4 Edge Inference Systems

**llama.cpp** is the de facto edge inference library, supporting a wide range of model architectures and quantisation formats. SIG is implemented as a llama.cpp extension, leveraging its `kv_cache_seq_rm` and related APIs for KVCache management. The hybrid architecture's translation layer (§6.3) bridges llama.cpp's contiguous KVCache layout with vLLM's PagedAttention format.

**SARATHI** [14] (2024) introduced chunked prefill for mitigating prefill-decode interference in co-located serving. Whilst SARATHI addresses the same prefill bottleneck as SIG, it does so through scheduling (chunking long prefills) rather than through KVCache preservation. SARATHI is complementary to SIG: chunked prefill optimises the *cost* of prefill, whilst SIG *eliminates* the need for prefill at tool-call boundaries.

### 9.5 Multi-Agent and Agent Serving Systems

The rise of LLM-based agents (AutoGPT, Devin, SWE-agent) creates new demands on serving systems: long-running sessions with many tool calls, growing KVCache that exceeds single-node VRAM, and the need for injection continuity across tool-call boundaries. Current cloud serving systems (vLLM, SGLang, TensorRT-LLM) handle these workloads through KVCache transfer between nodes, but lack a principled framework for injection continuity. SIG's suspend-inject-resume cycle provides this framework, and our reverse transferability analysis (§5.4) shows that SIG-inspired injection can improve Mooncake's handling of long-running agent sessions by enabling incremental KVCache growth rather than full re-encoding after migration.

---

## 10. Conclusion

This paper has demonstrated that Mooncake and SIG—two independently developed LLM inference systems targeting opposite deployment regimes—are convergent architectures that share a common "KVCache-as-first-class-citizen" design philosophy. The KFC framework formalises this convergence as a unified optimisation objective that subsumes both systems as regime-specific instantiations: Mooncake is the optimal solution under cloud constraints (multi-tenant, cluster-scale, throughput-oriented), and SIG is the optimal solution under edge constraints (single-user, device-scale, latency-oriented).

The 8-dimensional convergence analysis reveals that 5 of 8 dimensions show strong or moderate convergence, with the remaining 3 dimensions (transport, scale, failure handling) exhibiting regime-determined divergence. The transferability analysis identifies multi-tier storage (via CompSIG) and prefix-hash matching (for cross-session reuse) as the most transferable Mooncake patterns, whilst PD disaggregation is fundamentally incompatible with single-device edge deployment.

The hybrid SIG-Mooncake architecture composes edge injection continuity with cloud prefix caching, connected by a KVCache translation layer with bounded overhead (~2.7 ms on GPU for 1024-token session-level migration, measured on RTX 4070 SUPER). The asymmetric interaction analysis demonstrates that SIG captures 96–99.8% of achievable prefill reduction as a standalone mechanism, with prefix caching providing conditional incremental benefit (0.23–3.82% of SIG savings) that is material only in the narrow sweet spot of $N \geq 5$ sessions with $P_s + P_a \geq 500$ tokens. A CUDA-optimised feasibility analysis identifies the hybrid architecture's actual feasible domain: $\geq 5$ Gbps with $\geq 256$-token prefixes, achieving 2.1× speedup over local prefill at 10 Gbps / 1024 tokens. At $\leq 3$ Gbps (typical mobile or residential networks), pure-edge SIG remains the superior choice. For most edge deployments, **local KVCache persistence** provides equivalent cold-start elimination at substantially lower engineering cost.

This paper extends Paper 1's positioning of CO+SIG from *non-competition* with cloud frameworks to *architectural convergence* with them. SIG and Mooncake share a common KVCache-first design philosophy, formalised as the KFC framework. However, the practical implications of this convergence are asymmetric: SIG captures 96–99.8% of achievable edge-side savings as a standalone mechanism. The composition of SIG with cloud prefix caching is technically feasible but yields marginal benefit at substantial engineering cost. We recommend local KVCache persistence as the practical path for most edge deployments, relegating the full hybrid architecture to the narrow sweet spot of large-scale multi-agent edge deployments without local storage and with reliable high-bandwidth connectivity. The three-horizon roadmap charts a path from edge-only SIG to distributed SIG with Mooncake transport, identifying the technical requirements and dependencies at each stage.

The convergence of Mooncake and SIG on the same design philosophy from opposite starting points suggests a deeper truth about LLM inference system design: as models grow larger and context windows grow longer, KVCache management becomes the central architectural concern, regardless of deployment regime. Systems that treat KVCache as a first-class citizen—preserving it, sharing it, and moving it efficiently—will outperform systems that treat it as a transient byproduct. The KFC framework makes this principle explicit and transferable, providing a foundation for future systems that span the edge-cloud boundary.

### 10.1 Limitations and Future Work

This paper's original limitation was the absence of empirical validation for the hybrid architecture. We have addressed this through targeted experiments on Qwen3.5-0.8B and Qwen3.5-4B (GQA) on NVIDIA RTX 4070 SUPER:

**Validated claims:**
- §6.3: KVCache format translation overhead confirmed at ~2.7 ms (GPU, FP16→FP16, 1024 tokens), within the original ~1–7 ms estimate. CPU-based translation is 10–50× slower, qualifying the estimate's scope.
- §6.6: Break-even bandwidth measured at ~4.1 Gbps (GQA, raw) / ~1.0 Gbps (4× compressed), close to the original 3.6 Gbps estimate. The original estimate's near-agreement was coincidental (overestimated KVCache size offset by overestimated prefill time).
- §6.7b: End-to-end migration PoC confirms the five-step pipeline is functional. The CPU-baseline PoC reveals serialisation as the bottleneck at 256+ tokens. A CUDA-optimised projection (using GPU D2H measurements at 28–94 GB/s) shifts the bottleneck to network transfer and identifies the feasible domain: hybrid wins at $\geq 5$ Gbps with $\geq 256$-token prefixes, achieving 2.1× speedup at 10 Gbps / 1024 tokens.
- §6.7c: Monte Carlo jitter analysis at 3 Gbps shows the CPU-baseline hybrid fails to achieve 95% reliability. The CUDA-optimised analysis at 10 Gbps shows P(hybrid wins) $\geq 99.67\%$ for all tested σ values ($\leq 2.0$ Gbps), confirming reliability in enterprise-grade networks.
- §6.7d: Multi-tenant break-even analysis shows 2 devices can share 10 Gbps before hybrid loses advantage (4B model, CUDA-optimised), compared to 3 devices with the CPU-baseline.
- §6.8: Entropy-based privacy grading (3 tiers: L0/L1/L2) retains 65.9% locally, confirming the binary estimate (67.6%) whilst providing finer-grained, quantifiable privacy control. We note the grading's limitations: KVCache reconstruction risk, entropy-sensitivity mismatch, and the need for defence-in-depth.
- §7.6: Asymmetric interaction term $(N-1)(P_s + P_a)$ validated with multi-session latency measurements (13–28% savings at N=3, 18–21% at N=10) and verified analytically for N=1–50. Interaction term contributes 0.23–3.82% of SIG savings depending on prefix size.
- §6.8: KFC framework's generative capacity validated by predicting the "SIG + Cloud Prefix Prefetch" mode for the connected-edge scenario—a mode not present in either Mooncake or SIG.
- §9: FP16→FP8 compression ratio is content-invariant (50% across code, JSON, and natural language), though quantisation error varies by 15% (0.43–0.52).

**Remaining limitations:**
- The CUDA-optimised PoC is a projection based on measured GPU D2H transfer rates (28–94 GB/s), not an end-to-end GPU-direct implementation. A production implementation with CUDA kernels and direct NVMe/network I/O would provide definitive validation.
- The jitter analysis assumes normally distributed bandwidth; real-world network behaviour may exhibit different distributions (e.g., Pareto, log-normal) with heavier tails.
- Cross-model KVCache sharing (e.g., edge runs 0.8B, cloud runs 70B) remains unexplored—this is a fundamentally harder problem requiring KVCache projection or knowledge distillation.
- The entropy-based privacy grading is a heuristic guideline, not a security guarantee. A formal threat model incorporating adversarial inference from KVCache activations is needed before deployment.
- Partial prefix sharing (where different agents share 80% of the system prompt but diverge in the last 20%) requires prefix tree splitting at the KVCache block level, which is not yet implemented.

**Recommendation: Local KVCache persistence as the practical path.** Our analysis reveals that the hybrid SIG-Mooncake architecture, while technically feasible at $\geq 5$ Gbps, yields marginal benefit (0.23–3.82% of SIG savings) at substantial engineering cost (~9,000 lines of code, 16–23 person-months). For the majority of edge deployments, **local KVCache persistence on disk** (§8, H1.1) provides equivalent cold-start elimination at approximately 1/6 the engineering cost (2–3 months), with no network dependency, no privacy risk, and full deterministic latency. We recommend local persistence as the first priority for the SIG research program, with the full hybrid architecture reserved for the narrow sweet spot of large-scale multi-agent edge deployments without local storage and with reliable high-bandwidth ($\geq 5$ Gbps) connectivity.

Future work includes: (a) implementing local KVCache disk persistence in llama.cpp (H1.1, highest priority), (b) deploying the CUDA-optimised hybrid architecture on a testbed with edge devices and a Mooncake cluster for definitive validation, (c) extending the KFC framework to other systems (DistServe, Splitwise, CacheGen) to assess its generality, (d) conducting a formal security model analysis of edge-cloud KVCache migration, and (e) exploring cross-model KVCache sharing through knowledge distillation or KVCache projection.

---

## References

[1] Cognitive Outsourcing with Suspend-and-Inject Generation for Scalable Embodied Intelligence. *CO+SIG Research Program, Paper 1*, 2025.

[2] Beyond the Injection Engine: A Five-Dimensional Analysis of CO-SIG. *CO+SIG Research Program, Paper 2*, 2025.

[3] CO-SIG Architecture, Theory, and Empirical Design Space for Scalable Edge Intelligence. *CO+SIG Research Program, Paper 3*, 2025.

[4] Suspend-and-Inject Generation as an Edge Inference Runtime Primitive for Long-Horizon Agent Tasks. *CO+SIG Research Program, Paper 4*, 2026.

[5] Orthogonal Acceleration: Fusing Speculative Decoding with Suspend-and-Inject Generation. *CO+SIG Research Program, Paper 5*, 2026.

[6] R. Qin, Z. Li, W. He, M. Zhang, Y. Wu, W. Zheng, and X. Xu. Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving. *arXiv:2407.00079*, 2024. FAST 2025 Best Paper.

[7] Y. Yu et al. Orca: A Distributed Serving System for Transformer-Based Generative Models. *OSDI 2022*.

[8] Y. Zhong et al. DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving. *OSDI 2024*.

[9] P. Patel et al. Splitwise: Efficient generative LLM inference using phase splitting. *MLSys 2024*.

[10] W. Kwon et al. Efficient Memory Management for Large Language Model Serving with PagedAttention. *SOSP 2023*.

[11] Y. Liu et al. RadixAttention: Efficient Context-Aware Inference for Large Language Models. *ICML 2024*.

[12] X. Liu et al. CacheGen: KV Cache Compression and Streaming for Fast Large Language Model Serving. *2024*.

[13] Y. He et al. CacheBlend: Fast Large Language Model Serving for RAG with Prefix Cache Blending. *2024*.

[14] A. Agrawal et al. SARATHI: Efficient LLM Inference by Piping Parallelism with Chunked Prefills. *2024*.

[15] L. Zheng et al. SGLang: Efficient Execution of Structured Language Model Programs. *NeurIPS 2024*.

[16] T. Dao et al. FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness. *NeurIPS 2022*.

[17] T. Dao. FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning. *NeurIPS 2023*.

[18] G. Gerganov et al. llama.cpp: LLM inference in C/C++. https://github.com/ggerganov/llama.cpp, 2023–2026.

[19] vLLM Project. vLLM: High-throughput and memory-efficient inference and serving engine for LLMs. https://github.com/vllm-project/vllm, 2023–2026.

[20] Mooncake Project. Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving. https://github.com/kvcache-ai/Mooncake, 2024–2026.
