# Cognitive Outsourcing (CO) with Suspend-and-Inject Generation

> **SIG/CO Research Program** — [GitHub](https://github.com/sig-co)

Cognitive Outsourcing (CO) is an edge-AI architecture that empowers lightweight on-device language models (0.8B–4B) to orchestrate complex physical tasks by dynamically accessing external cognitive resources—local sensors, skill libraries, cloud-scale LLM "teachers"—while preserving continuous attention state through the Suspend-and-Inject Generation (SIG) primitive. SIG enables a running model to pause decoding, invoke external modules, and seamlessly absorb their responses into the KV cache without costly re-encoding, eliminating the quadratic prefill overhead of traditional tool-calling loops.

**Target regime**: single-user, single-instance edge inference (robots, smartphones, embedded systems) where prefill dominates wall-clock time and cloud-scale serving optimizations (FlashAttention, RadixAttention) are inapplicable.

## Research Program Progress

| Paper | Title | Focus | Key Results | Status |
|-------|-------|-------|-------------|--------|
| 1 | [Cognitive Outsourcing with SIG](paper/Cognitive%20Outsourcing%20with%20Suspend-and-Inject%20Generation%20for%20Scalable%20Embodied%20Intelligence.md) | CO architecture, SIG protocol | 84–96% prefill savings, 1.57× speedup, 3× information coverage | ✅ |
| 2 | [Beyond the Injection Engine](paper/Beyond_the_Injection_Engine_A_Five_Dimensional_Analysis_of_CO-SIG.md) | Theory (R1–R5): attention shift, cache lifecycle, capability gap | Per-token rate equivalence ±2%, attention head agreement 0.25→0.43, no cache degradation at 64 rounds / 13.6K tokens across 3 model families | ✅ |
| 3 | [CO-SIG Architecture & Design Space](paper/CO_SIG_Architecture_Theory_Empirical_Design_Space_for_Scalable_Edge_Intelligence.md) | Boundaries (R6–R14), cross-architecture, Batch-SIG | 2.79–5.26× deep-chain speedup, Batch-SIG 4.24–6.82× (architecture-independent), SIG Decision Framework | ✅ |
| 4 | [SIG as Edge Runtime Primitive](paper/Suspend-and-Inject%20Generation%20as%20an%20Edge%20Inference%20Runtime%20Primitive%20for%20Long-Horizon%20Agent%20Tasks.md) | Deployment (R15–R19), Kitchen benchmark | 2.54× wall-clock speedup, prefill crossover ~1.5–2B, per-token rate 108 vs 103 tok/s | ✅ |
| **5** | [**Orthogonal Acceleration**](paper/5_Orthogonal_Acceleration/paper.md) | **CO + MTP compound acceleration** | **SIG 3.50×, native MTP 1.27×, SIG+MTP 4.52× compound, ρ = 1.239 (Kitchen)** | **✅ Round 6** |
| **6** | [**Convergent KVCache Architectures**](paper/6_Convergent_KVCache_Architectures/paper.md) | **KFC unified framework, cloud-edge convergence** | **8-dim convergence analysis, 96–99.8% prefill reduction captured by SIG alone, prefix caching marginal (0.23–3.82%)** | **✅** |
| **7** | [**Disk-Backed KV-Cache Persistence**](paper/7_DiskKVCache/paper.md) | **H1.1 roadmap implementation, disk-backed prefix reuse** | **State dominated by fixed overhead (99.6% nonzero), break-even N≥6 (0.8B)/N≥14 (4B), sweet spot at 5–10% context utilization** | **✅** |
| **8** | [**State-Externalizing Cognitive Module Harnesses**](paper/8_State_Externalizing_Cognitive_Modules/paper.md) | **SECM-H architecture, agent-driven evaluation** | **H1 confirmed (76.5% externalizable); Agent-driven: 97.1% ToolAcc vs SIG 94.3% under noise; Path B confirmed (+0.050 CQ); NL rendering +0.113 CQ; H5 refuted** | **✅** |
| **9** | [**Consolidating the CO-SIG Research Program**](paper/9_Consolidated_Experiment_Restructuring/paper.md) | **Meta-empirical synthesis, unified experiments, contradiction resolution** | **Definitive SIG speedup 2.55× (±0.5%), crossover ~0.7B, all 6 contradictions resolved, utilization gap vs information loss distinction** | **✅** |

## Survey Papers

| Paper | Language | Focus |
|-------|----------|-------|
| [综述：从认知外包到智能体推理引擎](paper/survey_SIG_CO_agent_frameworks_CN.md) | 中文 | SIG 核心理念 + CO 认知架构 + 对现代智能体框架的启发 |
| [Survey: From Cognitive Outsourcing to Agent Inference Engines](paper/survey_SIG_CO_agent_frameworks_EN.md) | English | SIG core ideas + CO cognitive architecture + implications for modern agent frameworks |

## Consolidation: Paper 9

Paper 9 is a meta-empirical synthesis that resolves all six contradictions accumulated across Papers 1–8 through a unified experiment framework (8 experiments, ~270 runs, RTX 4070 SUPER, Qwen3.5-0.8B/4B Q4_K_M).

### Definitive SIG Speedup

The SIG speedup under unified protocol is **2.55×** (95% CI [2.548, 2.551], n=10, p<0.001, Cohen's d=214.2), consistent across six independent experiments within ±0.5%.

### Contradiction Resolution

| # | Contradiction | Root Cause | Resolution |
|---|--------------|------------|------------|
| C1 | Speedup: 2.54× vs 3.50× vs 4.71× | Different step counts, infrastructure, FA normalization | **2.55× under unified protocol** |
| C2 | Crossover: ~0.7B vs 1.0B vs 1.5–2B | Only 2 data points in prior estimates | **~0.7B confirmed (4-point parametric fit)** |
| C3 | Generation inflation: KV-cache vs prompt-format | Paper 1 misattributed to KV-cache expansion | **Prompt-format artifact (1.85× ratio)** |
| C4 | SECM-H: negative vs positive | Pre-scripted benchmarks bypass module selection | **ΔQ_content swing +0.242 (pre-scripted → agent-driven)** |
| C5 | Coverage: 33% vs 1% | Chain depth interaction, not KV-cache degradation | **Utilization gap (zero cache degradation across 32 rounds)** |
| C6 | AppLoop-PC: worse vs useful | Within-session harmful, cross-session beneficial | **Break-even: N≥14 (4B), N≥6 (0.8B)** |

### Key Insight: Utilization Gap vs Information Loss

Deep KV-cache recall validation across 32 injection rounds (4B, up to 6800 cache tokens) shows **zero observable degradation** — short-term recall stable at 0.90, long-term at 0.933. Paper 1's 1% coverage in Deep chain was NOT caused by KV-cache degradation but by the model's failure to actively utilize injected information during generation. This is a **utilization gap**, not information loss — SIG preserves information perfectly; the bottleneck is downstream retrieval during generation.

### Speedup vs Chain Depth

| Depth (steps) | 4B Speedup | 0.8B Speedup |
|--------------|-----------|-------------|
| 5 | 1.43× | 0.60× |
| 10 | 1.75× | 0.60× |
| 20 | 2.60× | 0.85× |
| 35 | 2.52× | 1.10× |
| 50 | **3.65×** | **1.49×** |

### Cross-Architecture Speedup

| Architecture | SIG Speedup | Verdict |
|-------------|-----------|---------|
| Qwen3.5-4B | **2.54×** | SIG faster |
| Nemotron-3-Nano-4B | **1.35×** | SIG faster |
| Gemma-4-E2B | **0.86×** | SIG slower |

## Core Findings Across the Program

### CO Architecture (Papers 1–3)

CO organizes edge intelligence into three layers:

1. **Meaning Compiler** — lightweight local model (0.8B–4B) that orchestrates task execution
2. **Injection Engine** — SIG runtime that maintains KV-cache continuity across tool calls
3. **Cognitive Module Ecosystem** — pluggable external resources (cloud teachers, perception APIs, skill libraries)

SIG's five-stage suspend-inject-resume cycle (Suspend → Resolve → Fetch → Inject → Resume) reduces prefill cost from quadratic to linear in injection size, independent of conversation length. Stabilization templates mitigate distribution shift in small models (malformed outputs: >30% → <2%).

### Performance Characterization (Papers 1–4)

| Metric | 0.8B Model | 4B Model | Gemma-4-E2B |
|--------|-----------|----------|-------------|
| Prefill token savings | 73–97% | 73–97% | — |
| End-to-end speedup (9 scenarios) | 2.38× | 2.70× | — |
| Deep-chain speedup (14–22 tools) | 4.96× | 5.26× | 3.20× |
| Kitchen benchmark (32 steps) | — | 2.54× | — |
| Batch-SIG vs AppLoop-PC | 4.65× | 4.65× | — |
| Per-token rate (SIG vs AppLoop) | ±2% | ±5% | — |
| Prefill crossover | ~1.5–2B params | — | — |
| KV-cache recall (long-term) | 0.93 (64 rounds) | — | 1.00 (32 rounds) |

Cross-architecture validation: SIG achieves consistent speedup across Qwen3.5 (2.38–2.70×) and Gemma-4-E2B (3.20×), confirming architecture-agnostic applicability including SWA + shared_kv_layers + GQA architectures. Batch-SIG generalizes robustly (4.24–6.82× across all architectures).

### Orthogonal Acceleration: CO + MTP (Paper 5)

After SIG eliminates the prefill bottleneck, generation time dominates 76% of remaining latency. Paper 5 investigates fusing CO's macro-level prefill elimination with MTP's micro-level generation compression.

**SIG Baseline (n-gram SpecDec, sequential verification)**

| Metric | 4B Model | 0.8B Model |
|--------|----------|------------|
| SIG speedup | **2.92×** | 1.06× |
| SIG+SpecDec (sequential) | 2.17× | — |
| Acceptance rate under SIG | **72.5%** | **88.1%** |
| Acceptance rate under AppLoop | 66.9% | 77.8% |
| Orthogonality ratio ρ (sequential) | **0.851** ✅ | 2.027 |

**Kitchen Benchmark: Native MTP Parallel Verification (EXP-7, n=5, 35 turns)**

| Condition | Wall-clock (s) | tok/s | Speedup |
|-----------|---------------|-------|---------|
| AppLoop | 142.89 | 107.6 | 1.00× |
| SIG | 40.79 | 104.1 | **3.50×** |
| AppLoop+MTP | 137.03 | 137.0 | 1.04× |
| SIG+MTP | 31.58 | 135.7 | **4.52×** |

**ρ = 1.239** (PASS, near-multiplicative with slight super-multiplicative tendency from SIG cache persistence boosting MTP acceptance)

**Native MTP SpecDec (parallel verification, Qwen3.5-4B)**

| Draft-n-max | Throughput | Acceptance Rate | Speedup |
|-------------|-----------|-----------------|---------|
| Baseline | 120.2 tok/s | — | 1.00× |
| n=1 | 149.2 tok/s | 87.1% | 1.24× |
| **n=2** | **152.5 tok/s** | **67.3%** | **1.27×** |
| n=3 | 141.7 tok/s | 55.7% | 1.18× |

**SIG + MTP Compound Acceleration (Multi-turn)**

| Turn | Throughput | Acceptance Rate |
|------|-----------|-----------------|
| 0 | 127.0 tok/s | 47.5% |
| 1 | 148.7 tok/s | 69.4% |
| **2** | **159.7 tok/s** | — |

Compound speedup increases with context length, validating the orthogonality hypothesis — CO's macro-level and MTP's micro-level accelerations compose on independent dimensions.

**Key engineering finding**: Three categories of toolchain obstacles for orthogonal acceleration:

| Category | Architecture | Barrier | Status |
|----------|-------------|---------|--------|
| Type 1: SWA partial deletion | Qwen3.5 hybrid | `kv_cache_seq_rm` fails on SWA circular buffer | Workaround: llama.cpp native MTP |
| Type 2: Python API crash | Qwen3.5 hybrid | `generate() + drafter` crashes with `llama_decode -1` | Workaround: llama-server HTTP |
| Type 3: Architecture unsupported | Gemma-4 assistant | `gemma4_assistant` not recognized by llama.cpp | BLOCKED: awaiting PR #23211/#23398 |

Gemma-4 SpecDec compatibility: 5/6 behavioral tests pass (main model fully functional); MTP draft model pairing blocked by Type 3 obstacle.

### State-Externalizing Cognitive Module Harnesses (Paper 8)

Paper 8 investigates whether the Meaning Compiler's implicit module management can be externalized to a structured harness (SECM-H). The key methodological contribution is the discovery that **benchmark design must test the capability being evaluated**: pre-scripted tool-call benchmarks bypass the very module management SECM-H externalizes, creating a ceiling effect that masks its value.

**State decomposition (EXP-1)**: 76.5% of module management functions (13/17) are fully externalizable, consuming up to 12% of context at 35 steps.

**Pre-scripted benchmarks (EXP-3/9/10)**: The SIG baseline outperforms all injection strategies. SECM-H-full degrades $Q_{content}$ by $\Delta = -0.141$ and increases latency 3.7×. The model's implicit KV-cache tracking is well-adapted to deterministic tasks.

**Agent-driven benchmarks (EXP-11/12)**: When the model autonomously selects tools, SECM-H demonstrates genuine value under noise:

| Metric | SIG | SECM-H (full) | SECM-H (selective) |
|--------|-----|--------------|-------------------|
| Tool accuracy (noisy) | 94.3% | **97.1%** | 91.4% |
| $Q_{content}$ (noisy) | 0.535 | **0.636** | 0.661 |
| $Q_{content}$ (clean) | 0.596 | 0.626 | **0.718** |

**Path A/B disentanglement (EXP-13)**: Forced-selection experiments confirm that SECM-H's state injection changes generation behavior (Path B: $\Delta Q_{content} = +0.050$), independent of tool selection improvement.

**Format interference (EXP-14)**: Natural-language state rendering achieves $\Delta Q_{content} = +0.113$ over template format, confirming that format interference is a significant component of the attention disruption mechanism.

**Core lesson**: SECM-H functions as a **generation stabilizer for autonomous agents under decision uncertainty**, not as a general-purpose cognitive architecture layer. Its value is specific to scenarios requiring dynamic module selection with varying reliability.

## Read the Papers

| Paper | Link |
|-------|------|
| 1 | [Cognitive Outsourcing with SIG](paper/Cognitive%20Outsourcing%20with%20Suspend-and-Inject%20Generation%20for%20Scalable%20Embodied%20Intelligence.md) |
| 2 | [Beyond the Injection Engine](paper/Beyond_the_Injection_Engine_A_Five_Dimensional_Analysis_of_CO-SIG.md) |
| 3 | [CO-SIG Architecture & Design Space](paper/CO_SIG_Architecture_Theory_Empirical_Design_Space_for_Scalable_Edge_Intelligence.md) |
| 4 | [SIG as Edge Runtime Primitive](paper/Suspend-and-Inject%20Generation%20as%20an%20Edge%20Inference%20Runtime%20Primitive%20for%20Long-Horizon%20Agent%20Tasks.md) |
| 5 | [Orthogonal Acceleration](paper/5_Orthogonal_Acceleration/paper.md) |
| 6 | [Convergent KVCache Architectures](paper/6_Convergent_KVCache_Architectures/paper.md) |
| 7 | [Disk-Backed KV-Cache Persistence](paper/7_DiskKVCache/paper.md) |
| 8 | [State-Externalizing Cognitive Module Harnesses](paper/8_State_Externalizing_Cognitive_Modules/paper.md) |
| 9 | [Consolidating the CO-SIG Research Program](paper/9_Consolidated_Experiment_Restructuring/paper.md) |
| Report | [Experiment Report](EXPERIMENT_REPORT.md) (Paper 8 data summary) |
| Survey (CN) | [综述：从认知外包到智能体推理引擎](paper/survey_SIG_CO_agent_frameworks_CN.md) |
| Survey (EN) | [Survey: From CO to Agent Inference Engines](paper/survey_SIG_CO_agent_frameworks_EN.md) |

## Repository Structure

```
├── paper/                              # All papers and surveys
│   ├── 5_Orthogonal_Acceleration/      # Paper 5: Orthogonal Acceleration
│   │   ├── paper.md                    # Full paper (Round 6)
│   │   └── figures/                    # All figures
│   ├── 6_Convergent_KVCache_Architectures/ # Paper 6: KFC unified framework
│   │   └── paper.md                    # Full paper
│   ├── 7_DiskKVCache/                  # Paper 7: Disk-backed KV-Cache persistence
│   │   ├── paper.md                    # Full paper
│   │   └── figures/                    # All figures
│   ├── 8_State_Externalizing_Cognitive_Modules/ # Paper 8: SECM-H
│   │   └── paper.md                    # Full paper
│   ├── 9_Consolidated_Experiment_Restructuring/ # Paper 9: Meta-empirical synthesis
│   │   ├── paper.md                    # Full paper
│   │   ├── experiments/                # Experiment scripts (EXP-1~8)
│   │   ├── results/                    # Experiment result JSON files
│   │   └── figures/                    # 10 publication-quality figures
│   ├── Cognitive Outsourcing...md      # Paper 1: CO architecture
│   ├── Beyond_the_Injection_Engine.md  # Paper 2: Theory (R1–R5)
│   ├── CO_SIG_Architecture...md        # Paper 3: Design space (R6–R14)
│   ├── Suspend-and-Inject...md         # Paper 4: Deployment (R15–R19)
│   ├── survey_SIG_CO_agent_frameworks_CN.md  # Survey (Chinese)
│   └── survey_SIG_CO_agent_frameworks_EN.md  # Survey (English)
├── data/                               # Raw experiment data (JSON)
│   ├── exp8_*/                         # Paper 8 data (not in git, see EXPERIMENT_REPORT.md)
│   ├── exp1_4B/                        # EXP-1: Orthogonality (4B)
│   ├── exp1_08B/                       # EXP-1: Orthogonality (0.8B)
│   ├── exp2_4B/                        # EXP-2: Injection-event depression
│   ├── exp3_final_4B/                  # EXP-3: SIG+SpecDec on 4B
│   ├── exp3_llamacpp_4B/              # EXP-3: llama.cpp results
│   ├── exp456_4B/                      # EXP-4/5/6 on 4B
│   ├── exp456_08B/                     # EXP-4/5/6 on 0.8B
│   ├── exp123_n5/                      # EXP-1/2/3 (n=5, n-gram SpecDec)
│   ├── exp123_real_specdec/            # EXP-1/2/3 (real SpecDec)
│   ├── cross_arch/                    # Cross-architecture results (Gemma-4)
│   ├── r2_degradation_deep_*.json     # R2 deep validation (64 rounds)
│   ├── r1_multiprompt_results.json    # R1 multi-prompt attention analysis
│   ├── exp_specdec_gemma4.json        # Gemma-4 SpecDec compatibility
│   ├── exp_mtp_4B_results.json         # MTP baseline (v1)
│   ├── exp_mtp_v2_results.json         # MTP + SIG+MTP (v2)
│   └── exp_mtp_kitchen_4B.json         # Kitchen 4-condition benchmark (EXP-7)
├── core/                               # CO/SIG inference engine
│   ├── compiler.py                     # KV-cache management (rm/cp/keep/shift)
│   ├── injection.py                    # Injection engine
│   ├── llamacpp_specdec.py             # Manual SpecDec + NgramDrafter
│   ├── qwen35_compat.py               # Qwen3.5 compatibility layer
│   ├── qwen35_specdec.py              # Qwen3.5 SpecDec wrapper
│   ├── mtp_heads.py                    # MTP head training (Track A)
│   ├── acceptance_tracker.py           # Acceptance rate tracking & recovery
│   ├── meaning_compiler.py             # KV-cache-aware meaning compiler
│   ├── harness/                        # SECM-H harness module (Paper 8)
│   │   ├── harness.py                  # Main SECMHarness class
│   │   ├── registry.py                 # R_t: module registry
│   │   ├── history.py                  # H_t: invocation history
│   │   ├── confidence.py               # C_t: reliability tracking
│   │   ├── dependency.py               # D_t: dependency graph
│   │   ├── pattern_cache.py            # P_t: cognitive patterns
│   │   ├── budget.py                   # B_t: budget management
│   │   └── renderer.py                 # State rendering
│   ├── quality.py                      # ContentQualityEvaluator (5-dim) + SemanticScorer
│   ├── metrics.py                      # Statistical + semantic similarity utilities
│   └── gpu.py                          # GPU monitoring
├── EXPERIMENT_REPORT.md                # Paper 8: all experiment data summaries (15 tables)
├── exp8_state_decomposition.py         # Paper 8: EXP-1 state decomposition audit
├── exp8_kitchen_benchmark.py           # Paper 8: EXP-3/4/5 pre-scripted kitchen
├── exp8_v2_channel_strategies.py       # Paper 8: EXP-9/10 channel strategies + noisy
├── exp8_v3_agent_driven.py             # Paper 8: EXP-11/12/13/14/15 agent-driven
├── r2_degradation_deep.py              # R2 deep validation (64 rounds, 3 models)
├── r1_attention_multiprompt.py         # R1 multi-prompt attention analysis
├── cross_arch_sig_bench.py             # Cross-architecture SIG benchmark
├── exp_specdec_gemma4.py              # Gemma-4 SpecDec compatibility tests
├── exp_mtp_gemma4_assistant.py         # Gemma-4 MTP experiment (BLOCKED)
├── exp_mtp_full_v2.py                  # Qwen3.5-4B MTP full experiment + Kitchen benchmark (--task kitchen)
├── r4_teacher_scan.py                  # R4 teacher-size scan
├── exp_mtp_v2.py                       # MTP benchmark + SIG+MTP (v2, main)
├── exp_mtp_native.py                   # Native MTP via llama-server
├── exp_mtp_full.py                     # Full 4-condition experiment (v1)
├── exp_sig_mtp.py                      # SIG+MTP combination test
├── run_exp123_real_specdec.py          # EXP-1/2/3 runner (real SpecDec)
├── co_benchmark.py                     # CO benchmark (Papers 1–3)
├── cross_arch_bench.py                 # Cross-architecture benchmark (Paper 3)
├── edge_agent_bench.py                 # EdgeAgent-Kitchen benchmark (Paper 4)
└── generate_figures.py                 # Figure generation script
```

## Quick Start

```bash
# Use conda environment sig_bench
conda activate sig_bench

# Download models
# Qwen3.5-4B-Q4_K_M.gguf → models/
# Qwen3.5-0.8B-Q4_K_M.gguf → models/
# gemma-4-E2B-it-Q4_K_M.gguf → models/  (cross-arch validation)
# MTP GGUF models → models/mtp/

# Run CO benchmark (Papers 1–3)
python co_benchmark.py --model models/Qwen3.5-4B-Q4_K_M.gguf

# Run Kitchen benchmark (Paper 4)
python edge_agent_bench.py --model models/Qwen3.5-4B-Q4_K_M.gguf

# Run MTP benchmark (Paper 5, native llama-server required)
python exp_mtp_v2.py

# Run Kitchen 4-condition benchmark (Paper 5, EXP-7)
python exp_mtp_full_v2.py --task kitchen --model models/Qwen3.5-4B-Q4_K_M.gguf --mtp-model models/mtp/Qwen3.5-4B-Q4_K_M.gguf --n-runs 5

# Run cross-architecture benchmark (Paper 3)
python cross_arch_bench.py

# Run R2 deep validation (64 rounds)
python r2_degradation_deep.py --model models/Qwen3.5-0.8B-Q4_K_M.gguf --max-rounds 64

# Run Gemma-4 cross-architecture SIG benchmark
python cross_arch_sig_bench.py --model models/gemma-4-E2B-it-Q4_K_M.gguf

# Run Paper 8 experiments
python exp8_state_decomposition.py                                    # EXP-1: state decomposition audit
python exp8_kitchen_benchmark.py --n-runs 5                           # EXP-3/4/5: pre-scripted kitchen
python exp8_v2_channel_strategies.py --task all --n-runs 3            # EXP-9/10: channel strategies
python exp8_v3_agent_driven.py --task all --n-runs 3 --n-steps 35    # EXP-11/12/13/14: agent-driven
python exp8_v3_agent_driven.py --task exp15 --n-runs 3 --n-steps 35  # EXP-15: 0.8B agent-driven

# Run Paper 9 consolidated experiments
cd paper/9_Consolidated_Experiment_Restructuring/experiments
python run_all.py --list                 # List all experiments
python run_all.py --dry-run              # Dry-run plan
python run_all.py --only exp1            # Run single experiment
python run_all.py                        # Run all experiments (~270 runs)
python analyze_results.py --tables       # Print reconciliation tables
python generate_figures.py               # Generate all figures
```

## Citation

```bibtex
@article{co-sig-2026,
  title={Cognitive Outsourcing with Suspend-and-Inject Generation
         for Scalable Embodied Intelligence},
  author={SIG/CO Research Program},
  year={2026},
  note={Papers 1--9 in the SIG/CO Research Program}
}

@article{co-sig-survey-2026,
  title={From Cognitive Outsourcing to Agent Inference Engines:
         Core Ideas of Suspend-and-Inject Generation and Their
         Implications for Modern Agent Frameworks},
  author={SIG/CO Research Program},
  year={2026},
  note={Survey paper (CN/EN)}
}
```

## License

This research is part of the SIG/CO Research Program. See individual files for specific licenses.
