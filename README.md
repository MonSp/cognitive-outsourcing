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

## Survey Papers

| Paper | Language | Focus |
|-------|----------|-------|
| [综述：从认知外包到智能体推理引擎](paper/survey_SIG_CO_agent_frameworks_CN.md) | 中文 | SIG 核心理念 + CO 认知架构 + 对现代智能体框架的启发 |
| [Survey: From Cognitive Outsourcing to Agent Inference Engines](paper/survey_SIG_CO_agent_frameworks_EN.md) | English | SIG core ideas + CO cognitive architecture + implications for modern agent frameworks |

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

## Read the Papers

| Paper | Link |
|-------|------|
| 1 | [Cognitive Outsourcing with SIG](paper/Cognitive%20Outsourcing%20with%20Suspend-and-Inject%20Generation%20for%20Scalable%20Embodied%20Intelligence.md) |
| 2 | [Beyond the Injection Engine](paper/Beyond_the_Injection_Engine_A_Five_Dimensional_Analysis_of_CO-SIG.md) |
| 3 | [CO-SIG Architecture & Design Space](paper/CO_SIG_Architecture_Theory_Empirical_Design_Space_for_Scalable_Edge_Intelligence.md) |
| 4 | [SIG as Edge Runtime Primitive](paper/Suspend-and-Inject%20Generation%20as%20an%20Edge%20Inference%20Runtime%20Primitive%20for%20Long-Horizon%20Agent%20Tasks.md) |
| 5 | [Orthogonal Acceleration](paper/5_Orthogonal_Acceleration/paper.md) |
| 6 | [Convergent KVCache Architectures](paper/6_Convergent_KVCache_Architectures/paper.md) |
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
│   ├── Cognitive Outsourcing...md      # Paper 1: CO architecture
│   ├── Beyond_the_Injection_Engine.md  # Paper 2: Theory (R1–R5)
│   ├── CO_SIG_Architecture...md        # Paper 3: Design space (R6–R14)
│   ├── Suspend-and-Inject...md         # Paper 4: Deployment (R15–R19)
│   ├── survey_SIG_CO_agent_frameworks_CN.md  # Survey (Chinese)
│   └── survey_SIG_CO_agent_frameworks_EN.md  # Survey (English)
├── data/                               # Raw experiment data (JSON)
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
│   ├── quality.py                      # Hybrid semantic quality scorer
│   ├── metrics.py                      # Statistical + semantic similarity utilities
│   └── gpu.py                          # GPU monitoring
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
```

## Citation

```bibtex
@article{co-sig-2026,
  title={Cognitive Outsourcing with Suspend-and-Inject Generation
         for Scalable Embodied Intelligence},
  author={SIG/CO Research Program},
  year={2026},
  note={Papers 1--6 in the SIG/CO Research Program}
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
