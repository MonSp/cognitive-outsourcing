# Cognitive Outsourcing (CO) with Suspend-and-Inject Generation

> **SIG/CO Research Program** — [GitHub](https://github.com/sig-co)

Cognitive Outsourcing (CO) is an edge-AI architecture that empowers lightweight on-device language models (0.8B–4B) to orchestrate complex physical tasks by dynamically accessing external cognitive resources—local sensors, skill libraries, cloud-scale LLM "teachers"—while preserving continuous attention state through the Suspend-and-Inject Generation (SIG) primitive. SIG enables a running model to pause decoding, invoke external modules, and seamlessly absorb their responses into the KV cache without costly re-encoding, eliminating the quadratic prefill overhead of traditional tool-calling loops.

**Target regime**: single-user, single-instance edge inference (robots, smartphones, embedded systems) where prefill dominates wall-clock time and cloud-scale serving optimizations (FlashAttention, RadixAttention) are inapplicable.

## Research Program Progress

| Paper | Title | Focus | Key Results | Status |
|-------|-------|-------|-------------|--------|
| 1 | [Cognitive Outsourcing with SIG](paper/Cognitive%20Outsourcing%20with%20Suspend-and-Inject%20Generation%20for%20Scalable%20Embodied%20Intelligence.md) | CO architecture, SIG protocol | 84–96% prefill savings, 1.57× speedup, 3× information coverage | ✅ |
| 2 | [Beyond the Injection Engine](paper/Beyond_the_Injection_Engine_A_Five_Dimensional_Analysis_of_CO-SIG.md) | Theory (R1–R5): attention shift, cache lifecycle, capability gap | Per-token rate equivalence ±2%, attention head agreement 0.25→0.43, no cache degradation at 6–10 rounds | ✅ |
| 3 | [CO-SIG Architecture & Design Space](paper/CO_SIG_Architecture_Theory_Empirical_Design_Space_for_Scalable_Edge_Intelligence.md) | Boundaries (R6–R14), cross-architecture, Batch-SIG | 2.79–5.26× deep-chain speedup, Batch-SIG 4.24–6.82× (architecture-independent), SIG Decision Framework | ✅ |
| 4 | [SIG as Edge Runtime Primitive](paper/Suspend-and-Inject%20Generation%20as%20an%20Edge%20Inference%20Runtime%20Primitive%20for%20Long-Horizon%20Agent%20Tasks.md) | Deployment (R15–R19), Kitchen benchmark | 2.54× wall-clock speedup, prefill crossover ~1.5–2B, per-token rate 108 vs 103 tok/s | ✅ |
| **5** | [**Orthogonal Acceleration**](paper/5_Orthogonal_Acceleration/paper.md) | **CO + MTP compound acceleration** | **SIG 2.92×, native MTP 1.27×, SIG+MTP compound validated** | **🔄 Round 5** |

## Core Findings Across the Program

### CO Architecture (Papers 1–3)

CO organizes edge intelligence into three layers:

1. **Meaning Compiler** — lightweight local model (0.8B–4B) that orchestrates task execution
2. **Injection Engine** — SIG runtime that maintains KV-cache continuity across tool calls
3. **Cognitive Module Ecosystem** — pluggable external resources (cloud teachers, perception APIs, skill libraries)

SIG's five-stage suspend-inject-resume cycle (Suspend → Resolve → Fetch → Inject → Resume) reduces prefill cost from quadratic to linear in injection size, independent of conversation length. Stabilization templates mitigate distribution shift in small models (malformed outputs: >30% → <2%).

### Performance Characterization (Papers 1–4)

| Metric | 0.8B Model | 4B Model |
|--------|-----------|----------|
| Prefill token savings | 73–97% | 73–97% |
| End-to-end speedup (9 scenarios) | 2.38× | 2.70× |
| Deep-chain speedup (14–22 tools) | 4.96× | 5.26× |
| Kitchen benchmark (32 steps) | — | 2.54× |
| Batch-SIG vs AppLoop-PC | 4.65× | 4.65× |
| Per-token rate (SIG vs AppLoop) | ±2% | ±5% |
| Prefill crossover | ~1.5–2B params | — |

Cross-architecture validation (Paper 3): raw prefill savings are Qwen-family-specific (0.98–1.12× on Nemotron/Gemma), but Batch-SIG generalizes robustly (4.24–6.82× across all architectures).

### Orthogonal Acceleration: CO + MTP (Paper 5)

After SIG eliminates the prefill bottleneck, generation time dominates 76% of remaining latency. Paper 5 investigates fusing CO's macro-level prefill elimination with MTP's micro-level generation compression.

**SIG Baseline (n-gram SpecDec, sequential verification)**

| Metric | 4B Model | 0.8B Model |
|--------|----------|------------|
| SIG speedup | **2.92×** | 1.06× |
| SIG+SpecDec (sequential) | 2.17× | — |
| Acceptance rate under SIG | **72.5%** | **88.1%** |
| Acceptance rate under AppLoop | 66.9% | 77.8% |
| Orthogonality ratio ρ | **0.851** ✅ | 2.027 |

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

**Key engineering finding**: llama-cpp-python's SpecDec is incompatible with Qwen3.5's hybrid attention (three independent failures), but llama.cpp's native MTP (`--spec-type draft-mtp`, b9415+) bypasses the barrier at the C++ level.

## Read the Papers

| Paper | Link |
|-------|------|
| 1 | [Cognitive Outsourcing with SIG](paper/Cognitive%20Outsourcing%20with%20Suspend-and-Inject%20Generation%20for%20Scalable%20Embodied%20Intelligence.md) |
| 2 | [Beyond the Injection Engine](paper/Beyond_the_Injection_Engine_A_Five_Dimensional_Analysis_of_CO-SIG.md) |
| 3 | [CO-SIG Architecture & Design Space](paper/CO_SIG_Architecture_Theory_Empirical_Design_Space_for_Scalable_Edge_Intelligence.md) |
| 4 | [SIG as Edge Runtime Primitive](paper/Suspend-and-Inject%20Generation%20as%20an%20Edge%20Inference%20Runtime%20Primitive%20for%20Long-Horizon%20Agent%20Tasks.md) |
| 5 | [Orthogonal Acceleration](paper/5_Orthogonal_Acceleration/paper.md) |

## Repository Structure

```
├── paper/                              # All five papers
│   ├── 5_Orthogonal_Acceleration/      # Paper 5 (latest)
│   │   ├── paper.md                    # Full paper (Round 5)
│   │   └── figures/                    # All figures
│   ├── Cognitive Outsourcing...md      # Paper 1: CO architecture
│   ├── Beyond_the_Injection_Engine.md  # Paper 2: Theory (R1–R5)
│   ├── CO_SIG_Architecture...md        # Paper 3: Design space (R6–R14)
│   └── Suspend-and-Inject...md         # Paper 4: Deployment (R15–R19)
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
│   ├── exp_mtp_4B_results.json         # MTP baseline (v1)
│   └── exp_mtp_v2_results.json         # MTP + SIG+MTP (v2)
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
│   ├── metrics.py                      # Statistical utilities
│   └── gpu.py                          # GPU monitoring
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
# MTP GGUF models → models/mtp/

# Run CO benchmark (Papers 1–3)
python co_benchmark.py --model models/Qwen3.5-4B-Q4_K_M.gguf

# Run Kitchen benchmark (Paper 4)
python edge_agent_bench.py --model models/Qwen3.5-4B-Q4_K_M.gguf

# Run MTP benchmark (Paper 5, native llama-server required)
python exp_mtp_v2.py

# Run cross-architecture benchmark (Paper 3)
python cross_arch_bench.py
```

## Citation

```bibtex
@article{co-sig-2026,
  title={Cognitive Outsourcing with Suspend-and-Inject Generation
         for Scalable Embodied Intelligence},
  author={SIG/CO Research Program},
  year={2026},
  note={Papers 1--5 in the SIG/CO Research Program}
}
```

## License

This research is part of the SIG/CO Research Program. See individual files for specific licenses.
