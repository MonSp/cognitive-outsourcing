# Orthogonal Acceleration: SIG + Speculative Decoding for Edge Agent Inference

> **Paper 5 in the [SIG/CO Research Program](https://github.com/sig-co)**

## Key Results

| Metric | 4B Model | 0.8B Model |
|--------|----------|------------|
| SIG speedup (prefill elimination) | **2.91x** | 1.06x |
| SIG+SpecDec speedup | **3.20x** | 0.83x |
| N-gram acceptance rate (n=3) | **81.6%** | **90.8%** |
| Orthogonality ratio ρ | 0.365* | **0.912** ✅ |
| SIG-vs-AppLoop crossover | ~0.7B | — |

*\*4B ρ reflects generate()-path overhead, not mechanism interference.*

## What This Paper Shows

1. **SIG achieves 2.91x wall-clock speedup** on 4B models by eliminating prefill redundancy across tool-call rounds
2. **SIG + n-gram SpecDec achieves 3.20x** — a 10% improvement over SIG-only
3. **N-gram drafter acceptance rates are 81–91%** on the EdgeAgent-Kitchen benchmark
4. **The orthogonality framework is validated** (ρ = 0.912 on 0.8B, PASS)
5. **Post-injection acceptance rate depression** of −36% at t=2 confirms the injection-event signal
6. **Qwen3.5's hybrid attention architecture** (`full_attention_interval=4`) blocks llama.cpp's native `generate()` + drafter path

## Read the Paper

📄 **[Full Paper (Markdown)](paper/5_Orthogonal_Acceleration/paper.md)**

## Repository Structure

```
├── README.md                          # This file
├── paper/
│   ├── 5_Orthogonal_Acceleration/     # This paper (Paper 5)
│   │   ├── paper.md                   # Full paper (GitHub-rendered)
│   │   └── figures/                   # All figures
│   ├── Cognitive Outsourcing...md     # Paper 1
│   ├── Beyond_the_Injection_Engine.md # Paper 2
│   ├── CO_SIG_Architecture...md       # Paper 3
│   └── Suspend-and-Inject...md        # Paper 4
├── data/                              # Raw experiment data (JSON)
│   ├── exp3_final_4B/                 # EXP-3: SIG+SpecDec on 4B
│   ├── exp3_llamacpp_4B/              # EXP-3: llama.cpp results
│   ├── exp456_4B/                     # EXP-4/5/6 on 4B
│   └── exp456_08B/                    # EXP-4/5/6 on 0.8B
├── core/                              # SIG inference engine
│   ├── meaning_compiler.py            # KV-cache-aware compiler
│   ├── injection.py                   # Injection engine
│   ├── llamacpp_specdec.py            # Manual SpecDec (eval+sample)
│   └── qwen35_compat.py              # Qwen3.5 compatibility layer
├── edge_agent_bench.py                # EdgeAgent-Kitchen benchmark
├── run_exp3_patched.py                # EXP-3 runner
├── run_exp456.py                      # EXP-4/5/6 runner
└── generate_figures.py                # Figure generation script
```

## Quick Start

```bash
# Install dependencies
pip install llama-cpp-python numpy matplotlib

# Download models
# Qwen3.5-4B-Q4_K_M.gguf → models/
# Qwen3.5-0.8B-Q4_K_M.gguf → models/

# Run EXP-3 (SIG + SpecDec)
python run_exp3_patched.py --model models/Qwen3.5-4B-Q4_K_M.gguf --n-runs 3

# Run EXP-4/5/6
python run_exp456.py --model models/Qwen3.5-4B-Q4_K_M.gguf --exp all

# Generate figures
python generate_figures.py
```

## The SIG/CO Research Program

| Paper | Title | Status |
|-------|-------|--------|
| 1 | Cognitive Outsourcing with SIG for Scalable Embodied Intelligence | ✅ |
| 2 | Beyond the Injection Engine: A Five-Dimensional Analysis of CO-SIG | ✅ |
| 3 | CO-SIG Architecture, Theory, and Empirical Design Space | ✅ |
| 4 | SIG as an Edge Inference Runtime Primitive for Long-Horizon Agent Tasks | ✅ |
| **5** | **Orthogonal Acceleration: SIG + Speculative Decoding** | **✅ This paper** |

## Citation

```bibtex
@article{sig-mtp-2026,
  title={Orthogonal Acceleration: Fusing Multi-Token Prediction with 
         Suspend-and-Inject Generation for Compound Edge Agent Inference},
  author={SIG/CO Research Program},
  year={2026},
  note={Paper 5 in the SIG/CO Research Program}
}
```

## License

This research is part of the SIG/CO Research Program. See individual files for specific licenses.
