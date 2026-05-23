# Cognitive Outsourcing with Suspend-and-Inject Generation

Edge-AI architecture enabling lightweight on-device language models (as small as 0.8B) to dynamically access external cognitive resources while preserving continuous attention state through **Suspend-and-Inject Generation (SIG)**.

---

## Key Results

| Metric | Qwen3.5-0.8B | Qwen3.5-4B |
|--------|-------------|------------|
| Prefill token savings | 73–93% | 78–97% |
| Prefill time savings | 81% | 87% |
| End-to-end speedup (teacher-precomputed) | **2.38×** | **2.70×** |
| Peak speedup (deep chain, 14 tools) | **4.96×** | **5.26×** |
| Per-token generation rate difference | < 2% | < 2% |
| GPU VRAM overhead (SIG vs AppLoop) | +0.1 GB | +0.1 GB |

> The speedup originates from **prefill savings**, not from faster per-token generation. See [BENCHMARK_RESULTS.md](BENCHMARK_RESULTS.md) for the full ten-chapter report.

---

## Five-Dimensional Analysis

| Dimension | Status | Key Finding |
|-----------|--------|-------------|
| **R1** Information Theory | ✅ Direct measurement | Attention head agreement 0.25→0.43 across layers (Qwen2.5-0.5B); early layers most sensitive |
| **R2** Cache Degradation | ✅ Direct measurement | 6–10 round weather recall, no degradation (stable 0.50–1.00) |
| **R3** Beyond Transformer | 🔧 Simulation + calibration | Transformer > xLSTM > RWKV > SSM/Mamba projected; empirically parameterized |
| **R4** Teacher-Student Gap | ✅ First measurement | CoT amplification +0.80, SIG amplification +0.59, teacher margin 0.72 |
| **R5** Privacy Guarantees | 🧪 Concept demo | PII redaction + intent-only outsourcing across 4 query types |

---

## Quick Start

### Prerequisites

```bash
# Core dependencies
pip install llama-cpp-python numpy

# For R1 attention analysis
pip install torch transformers modelscope
```

Download quantized GGUF models to `./models/`:
```
models/Qwen3.5-0.8B-Q4_K_M.gguf
models/Qwen3.5-4B-Q4_K_M.gguf
```

### Three Entry Points

```bash
# ── CO benchmarks + R2/R4/R5 ──
python co_benchmark.py --model models/Qwen3.5-0.8B-Q4_K_M.gguf --n-gpu-layers 99 --task baseline
python co_benchmark.py --model models/Qwen3.5-0.8B-Q4_K_M.gguf --n-gpu-layers 99 --task r2 --r2-n-cities 8 --r2-probe-interval 2
python co_benchmark.py --task r4                               # no model required
python co_benchmark.py --task r5                               # no model required
python co_benchmark.py --task r3                               # numpy simulation

# ── SIG benchmarks + R1/R3 ──
python sig_benchmark.py --model models/Qwen3.5-0.8B-Q4_K_M.gguf --n-gpu-layers 99 --task baseline
python sig_benchmark.py --task r1 --r1-model-id Qwen/Qwen2.5-0.5B  # HuggingFace model
python sig_benchmark.py --task r3

# ── Universal Transformer testing engine ──
python transformer_bench.py --task r1 --model-id Qwen/Qwen2.5-0.5B
python transformer_bench.py --task r3
python transformer_bench.py --task r3-empirical
python transformer_bench.py --task all --output report.json
```

### Task Matrix

| Script | `--task` | Needs GGUF? | Needs HF? | Description |
|--------|----------|-------------|-----------|-------------|
| `co_benchmark.py` | `baseline` | ✅ | | CO 9-scenario AppLoop vs SIG |
| `co_benchmark.py` | `r2` | ✅ | | R2 KV-cache degradation (weather recall) |
| `co_benchmark.py` | `r3` | | | R3 cross-architecture simulation |
| `co_benchmark.py` | `r4` | | | R4 teacher-student capability gap |
| `co_benchmark.py` | `r5` | | | R5 privacy PII anonymization demo |
| `sig_benchmark.py` | `baseline` | ✅ | | SIG 9-scenario autonomous mode |
| `sig_benchmark.py` | `r1` | | ✅ | R1 attention distribution analysis |
| `sig_benchmark.py` | `r3` | | | R3 simulation + empirical calibration |
| `transformer_bench.py` | `r1` | | ✅ | Standalone R1 attention analysis |
| `transformer_bench.py` | `r3` | | | Standalone R3 cross-architecture |
| `transformer_bench.py` | `r3-empirical` | | | CO benchmark parameterization |
| `transformer_bench.py` | `all` | | ✅ | All tasks + JSON export |

---

## Architecture

```
User Query → Meaning Compiler (local 0.8B/4B)
    ↓
Injection Engine (SIG — KV-cache continuity)
    ├── Suspend: pause autoregressive decoding
    ├── Inject: append tool results / CoT plans to KV-cache
    └── Resume: continue generation without re-prefill
    ↓
Cognitive Module Ecosystem
    ├── Cloud teachers (GPT-4, Claude, DeepSeek)
    ├── Tool APIs (weather, search, flight, code)
    └── Local skill libraries
```

---

## Repository Structure

```
cognitive-outsourcing/
├── core/                       # Runtime engine
│   ├── compiler.py             # MeaningCompiler (llama.cpp wrapper)
│   ├── injection.py            # InjectionEngine (SIG primitive)
│   ├── prompts.py              # Prompt templates
│   ├── tools.py                # Mock tool implementations
│   ├── metrics.py              # Tool accuracy, timing
│   ├── gpu.py                  # GPU memory monitor
│   └── text_utils.py           # Text processing utilities
├── co_benchmark.py             # CO entry point (baseline + R2/R3/R4/R5)
├── sig_benchmark.py            # SIG entry point (baseline + R1/R3)
├── transformer_bench.py        # Universal Transformer testing engine
├── gen_plans.py                # Teacher CoT plan generator
├── co_benchmark_plans.json     # Pre-computed teacher plans
├── co_benchmark_prompts.json   # Scenario prompt definitions
├── BENCHMARK_RESULTS.md        # Complete benchmark report (10 chapters)
├── paper/                      # Paper and supporting documents
│   ├── Beyond_the_Injection_Engine_*.md
│   ├── Cognitive_Outsourcing_*.md
│   └── *.pdf
├── README.md
├── TODO.md                     # Research roadmap (R1–R14)
├── CODE_WIKI.md                # Developer documentation
└── LICENSE
```

---

## Paper

- **Beyond the Injection Engine: A Five-Dimensional Analysis of CO+SIG** — `paper/Beyond_the_Injection_Engine_A_Five_Dimensional_Analysis_of_CO-SIG.md`
- **Original paper**: *Cognitive Outsourcing with Suspend-and-Inject Generation for Scalable Embodied Intelligence* — `paper/`

---

## License

MIT
