# Cross-Architecture CO/SIG Validation Test Report

**Date:** 2026-05-27
**Environment:** Conda `sig_bench` (Python 3.11.15, torch 2.6.0+cu124, llama-cpp-python CUDA)
**Hardware:** NVIDIA GeForce RTX 4070 SUPER (12,282 MB VRAM)
**Baseline Model:** Qwen3.5-4B-Q4_K_M (dense attention, 4B)

---

## 1. Unit Test Suite

| Test File | Tests | Result |
|-----------|-------|--------|
| `test_metrics.py` | 5 | ✅ All pass |
| `test_quality.py` | 15 | ✅ All pass |
| `test_tools.py` | 12 | ✅ All pass |
| **Total** | **32** | **✅ 32/32 PASS** |

---

## 2. Cross-Architecture Models

| Model | Architecture | Size | Key Features |
|-------|-------------|------|-------------|
| `nvidia_Nemotron-3-Nano-4B-Q4_K_M` | **Hybrid Mamba+attention** | 4B | ~3.2B Mamba + ~0.8B attention; SSM recurrence + selective KV-cache |
| `gemma-4-E2B-it-Q4_K_M` | Gemma 4 (GQA) | 2B | GeGLU, pre-norm, SWA cache. Near 1.5-2B crossover boundary |

---

## 3. Diagnostic Experiments (Qwen 4B Baseline)

### 3.1 Diagnostic Q1 — Kitchen Task Quality (n=5, 32 steps)

| Metric | SIG | AppLoop |
|--------|-----|---------|
| Wall-Clock (s) | 18.11±0.33 | 62.02±0.80 |
| Speedup | **3.43×** | — |
| Gen Tokens | 670 | 1892 |
| Composite Quality (hybrid) | 0.449 | 0.567 |
| Quality Δ (SIG−App) | **−0.119** | — |
| Allergen Awareness | **1.00** | 0.50 |
| Recipe Mention (semantic) | 0.00 | 0.42 |
| Tool Execution Rate | 0.97 | 0.97 |

### 3.2 Diagnostic Q6 — Batch-SIG Sequential Dependency (n=10)

**Independent Chain (batch-compatible):**

| Mode | Wall-Clock | Gen Calls |
|------|-----------|-----------|
| Per-Step SIG | 2.923±0.136s | 8 |
| Batch-SIG (bs=4) | 1.130±0.018s | 2 |
| Batch-SIG (bs=8) | 0.673±0.009s | 1 |

**Sequential Chain (NOT batch-compatible):**

| Mode | Wall-Clock | Note |
|------|-----------|------|
| Per-Step SIG | 2.669±0.031s | Structurally constrained |
| Batch-SIG (bs=4) | 1.191±0.034s | Gen calls reduced but tools still sequential |
| Batch-SIG (bs=8) | 0.702±0.028s | Gen calls=1, tools still sequential |

**Key:** Even on sequential chains, Batch-SIG still reduces gen calls — but tools MUST execute in dependency order.

### 3.3 Diagnostic Q7 — KV-Cache Probing (n=10)

| Condition | Entity Completion Match |
|-----------|------------------------|
| SIG (injected) | **100%** |
| SIG (NOT injected) | 0% |
| AppLoop (explicit) | 0% (model emitted `<think>` tag) |
| AppLoop (NOT in ctx) | 0% |

**Multi-Entity Enumeration:** Both SIG and AppLoop found 0/5 entities.
**Middle-Position Recall:** SIG 0/10, APP 0/10.

**Key Finding:** KV-cache injection supports **single-entity completion** perfectly (100%), proving information IS accessible via attention. The failure mode is **multi-entity exhaustive enumeration**, not single-entity retrieval.

---

## 4. Cross-Architecture Results

### 4.1 CO Prefill Savings Benchmark (SIG vs AppLoop)

| Chain Depth | Qwen 4B (ref) | Gemma 4 (2B) | Nemotron (4B hybrid) |
|-------------|--------------|-------------|---------------------|
| 4 tools | 2.38× | **1.13×** | **1.03×** |
| 8 tools | 2.70× | **0.92×** | **0.98×** |
| 12 tools | 2.70× | **1.31×** | **0.92×** |
| **Avg** | **~2.59×** | **1.12×** | **0.98×** |

> **Critical:** SIG's prefill savings do NOT generalize uniformly. On Nemotron (hybrid Mamba+attention), SIG provides **zero net speedup** (avg 0.98×). On Gemma 4, speedup is marginal (avg 1.12×). This confirms the reviewer's concern: the 2.38-2.70× speedups are Qwen-family-specific.

### 4.2 R13 Batch Injection (n=5, 8 fragments)

| Mode | Qwen 4B | Gemma 4 (2B) | Nemotron (4B hybrid) |
|------|---------|-------------|---------------------|
| Per-Step SIG | 2.673s | 0.467s | 0.849s |
| Batch-SIG (bs=8) | 0.283s | 0.177s | 0.158s |
| Speedup (bs=8 vs per-step) | **9.45×** | **2.64×** | **5.38×** |
| AppLoop-PC | 1.315s | 1.207s | 0.670s |
| Batch-SIG/AppLoop-PC | **4.65×** | **6.82×** | **4.24×** |

> **Key:** Batch-injection advantage **generalizes across all three architectures**. Even on Nemotron where raw SIG fails, Batch-SIG achieves 5.38× vs per-step and 4.24× vs AppLoop-PC. The generation-call reduction mechanism is architecture-independent.

### 4.3 Kitchen Quality (n=5, 30 steps)

| Metric | Qwen 4B (ref) | Gemma 4 (2B) | Nemotron (4B hybrid) |
|--------|--------------|-------------|---------------------|
| SIG Wall-Clock | 18.11s | 0.962±0.162s | 2.853±0.475s |
| AppLoop Wall-Clock | 62.02s | 1.826±0.280s | 2.145±0.383s |
| **Speedup** | **3.43×** | **1.90×** | **0.75×** |
| SIG Quality | 0.449 | 0.399±0.043 | 0.325±0.000 |
| AppLoop Quality | 0.567 | 0.325±0.000 | 0.325±0.000 |
| **Quality Δ** | **−0.119** | **+0.074** ★ | **±0.000** |
| SIG Recipe Mention | 0.00 | 0.268 | 0.000 |
| AppLoop Recipe Mention | 0.42 | 0.000 | 0.000 |

> **★ Notable:** On Gemma 4, SIG actually **outperforms** AppLoop in composite quality (+0.074). This is the first model where SIG's quality is strictly better. On Nemotron, quality is identical but SIG is slower (0.75×).

---

## 5. Summary: Architectural Generalization Matrix

| Capability | Qwen 3.5 (4B dense) | Gemma 4 (2B GQA) | Nemotron (4B hybrid) |
|-----------|--------------------|--------------------|-----------------------|
| **Prefill savings speedup** | ✅ 2.38−2.70× | ⚠️ 0.92−1.31× (marginal) | ❌ 0.92−1.03× (none) |
| **Batch-SIG advantage** | ✅ 9.45× / 4.65× | ✅ 2.64× / 6.82× | ✅ 5.38× / 4.24× |
| **Kitchen speedup** | ✅ 3.43× | ⚠️ 1.90× | ❌ 0.75× |
| **Kitchen quality vs AppLoop** | ⚠️ −0.119 | ✅ +0.074 | ➖ ±0.000 |
| **KV-cache single-entity recall** | ✅ 100% | — | — |
| **KV-cache multi-entity recall** | ❌ 0% | — | — |

**Legend:** ✅ = advantage | ⚠️ = marginal/conditional | ❌ = no advantage/penalty | ➖ = neutral

---

## 6. Conclusions

### 6.1 What Generalizes

1. **Batch-SIG is architecture-independent.** The generation-call reduction mechanism (accumulate tool results → single `generate()`) provides speedup across dense attention (Qwen), GQA (Gemma 4), and hybrid SSM+attention (Nemotron). This is the strongest cross-architecture finding.

2. **KV-cache single-entity recall is preserved** across injection modes — a single recipe name injected into KV-cache can be reliably retrieved at 100% rate.

### 6.2 What Does NOT Generalize

1. **Raw prefills savings are Qwen-specific.** The 2.38−2.70× CO baseline speedup collapses to 0.98−1.12× on non-Qwen architectures. SIG does not accelerate inference on Nemotron's hybrid Mamba+attention.

2. **The quality-speedup trade-off is architecture-dependent.** Qwen: −0.119 quality for +3.43× speedup. Gemma 4: **+0.074** quality for +1.90× speedup (strict improvement). Nemotron: 0.000 quality for 0.75× (strict regression).

### 6.3 Recommendations

1. **The paper's speedup claims must be explicitly scoped to the Qwen model family** until cross-family validation is extended to more architectures at matched parameter counts.

2. **Batch-SIG should be the recommended deployment strategy across ALL architectures** — it's the one advantage that generalizes robustly.

3. **Nemotron provides the strongest counter-evidence** to architectural generality. Future work should investigate why hybrid SSM+attention models fail to benefit from KV-cache injection, potentially due to Mamba layers' fixed-size state compression.

4. **Gemma 4's quality inversion** (SIG > AppLoop on quality) warrants deeper investigation — the pre-norm architecture may interact favorably with KV-cache continuity.

---

## 7. Reproduction Commands

```powershell
# Set environment
$env:PATH = "C:\Users\钱光华\AppData\Roaming\Python\Python311\site-packages\torch\lib;C:\Users\钱光华\AppData\Roaming\Python\Python311\site-packages\llama_cpp\lib;" + $env:PATH

# Unit tests
python -m unittest discover -s tests -v

# Diagnostics (Qwen baseline)
python diagnostic_bench.py --task quality_kitchen --model models/Qwen3.5-4B-Q4_K_M.gguf --n-gpu-layers 99 --quality-runs 5
python diagnostic_bench.py --task seq_dependency --model models/Qwen3.5-4B-Q4_K_M.gguf --n-gpu-layers 99 --profile-runs 10
python diagnostic_bench.py --task kv_probe --model models/Qwen3.5-4B-Q4_K_M.gguf --n-gpu-layers 99 --profile-runs 10

# Cross-architecture
python cross_arch_bench.py --model models/gemma-4-E2B-it-Q4_K_M.gguf --task all --n-gpu-layers 99
python cross_arch_bench.py --model models/nvidia_Nemotron-3-Nano-4B-Q4_K_M.gguf --task all --n-gpu-layers 99
```

---

*Report generated automatically by the CO/SIG cross-architecture validation pipeline.*
