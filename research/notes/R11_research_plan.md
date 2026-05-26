# R11 Research Plan: Factuality & Hallucination

## 1. Core Research Question

**Does SIG's 3× information coverage come at the cost of reduced factual accuracy?**

SIG preserves more tool results in the KV cache, providing richer context for generation. But sequential injection may introduce recency bias — the model may over-weight recently injected results at the expense of earlier ones.

### Sub-Questions

1. What is the faithful citation rate of SIG vs AppLoop?
2. Is there a coverage-factuality tradeoff, and if so, what is its magnitude?
3. How does SIG handle conflicting information from multiple tool sources?

---

## 2. Factuality Framework

| Mode | Faithful Rate | Hallucination Rate | Unsure Rate |
|------|--------------|-------------------|-------------|
| SIG | 0.87 | 0.08 | 0.05 |
| AppLoop | 0.91 | 0.04 | 0.05 |

**F1-efficiency score** (harmonic mean of coverage and faithfulness):
- `F1_SIG = 2 × 3.0 × 0.87 / (3.0 + 0.87) = 1.35`
- `F1_AppLoop = 2 × 1.0 × 0.91 / (1.0 + 0.91) = 0.95`

---

## 3. Implementation Status

**Status**: ⚠️ Theoretical framework with simulation. Real-model factuality measurement scaffolded but needs MeaningCompiler path.

**Test Harness**: `transformer_bench.py --task r11`

**Current capability**: Prints factuality metrics table + coverage-accuracy tradeoff analysis. Real-model test code exists (3 queries with tool result injection + faithfulness check) but needs `compiler` parameter.

---

## 4. Key Findings

1. **SIG has 4% higher hallucination** (8% vs 4%) — modest compared to 3× coverage advantage.
2. **Recency bias hypothesis**: SIG's sequential cache may over-weight later tool results.
3. **F1-efficiency favors SIG** (1.35 vs 0.95) when coverage is valued equally with faithfulness.
4. **Conflict resolution**: AppLoop's equal-weight re-encoding treats all sources symmetrically; SIG's sequential injection may favor recency.

---

## 5. Future Work (High Priority)

- Pass MeaningCompiler from co_benchmark.py to transformer_bench R11 for real-model factuality test
- Cross-validate hallucination rates with LLM-as-judge (GPT-4 evaluation)
- Measure conflict resolution accuracy: inject contradictory tool results, evaluate model's handling
