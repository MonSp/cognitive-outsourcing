# R9 Research Plan: Real-Time Constrained SIG — Latency Budget Analysis

## 1. Core Research Question

**Under fixed latency budgets, how should prefill, generation, and teacher planning time be allocated?**

Safety-critical applications (autonomous driving, surgical assistance, drone navigation) impose strict latency constraints. Understanding prefill and generation latency scaling enables safe strategy selection.

### Sub-Questions

1. What is the measured prefill + generation latency at context sizes from 200 to 2000 tokens?
2. At what context size does prefill begin to rival generation time?
3. What strategy (teacher planning, SIG-only, SIG+predictive) is optimal for each context range?

---

## 2. Implementation Status

**Status**: ✅ Complete — Real model measurements on Qwen3.5-0.8B/4B via llama.cpp

**Test Harness**: `sig_benchmark.py --task r9`

**Method**: Prefill + generation latency measured at 200, 500, 1000, 2000 tokens. 2.0s latency budget. Generation: 100 tokens max_new.

### Key Measured Results (Qwen3.5, RTX 4070 SUPER, 2.0s budget)

**0.8B**:
| Context Tokens | Prefill (s) | Gen (s) | Total (s) |
|---------------|-----------|--------|----------|
| 200 | 0.007 | 0.065 | 0.072 |
| 500 | 0.007 | 0.071 | 0.078 |
| 1000 | 0.034 | 0.070 | 0.104 |
| 2000 | 0.087 | 0.067 | 0.154 |

**4B**:
| Context Tokens | Prefill (s) | Gen (s) | Total (s) |
|---------------|-----------|--------|----------|
| 200 | 0.013 | 0.156 | 0.170 |
| 500 | 0.028 | 0.179 | 0.206 |
| 1000 | 0.110 | 0.187 | 0.297 |
| 2000 | 0.288 | 0.185 | 0.473 |

### Strategy Recommendation Matrix

| Context | Strategy | Headroom for Teacher API |
|---------|----------|-------------------------|
| ≤500 tokens | Teacher planning safe | ~1.8s available |
| 500-2000 tokens | SIG-only (skip teacher) | Avoid API latency |
| >2000 tokens | SIG + predictive injection | Maintain safety margin |

### Key Findings

1. **All configurations well within 2.0s budget** (max 0.47s at 4B/2000 tokens).
2. **Generation dominates at 4B** (0.16-0.19s), relatively constant across context sizes.
3. **Prefill scales 12.3× (0.8B) and 22.2× (4B)** over 10× token increase.
4. **Multi-turn SIG advantage**: A single prefill is negligible; AppLoop repeating 10× makes SIG essential.

---

## 4. Future Work

- Implement predictive injection in live runtime
- Speculative decoding + SIG synergy measurement
- Dynamic strategy selection based on real-time cache size
