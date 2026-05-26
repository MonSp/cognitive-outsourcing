# R6 Research Plan: Dynamic Replanning — Online Recovery from Tool Failures

## 1. Core Research Question

**Can CO evolve from "plan once, execute fully" to online dynamic adjustment when tool calls fail?**

In real-world environments, teacher-generated plans encounter failures: tools become unavailable, return unexpected results, or time out. CO must detect plan insufficiency and recover without requiring a full teacher re-consultation.

### Sub-Questions

1. What is the overhead of SIG-based incremental retry vs AppLoop full re-encoding after a tool failure?
2. How does the SIG recovery advantage scale with tool chain depth?
3. What is the minimum failure rate at which dynamic replanning becomes justified?

---

## 2. Implementation Status

**Status**: ✅ Complete — Real model measurements on Qwen3.5-0.8B/4B via llama.cpp

**Test Harness**: `co_benchmark.py --task r6`

**Method**: 6-tool travel planning chain with 15% random failure injection. SIG (incremental injection + retry) vs AppLoop (full re-prefill + retry). Tool failure recovery time, prefill overhead, and total latency measured.

### Key Measured Results (Qwen3.5, RTX 4070 SUPER)

| Metric | 0.8B SIG | 0.8B AppLoop | 4B SIG | 4B AppLoop |
|--------|----------|-------------|--------|-----------|
| Total time (s) | 0.37 | 0.29 | 1.19 | 0.99 |
| Prefill time (s) | 0.14 | 0.07 | 0.20 | 0.20 |
| Generation time (s) | 0.20 | 0.19 | 0.94 | 0.76 |
| Recovery time (s) | 0.01 | 0.01 | 0.02 | 0.02 |

### Key Findings

1. **Small-chain overhead**: At 6 tools, SIG injection management has marginal overhead (0.8× vs AppLoop). Baseline shows 2.38-5.26× SIG advantage at 14-22 tools — advantage emerges at scale.
2. **Recovery is cheap**: Tool retry costs ~10 ms regardless of mode.
3. **SIG's structural advantage**: Per-step failure recovery without cache invalidation vs AppLoop's full-context re-encoding.

---

## 3. Experiment Protocol

| Phase | Action | Status |
|-------|--------|--------|
| Inject 15% random tool failures | 6-tool chain, 1/6 fails | ✅ Done |
| Measure SIG vs AppLoop recovery latency | Real llama.cpp eval timing | ✅ Done |
| Scale to 14+ tool chains | Where SIG advantage should be pronounced | 🔲 Future |
| Adaptive recovery strategy selection | Dynamic choice of retry/replan/teacher-reconsult | 🔲 Future |

---

## 4. Future Work

- Test at 14+ tool chains where baseline shows 4.96-5.26× SIG advantage
- Implement adaptive recovery strategy selection based on failure type
- Add teacher-reconsult mechanism for complex failure scenarios
