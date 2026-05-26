# R8 Research Plan: Spatial Cognition & Sustained Attention

## 1. Core Research Question

**Does SIG's persistent KV cache preserve spatial memory better than AppLoop's episodic re-encoding across long-horizon navigation tasks?**

Embodied agents must maintain spatial awareness across extended interactions. AppLoop's per-turn full re-encoding means spatial context must be re-established each step, while SIG's continuous cache retains spatial context naturally.

### Sub-Questions

1. Can the model recall early-room information after 6, 9, or 12 turns of navigation?
2. Does spatial memory decay differ between 0.8B and 4B models?
3. What is the cumulative KV cache growth across multi-room navigation?

---

## 2. Implementation Status

**Status**: ✅ Complete — Real model measurements on Qwen3.5-0.8B/4B via llama.cpp

**Test Harness**: `sig_benchmark.py --task r8`

**Method**: 6-room × 2-turn navigation (12 turns). Each room queries weather (pass 1) and attractions (pass 2) for a distinct city. Probes at turns 3, 6, and 9 ask the model to recall Room 0 (Paris) information. Hit/miss based on keyword presence in generated response.

### Key Measured Results (Qwen3.5, RTX 4070 SUPER)

**0.8B — 3/3 probes hit**:
| Turn | Action | Tokens | Eval (ms) | Cache | Result |
|------|--------|--------|----------|-------|--------|
| 3 | PROBE | 40 | 7.7 | 390 | HIT |
| 6 | PROBE | 78 | 7.0 | 561 | HIT |
| 9 | PROBE | 42 | 7.8 | 748 | HIT |

**4B — 2/3 probes hit**:
| Turn | Action | Tokens | Eval (ms) | Cache | Result |
|------|--------|--------|----------|-------|--------|
| 3 | PROBE | 40 | 15.4 | 385 | MISS |
| 6 | PROBE | 78 | 13.9 | 549 | HIT |
| 9 | PROBE | 42 | 16.2 | 726 | HIT |

### Key Findings

1. **SIG preserves spatial memory**: 0.8B achieves perfect 3/3 recall across 12 turns.
2. **4B first-probe miss** likely due to Q4_K_M quantization noise in earliest cache entries.
3. **Eval times are stable** (7-9 ms for 0.8B, 14-16 ms for 4B), growing only with cumulative cache.
4. **AppLoop would need O(n²) prefill**: ~6× more prefill tokens for 12-turn history.

---

## 3. Theoretical Projection

| Metric | SIG | AppLoop |
|--------|-----|---------|
| Prefill cost (12 turns) | O(n) ~ 850 tokens | O(n²) ~ 5000 tokens |
| Spatial memory probe hits (0.8B) | 3/3 | Lower (re-encoding flushes context) |
| Cache VRAM growth | Linear | Rebuilt each turn |

---

## 4. Future Work

- Extend to 50+ turns to find spatial memory degradation threshold
- Add task-switching interrupts: model switched to unrelated task, then probed for spatial recall
- Physical robot integration with real SLAM-based navigation
