# R7 Research Plan: Multimodal SIG — Structured Data Injection Efficiency

## 1. Core Research Question

**What is the most token-efficient format for injecting structured/multimodal data into the KV cache?**

When embodied agents collect structured sensor data (JSON, key-value pairs, semantic graphs), converting to text introduces a representation bottleneck. The choice of text format directly impacts KV cache footprint and prefill latency.

### Sub-Questions

1. Does JSON structured format save or waste tokens compared to plain text?
2. What is the information density (tokens per semantic character) of different formats?
3. How does eval time scale with token count across model sizes?

---

## 2. Implementation Status

**Status**: ✅ Complete — Real model measurements on Qwen3.5-0.8B/4B via llama.cpp

**Test Harness**: `sig_benchmark.py --task r7`

**Method**: Identical semantic content (weather, attractions, flights for Paris) encoded in three formats: Structured JSON, Plain text, Minimal key-info text. Tokenized and eval'd via llama.cpp.

### Key Measured Results (Qwen3.5, RTX 4070 SUPER)

| Format | 0.8B Tokens | 0.8B Eval (ms) | 4B Tokens | 4B Eval (ms) |
|--------|------------|---------------|----------|-------------|
| Structured JSON | 112 | 20.4 | 112 | 37.1 |
| Plain text | 77 | 27.8 | 77 | 55.8 |
| Minimal text | 70 | 22.7 | 70 | 54.8 |

### Key Findings

1. **JSON is 45% less efficient** than plain text (112 vs 77 tokens) — syntactic characters (brackets, quotes, colons) each consume individual tokens.
2. **Minimal text is optimal** (70 tokens, 9% better than plain text).
3. **4B is 1.8-2.0× slower** per token than 0.8B.

---

## 3. Implications

- **Multimodal SIG should use plain text or minimal key-info format**, not JSON.
- **Vision-to-text projection** (the original R7 hypothesis of direct visual feature injection) remains future work — current measurements focus on text representation efficiency.
- **Streaming sensor injection** should batch updates to amortize per-eval startup cost.

---

## 4. Future Work

- Build end-to-end vision-language SIG prototype (Qwen-VL / LLaVA)
- Direct visual feature projection into KV cache slots (bypass text entirely)
- Streaming injection with adaptive batching based on information novelty
