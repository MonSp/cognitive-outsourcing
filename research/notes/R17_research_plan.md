# R17 Research Plan: 上下文老化与压缩——SIG 长程记忆的显式管理

## 1. Core Research Question

**In ultra-long chains (>200 steps), how to prevent unbounded KV cache growth from causing speed degradation and attention dilution? Can we implement reversible context compression and selective forgetting within the SIG framework?**

SIG's persistent KV cache is both its greatest strength and its Achilles' heel: every injected tool result permanently occupies VRAM and slows attention computation. Without explicit memory management, SIG cannot support truly indefinite edge sessions.

### Sub-Questions

1. **Importance scoring**: Can we predict which KV segments are critical for future retrieval by analyzing attention patterns?
2. **Summarization vs deletion**: Is it better to summarize old tool results (inject a compressed representation) or simply drop low-importance tokens?
3. **Reversibility**: Can compressed KV segments be approximately restored if the agent later needs detailed recall?
4. **Compression throughput tradeoff**: What is the speedup-memory-F1 Pareto frontier under varying compression ratios?
5. **Attention dilution measurement**: Does context compression improve generation quality by reducing attention noise from stale information?

---

## 2. Experimental Design

### Platform
- llama.cpp with custom KV cache manipulation hooks
- Models: Qwen3.5-0.8B-Q4_K_M, Qwen3.5-4B-Q4_K_M
- Hardware: RTX 4070 SUPER (12 GB VRAM)

### Compression Strategies

| Strategy | Description |
|----------|-------------|
| **Importance-Gated Drop** | Compute per-token importance score (attention entropy or gradient-based). Drop bottom-K% tokens. |
| **Summarize-and-Replace** | Use the edge model itself to summarize old tool results (1-2 sentences). Delete original KV, inject summary KV. |
| **Chunk-Level LRU** | Treat each tool interaction as a chunk. Evict least-recently-used chunks while preserving the K most recent. |
| **Hierarchical Compression** | Maintain full KV for last 20 steps, summarized KV for steps 21-100, and discarded for steps 100+. |
| **No Compression (Baseline)** | SIG with unbounded KV cache growth. |

### Benchmark
- Extended EdgeAgent-Kitchen: 200-step session with probing questions every 30 steps targeting information from 10, 50, and 100 steps prior.
- Synthetic stress test: inject a key fact at step T, probe at step T+Δ for Δ ∈ {10, 50, 100, 150}.

### Metrics
| Metric | Description |
|--------|-------------|
| Throughput (tokens/s) | Generation speed at various KV cache sizes |
| VRAM usage | Peak and steady-state GPU memory |
| Retrieval F1 | Accuracy on probing questions targeting old information |
| Decision accuracy (LLM-Judge) | Quality of final outputs |
| Compression overhead (ms) | Time spent on compression operations |

---

## 3. Expected Findings

1. **Uncompressed SIG degrades measurably**: Beyond ~50K tokens in KV cache (for 0.8B), generation speed drops 30-50% due to attention quadratic scaling. Attention dilution causes 10-15% drop in retrieval F1 for 100-step-old facts.
2. **Hierarchical compression is the sweet spot**: Balances memory savings (60-70% VRAM reduction) with retrieval F1 (within 5% of uncompressed).
3. **Summarize-and-Replace outperforms simple dropping**: Model-generated summaries preserve semantic gist better than raw token selection heuristics.
4. **Compression enables unbounded sessions**: With hierarchical compression, SIG maintains consistent throughput and retrieval quality at step 200+.

---

## 4. Recommendation

- **Always deploy SIG with hierarchical compression** for sessions expected to exceed 50 tool-call steps.
- **Use the edge model itself for summarization**: The summarization overhead is amortized over many subsequent steps.
- **Expose importance API to application layer**: Agent frameworks can annotate critical information that must never be compressed (e.g., user allergy data).

---

## 5. Future Work

- Learned importance scoring via lightweight attention-pattern classifier
- KV cache quantization (INT8/INT4) as complementary compression
- Integration with R15 (hybrid scheduling): compress infrequently used sequences while keeping active ones full-fidelity
- Theoretical analysis: information-theoretic bound on KV compression for Transformer attention
