# R18 Research Plan: 预填充与生成的流水线分离——SIG 对推测解码的天然适配

## 1. Core Research Question

**SIG treats tool result injection as "prefill" and subsequent reasoning as "decode." Can this clean separation enable pipeline parallelism where injection and generation proceed concurrently? Does this structural property make SIG a natural fit for speculative decoding, amplifying acceleration beyond simple prefill savings?**

In standard AppLoop, every tool result forces a full context re-encode, serializing prefill and decode. SIG's incremental injection opens the door to overlapping computation: the model can generate from cached context while the next tool result is being prefilled into a parallel attention head.

### Sub-Questions

1. **Injection-generation parallelism**: Can SIG's `n_tokens` token injection run concurrently with ongoing generation without stalling the decode pipeline?
2. **Speculative decoding synergy**: When combined with a draft model for speculative decoding, does SIG's prefill separation increase the effective speculation window?
3. **Hardware utilization**: What is the GPU SM occupancy and memory bandwidth utilization under SIG pipeline vs AppLoop serial execution?
4. **Model size effect**: Does pipeline parallelism benefit scale with model size (larger models have more headroom for overlapping compute and memory)?

---

## 2. Experimental Design

### Platform
- llama.cpp with custom CUDA graph and pipeline modifications
- Models: Qwen3.5-0.8B-Q4_K_M, Qwen3.5-4B-Q4_K_M
- Hardware: RTX 4070 SUPER (12 GB VRAM)

### Pipeline Configurations

| Configuration | Description |
|---------------|-------------|
| **SIG-Serial** | Standard SIG: inject tool result, wait for prefill completion, then decode. |
| **SIG-Pipeline** | Modified SIG: initiate decode from cached KV while tool result prefill runs on a parallel CUDA stream. |
| **SIG-SpecDec** | SIG + speculative decoding with a 0.5B draft model. Draft generates speculative tokens during tool result prefill. |
| **AppLoop-Serial** | Full re-encode per step (baseline). |
| **AppLoop-SpecDec** | AppLoop + speculative decoding (baseline with speculation). |

### Benchmark
- EdgeAgent-Kitchen: 50-step session with tool results of varying sizes (10–2000 tokens) to stress pipeline balance.
- Synthetic: fixed-length tool chains with controlled injection sizes.

### Metrics
| Metric | Description |
|--------|-------------|
| Wall-clock time per step | End-to-end latency including prefill + decode |
| Prefill-decode overlap ratio | Fraction of prefill time hidden behind decode |
| Speculative acceptance rate | Draft token acceptance rate under SIG vs AppLoop |
| GPU utilization | SM occupancy, memory bandwidth via nvml |
| Tokens per second | Aggregate throughput |

---

## 3. Expected Findings

1. **Modest pipeline gain on small models (0.8B)**: Limited parallelism due to small model depth. ~10-15% prefill overlap.
2. **Significant pipeline gain on 4B**: Deeper model architecture allows 30-40% prefill overlap with decode on independent CUDA streams.
3. **SIG+SpecDec multiplicative gain**: Speculative decoding's draft window benefits from SIG's prefill separation — the draft model can generate during injection, effectively hiding prefill latency entirely. Expected 2-4× speedup over AppLoop+SpecDec in tool-heavy scenarios.
4. **AppLoop cannot pipeline**: Full re-encode requires the complete context before any decode token, making pipeline parallelism structurally impossible.

---

## 4. Recommendation

- **SIG+SpecDec is the definitive edge inference configuration** for agent workloads. The structural prefill-decode separation of SIG makes speculative decoding not just compatible but synergistic.
- **Prioritize 4B+ models** for pipeline parallelism benefits. 0.8B models see limited gain from overlap but still benefit from speculative decoding.
- **Tune injection batch size**: Larger injections amortize CUDA kernel launch overhead; batch multiple pending tool results when possible.

---

## 5. Future Work

- CUDA graph capture for repeated injection patterns
- Adaptive speculation depth based on prefill queue length
- Integration with R17: compressed KV cache reduces prefill time, shifting pipeline balance
- Multi-GPU: distribute prefill and decode across separate GPUs for full pipeline decoupling
