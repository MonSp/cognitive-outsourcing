# R19 Research Plan: 边缘集群中的 SIG 片段路由——面向分布式代理

## 1. Core Research Question

**When multiple edge devices collaboratively execute a task (e.g., multiple cameras + speakers in a home), can SIG's KV fragments be treated as first-class network citizens — routed, serialized, and merged across devices — to enable a distributed cognitive fabric?**

Current multi-device systems exchange raw text or structured data, forcing the receiving model to re-encode everything. SIG offers a radical alternative: devices can pre-compute and transmit KV fragments, allowing the coordinator to merge them without re-encoding — effectively "cognitive data" as the network primitive.

### Sub-Questions

1. **Fragment serialization efficiency**: What is the byte cost of serializing a KV fragment vs the original text? Is KV transmission bandwidth-efficient?
2. **Fragment merge semantics**: How to merge KV fragments from different devices and modalities (vision, text, audio) into a single coherent context?
3. **Latency vs re-encoding tradeoff**: At what network bandwidth does KV transmission beat local re-encoding of text?
4. **Partial fragment routing**: Can a coordinator request only specific layers' or heads' KV entries to reduce bandwidth?
5. **Fault tolerance**: What happens if a fragment arrives late or is corrupted? Can SIG gracefully degrade?

---

## 2. Experimental Design

### Platform
- Two Jetson Orin devices (or simulated with two processes on RTX 4070 SUPER)
- llama.cpp with custom KV serialization/deserialization hooks
- Wi-Fi connection (simulated bandwidth: 10–100 Mbps, latency: 5–50 ms)
- Models: Qwen3.5-0.8B-Q4_K_M, Qwen3.5-4B-Q4_K_M

### Distributed Kitchen Scenario
- **Device A (Vision)**: Processes camera feed for ingredient recognition. Produces KV fragments encoding "detected: tomatoes ×3, onions ×2, basil (fresh)".
- **Device B (Text)**: Processes recipe database queries. Produces KV fragments encoding recipe steps and nutritional information.
- **Coordinator (Controller)**: Receives fragments from both devices, merges into unified KV cache, and generates cooking instructions.

### Baselines
| Baseline | Description |
|----------|-------------|
| **SIG-Fragment** | Devices serialize and transmit KV fragments. Coordinator merges without re-encoding. |
| **Text-Transfer** | Devices transmit raw text descriptions. Coordinator re-encodes all text locally. |
| **AppLoop-Distributed** | Coordinator re-encodes full context (system prompt + received texts) per step. |

### Fragment Serialization Protocol
- KV cache entries: `(num_layers × 2(K+V) × num_heads × head_dim)` per token
- Quantization options: FP16 (full), INT8 (lossy), selective head transmission (top-K attention heads)
- Fragment header: device_id, timestamp, token_range, encoding_format

### Metrics
| Metric | Description |
|--------|-------------|
| End-to-end latency | From device capture to coordinator output |
| Network bandwidth per step | Bytes transmitted over Wi-Fi |
| Fragment merge latency | Time to deserialize and merge KV fragments |
| Re-encoding equivalent time | Time coordinator would spend re-encoding equivalent text |
| Task completion rate | Cooking instruction quality under fragment loss/latency |

---

## 3. Expected Findings

1. **KV fragments are bandwidth-efficient**: For 0.8B models (512-dim, 24 layers), a 50-token KV fragment is ~4.9 MB in FP16 vs ~150 bytes of equivalent text. However, at 100 Mbps Wi-Fi, 4.9 MB transmits in ~400 ms — which is comparable to local re-encoding time for 4B models. The crossover favors KV when (bandwidth × re-encode_time) > fragment_size.
2. **Selective head transmission is key**: Transmitting only top-K attention heads (e.g., top 25%) reduces fragment size by 4× with <5% quality degradation.
3. **INT8 quantization viable**: KV fragments quantized to INT8 retain >95% of FP16 quality while halving bandwidth.
4. **Merge is near-instantaneous**: KV fragment merge is a memory copy operation (<1 ms), making the network the sole bottleneck.
5. **Break-even at ~50 Mbps**: For 4B models, KV transmission beats text re-encoding when Wi-Fi bandwidth exceeds ~50 Mbps.

---

## 4. Recommendation

- **Use SIG fragment routing** when edge devices share the same model architecture (KV dimensions must match) and network bandwidth exceeds 50 Mbps.
- **Default to INT8 fragment quantization** for production deployments.
- **Implement deadline-aware fragment collection**: Coordinator sets a deadline; late fragments are dropped rather than blocking.

---

## 5. Future Work

- Heterogeneous model support: fragment translation between different model architectures
- Multi-hop fragment routing across mesh networks
- Privacy-preserving fragments: differential privacy on KV before transmission
- Integration with R16 (multi-sequence): each device corresponds to one sequence; coordinator merges fragments into active sequences
- Standardized KV fragment interchange format for cross-vendor edge AI
