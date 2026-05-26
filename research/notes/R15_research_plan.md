# R15 Research Plan: 混合推理调度——SIG 与 AppLoop-PC 的自适应切换

## 1. Core Research Question

**Can an online decision policy, based on task-graph structural features (contiguous length, context sharing degree, update granularity), dynamically choose between SIG and prefix caching to approach optimal performance under all conditions?**

SIG excels on long contiguous tool chains with incremental updates, while AppLoop-PC excels on short, independent, prefix-heavy queries. A single-mode strategy is suboptimal for real-world workloads that mix both patterns. This research investigates whether a lightweight hybrid scheduler can capture the best of both worlds.

### Sub-Questions

1. **Feature definition**: What task-graph features reliably predict SIG vs AppLoop-PC superiority? Candidates: chain length, branching factor, shared prefix ratio, average update size, interleaving depth.
2. **Decision boundary characterization**: At what chain length does SIG's prefill savings overtake AppLoop-PC's prefix reuse? Does this boundary shift with model size?
3. **Heuristic vs learned**: Can a simple threshold-based heuristic match a small learned classifier? What is the overhead of the decision itself?
4. **EdgeAgent-Kitchen validation**: On the mixed-workload EdgeAgent-Kitchen benchmark, does hybrid scheduling achieve Pareto-optimal wall-clock time without sacrificing task completion rate?

---

## 2. Experimental Design

### Platform
- llama.cpp with CUDA acceleration
- Models: Qwen3.5-0.8B-Q4_K_M, Qwen3.5-4B-Q4_K_M
- Hardware: RTX 4070 SUPER (12 GB VRAM)

### Benchmark
- EdgeAgent-Kitchen: 3-hour session with interleaved recipe planning (30 steps), cooking guidance (20 steps), inventory management (15 steps), and 5 random interruptions.
- Synthetic mixed-workload generator with controllable task-graph parameters.

### Baselines
| Baseline | Description |
|----------|-------------|
| SIG-only | Persistent KV cache for all operations |
| AppLoop-PC-only | Prefix caching with full re-encoding |
| SIG-Hybrid (heuristic) | Threshold-based switch: chain_length > T → SIG, else AppLoop-PC |
| SIG-Hybrid (learned) | Lightweight classifier (logistic regression or small MLP) trained on task features |

### Metrics
- Wall-clock time (primary)
- Task completion rate
- Decision overhead (ms)
- Per-step mode distribution analysis

### Decision Features
| Feature | Description |
|---------|-------------|
| `chain_depth` | Number of consecutive tool calls in current sub-chain |
| `shared_prefix_ratio` | Fraction of tokens shared with previous request |
| `avg_update_size` | Mean token count of tool result injections |
| `branching_factor` | Number of concurrent sub-tasks at current node |
| `context_length` | Current total KV cache length |
| `is_interruption` | Boolean: whether current turn is a replan |

---

## 3. Expected Findings

1. **Clear decision boundary**: SIG dominates when chain_depth ≥ 5–8 and shared_prefix_ratio < 0.3; AppLoop-PC dominates when queries are short and prefix-heavy.
2. **Heuristic suffices**: Simple threshold-based switching achieves >90% of the oracle's performance, making learned models unnecessary for production deployment.
3. **EdgeAgent-Kitchen result**: Hybrid outperforms both SIG-only (on short queries) and AppLoop-PC-only (on long chains), achieving the best overall wall-clock time.
4. **Zero-cost transition**: Decision overhead is negligible (<1 ms) compared to prefill/decode times.

---

## 4. Recommendation

- **Deploy heuristic hybrid unconditionally** on edge devices serving mixed workloads.
- **Tune thresholds per model size**: 0.8B models benefit from earlier SIG switching (shorter chain_depth threshold) due to proportionally higher prefill overhead.
- **Surface mode selection to application layer**: Allow agent frameworks to annotate task-graph nodes with expected characteristics for optimal scheduling.

---

## 5. Future Work

- Extend decision features to include attention entropy and KV cache utilization metrics
- Multi-armed bandit formulation for online threshold adaptation
- Integration with R17 (context compression) for tiered memory management: compress + SIG for very long retention, AppLoop-PC for ephemeral queries
