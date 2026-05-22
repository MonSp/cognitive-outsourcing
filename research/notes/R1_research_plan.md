# R1 Research Plan: SIG Injection Information Theory Analysis

## 1. Core Research Question

**What is the information loss upper bound between SIG injection vs full re-encoding?**

When using SIG (Sparse Injection Guidance) to inject pre-computed KV states into a target model versus performing full re-encoding of the input sequence, how much information is theoretically lost? This question can be formalized as:

```
I_loss = I_full(X; H_full) - I_inject(X; H_inject)
```

Where:
- `I_full` = mutual information between input X and hidden states under full re-encoding
- `I_inject` = mutual information between input X and hidden states under SIG injection
- `H_full` = hidden representations from full re-encoding
- `H_inject` = hidden representations from SIG injection

### Sub-Questions

1. **Layer-wise information degradation**: Does information loss accumulate monotonically across layers, or are there recovery points?
2. **Attention head heterogeneity**: Do different attention heads retain information at different rates under injection?
3. **Sequence length dependency**: How does the information loss scale with input sequence length?
4. **Semantic vs syntactic information**: Is one type of information more vulnerable to injection-induced loss?

---

## 2. Metrics

### 2.1 KL Divergence

Measures the distributional divergence between attention patterns under full re-encoding vs SIG injection.

```
KL(P_full || P_inject) = sum_i P_full(i) * log(P_full(i) / P_inject(i))
```

- **Application**: Compare attention distributions per head per layer
- **Interpretation**: Higher KL = greater deviation from full re-encoding behavior
- **Symmetrized variant**: Use `0.5 * (KL(P||Q) + KL(Q||P))` for symmetric comparison

### 2.2 JS Divergence

Symmetric and bounded alternative to KL divergence for comparing hidden state distributions.

```
JS(P || Q) = 0.5 * KL(P || M) + 0.5 * KL(Q || M), where M = 0.5 * (P + Q)
```

- **Range**: [0, log(2)] for base-2 log, or [0, 1] with sqrt transformation
- **Application**: Compare output distributions of corresponding layers
- **Advantage**: Bounded, making cross-layer and cross-model comparisons meaningful

### 2.3 Head Agreement Rate

Measures the fraction of attention heads that produce top-k attention targets that agree between the two methods.

```
Agreement Rate = (1 / (L * H)) * sum_{l,h} |TopK(A_full) intersect TopK(A_inject)| / k
```

Where:
- `L` = number of layers
- `H` = number of heads per layer
- `TopK(A)` = set of top-k attended positions

- **Interpretation**: 1.0 = perfect agreement, 0.0 = complete disagreement
- **Sensitivity**: k can be varied (e.g., k=1, 3, 5) to analyze precision at different granularities

### 2.4 Additional Metrics

| Metric | Formula | Purpose |
|--------|---------|---------|
| Cosine Similarity | `cos(H_full, H_inject)` | Layer-wise representation alignment |
| Centered Kernel Alignment (CKA) | `CKA(K_full, K_inject)` | Non-linear representation similarity |
| Information Bottleneck Ratio | `I(X; Z_inject) / I(X; Z_full)` | Relative information retention |

---

## 3. Layer Sensitivity Hypothesis

### Primary Hypothesis

**H1**: Early layers (1-4) are most sensitive to SIG injection, exhibiting the highest information loss, because they establish the foundational semantic representations that subsequent layers build upon.

### Secondary Hypotheses

**H2**: Middle layers (5-8) show moderate sensitivity but can partially recover from early-layer distortion through self-correction mechanisms in the attention computation.

**H3**: Late layers (9-12) are least sensitive to injection artifacts, as they operate on already-formed representations and focus on task-specific refinement.

**H4**: Attention heads with high "information flow" scores (measured by gradient-based attribution) will show disproportionately higher KL divergence under injection compared to low-flow heads.

### Expected Layer Sensitivity Curve

```
Information Loss
    ^
    |    *
    |   * *
    |  *   *
    | *     *
    |*       *---*---*
    +-------------------> Layer Index
    0  1  2  3  4  5  6
    (Early) (Mid) (Late)
```

---

## 4. Experiment Protocol

### 4.1 Benchmark Scenarios

| # | Scenario | Dataset | Input Length | Expected Difficulty |
|---|----------|---------|--------------|---------------------|
| 1 | Short factual QA | TriviaQA | 64-128 | Low |
| 2 | Long-context QA | QMSum | 512-1024 | Medium |
| 3 | Multi-hop reasoning | HotpotQA | 256-512 | Medium |
| 4 | Summarization | CNN/DailyMail | 512-1024 | Medium |
| 5 | Code generation | HumanEval | 128-256 | High |
| 6 | Math reasoning | GSM8K | 64-256 | High |
| 7 | Dialogue understanding | MuTOD | 256-512 | Medium |
| 8 | Cross-lingual transfer | XNLI | 128-256 | High |
| 9 | Instruction following | AlpacaEval | 256-1024 | High |

### 4.2 Experimental Setup

**Base Model**: LLaMA-2-7B (or equivalent open-weight model)

**SIG Injection Parameters**:
- Injection granularity: per-layer KV cache
- Injection points: after layers [2, 4, 6, 8]
- Compression ratio: [0.25, 0.5, 0.75, 1.0]

**Control Condition**: Full re-encoding of identical inputs

**Evaluation Steps**:

```
For each benchmark scenario:
    1. Sample N = 100 instances from dataset
    2. Run full re-encoding, collect:
       - Hidden states at each layer
       - Attention weights at each head
       - Final output distribution
    3. Run SIG injection with varying compression ratios
       - Collect same intermediate representations
    4. Compute metrics:
       - KL divergence per head per layer
       - JS divergence per layer
       - Head agreement rate per layer
       - Cosine similarity per layer
    5. Aggregate results across instances
    6. Record task performance delta (accuracy/F1/BLEU)
```

### 4.3 Statistical Rigor

- **Confidence intervals**: Bootstrap with 1000 resamples for 95% CI
- **Significance testing**: Paired t-test between full and injection conditions
- **Effect size**: Cohen's d for practical significance
- **Multiple comparison correction**: Benjamini-Hochberg FDR control

### 4.4 Ablation Studies

| Ablation | Variable | Values |
|----------|----------|--------|
| A1 | Injection depth | {early only, mid only, late only, all} |
| A2 | KV compression | {0.1, 0.25, 0.5, 0.75, 1.0} |
| A3 | Sequence length | {64, 128, 256, 512, 1024} |
| A4 | Model size | {1B, 7B, 13B} |

---

## 5. Expected Outcomes

### 5.1 Quantitative Predictions

| Metric | Early Layers | Middle Layers | Late Layers |
|--------|-------------|---------------|-------------|
| KL Divergence | 0.15 - 0.35 | 0.05 - 0.15 | 0.02 - 0.08 |
| JS Divergence | 0.08 - 0.18 | 0.03 - 0.08 | 0.01 - 0.04 |
| Head Agreement (k=5) | 0.70 - 0.85 | 0.80 - 0.92 | 0.88 - 0.96 |
| Cosine Similarity | 0.85 - 0.93 | 0.90 - 0.96 | 0.94 - 0.98 |

### 5.2 Qualitative Predictions

1. **Information loss upper bound**: Total information loss across all layers will be bounded by approximately 5-10% of the mutual information in full re-encoding, with the majority concentrated in early layers.

2. **Head specialization**: Approximately 20-30% of attention heads (likely syntactic and positional heads) will show near-perfect agreement, while semantic reasoning heads will show the highest divergence.

3. **Task performance correlation**: Information loss metrics will correlate with task performance degradation at r > 0.7, validating their use as proxy measures for injection quality.

4. **Compression threshold**: A compression ratio of 0.5 will represent a phase transition point, below which information loss increases superlinearly.

### 5.3 Deliverables

- [ ] Information loss upper bound estimates per benchmark scenario
- [ ] Layer-wise sensitivity heatmaps
- [ ] Head-level agreement analysis with clustering
- [ ] Compression ratio vs. information loss curves
- [ ] Statistical significance report for all hypotheses
- [ ] Recommendations for optimal injection configuration

---

## 6. Timeline

| Phase | Duration | Deliverable |
|-------|----------|-------------|
| Phase 1: Setup | Week 1-2 | Infrastructure, data sampling |
| Phase 2: Data Collection | Week 3-4 | Full re-encoding baselines |
| Phase 3: SIG Experiments | Week 5-6 | Injection results across scenarios |
| Phase 4: Analysis | Week 7-8 | Metric computation, hypothesis testing |
| Phase 5: Reporting | Week 9-10 | Final report, visualizations |
