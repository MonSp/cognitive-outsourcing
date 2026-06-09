# Paper 8: SECM-H Experimental Report

> State-Externalizing Cognitive Module Harnesses — Experimental Data Summary
>
> Model: Qwen3.5-2B-Q4_K_M (primary), Qwen3.5-0.8B-Q4_K_M (EXP-15)
> Platform: Windows, llama-cpp-python, n_ctx=16384, n_gpu_layers=99
> All experiments: temperature=0, deterministic generation

---

## 1. Experiment Overview

| Experiment | Description | Conditions | Runs | Total |
|-----------|-------------|-----------|------|-------|
| EXP-1 | State decomposition audit | Static analysis | 4 step counts | 4 |
| EXP-3 | Kitchen benchmark (pre-scripted) | AppLoop, SIG, SIG+SECM-H | 5 | 15 |
| EXP-4 | Ecosystem scaling | SIG, SIG+SECM-H × {5,15,30,50} modules | 3 | 24 |
| EXP-5 | Component ablation | 8 conditions (full, minus each, baseline) | 3 | 24 |
| EXP-9 | Channel strategies | SIG, Sweep-{0,5,10,20,40,80,120}, Selective, OOB | 3 | 60 |
| EXP-10 | Noisy kitchen | SIG, SIG+SECM-H (full), SIG+SECM-H (selective) | 3 | 9 |
| EXP-11 | Agent-driven clean | SIG, SIG+SECM-H (full), SIG+SECM-H (selective) | 3 | 9 |
| EXP-12 | Agent-driven noisy | SIG, SIG+SECM-H (full), SIG+SECM-H (selective) | 3 | 9 |
| EXP-13 | Path A/B disentanglement | ForcedSIG, ForcedSECMH | 3 | 6 |
| EXP-14 | NL state rendering | SECMH-template, SECMH-natural | 3 | 6 |
| EXP-15 | 0.8B agent-driven | Reuses EXP-11/12 logic | 3 | 18 |
| **Total** | | | | **184** |

---

## 2. EXP-1: State Decomposition (H1 Verification)

**Method**: Static audit of module management functions in the CO+SIG architecture.

| Category | Count | Percentage | Description |
|----------|-------|-----------|-------------|
| $F_{ext}$ (fully externalizable) | 13 | 76.5% | Functions that can be entirely offloaded to the harness |
| $F_{partial}$ (partially externalizable) | 2 | 11.8% | Functions requiring partial model involvement |
| $F_{sem}$ (policy-intrinsic) | 2 | 11.8% | Functions that must remain in the model |

**H1 confirmed**: 76.5% > 50% threshold. Context reduction potential: 69.3%.

### Token Overhead by Step Count

| Steps | Total Module Mgmt Tokens | % of 16K Context |
|-------|--------------------------|-----------------|
| 5 | ~120 | 0.7% |
| 15 | ~480 | 2.9% |
| 25 | ~1100 | 6.7% |
| 35 | ~2000 | 12.2% |

---

## 3. EXP-3: Kitchen Benchmark (Pre-Scripted, 2B)

**Scenario**: 35-step kitchen task with 18 tools, pre-scripted tool calls.

### Table 1: Pre-Scripted Benchmark Results (2B, n=5, mean)

| Condition | $Q_{content}$ | Coverage | RespQ | CtxUtil | Wall-Clock (s) | Gen Tokens | Prefill Tokens |
|-----------|---------------|----------|-------|---------|----------------|------------|----------------|
| AppLoop | $0.373$ | $0.116$ | $0.843$ | $0.138$ | $4.2$ | 256 | 33,980 |
| SIG | $\mathbf{0.461}$ | $\mathbf{0.164}$ | $0.881$ | $\mathbf{0.428}$ | $\mathbf{3.8}$ | 556 | 1,579 |
| SIG+SECM-H | $0.320$ | $0.050$ | $0.919$ | $0.179$ | $14.0$ | 2,463 | 4,356 |

**Key findings**:
- SIG baseline achieves highest $Q_{content}$ ($0.461$) at lowest latency ($3.8$s)
- Full SECM-H injection degrades quality by $\Delta = -0.141$ and increases latency 3.7×
- All conditions achieve $Q_{task} = 0.925$ (tool execution rate saturated at 100%)
- Harness overhead: <0.1 ms/step (negligible)

---

## 4. EXP-4: Ecosystem Scaling (2B)

**Method**: Vary number of available modules (5/15/30/50), compare SIG vs SIG+SECM-H.

### Table 2: Ecosystem Scaling Results (2B, n=3, mean)

| Modules | SIG $Q_{content}$ | SIG+SECM-H $Q_{content}$ | $\Delta Q_{content}$ | SIG WC (s) | SECM-H WC (s) |
|---------|-------------------|--------------------------|----------------------|------------|---------------|
| 5 | $0.195$ | $0.396$ | $\mathbf{+0.201}$ | $0.1$ | $0.5$ |
| 15 | $0.397$ | $0.394$ | $-0.003$ | $2.4$ | $10.6$ |
| 30 | $0.419$ | $0.449$ | $+0.030$ | $3.4$ | $11.8$ |
| 50 | $0.419$ | $0.437$ | $+0.018$ | $3.4$ | $11.0$ |

**Key findings**:
- Non-monotonic scaling: negative at 15 modules, positive at 30/50
- H5 (monotonic non-decreasing) **refuted**
- SECM-H latency always 3-4× higher than SIG

---

## 5. EXP-5: Component Ablation (2B)

**Method**: Remove one SECM-H component at a time, measure impact on $Q_{content}$.

### Table 3: Ablation Results (2B, n=3, mean)

| Condition | Removed Component | $Q_{content}$ | Coverage | $\Delta$ vs SIG | Tokens |
|-----------|------------------|---------------|----------|-----------------|--------|
| A0_SIG (baseline) | — | $\mathbf{0.461}$ | $0.164$ | — | 556 |
| A1_no_Rt | Registry | $0.375$ | $0.084$ | $-0.086$ | ~1,800 |
| A2_no_Ht | History | $0.319$ | $0.061$ | $-0.142$ | ~2,463 |
| A3_no_Ct | Confidence | $0.337$ | $0.020$ | $-0.124$ | ~2,463 |
| A4_no_Dt | Dependency | $0.322$ | $0.039$ | $-0.139$ | ~2,463 |
| A5_no_Pt | PatternCache | $0.341$ | $0.040$ | $-0.120$ | ~2,463 |
| A6_no_Bt | Budget | $0.431$ | $0.208$ | $-0.030$ | ~1,800 |
| A_full | None removed | $0.317$ | $0.050$ | $-0.144$ | 2,463 |

**Key findings**:
- All ablated conditions underperform SIG baseline
- Removing Budget ($B_t$) has least negative impact ($\Delta = -0.030$)
- Removing History ($H_t$) has most negative impact ($\Delta = -0.142$)
- Full injection performs worst among all conditions

---

## 6. EXP-9: Channel Strategies (2B + 0.8B)

**Method**: Sweep injection volume from 0 to 120 tokens/step, plus Selective and OOB strategies.

### Table 4: Channel Strategy Results — 2B (n=3, mean)

| Condition | Injection (tok/step) | $Q_{content}$ | Coverage | Wall-Clock (s) | Gen Tokens |
|-----------|---------------------|---------------|----------|----------------|------------|
| SIG (baseline) | 0 | $\mathbf{0.461}$ | $0.164$ | $\mathbf{3.8}$ | 556 |
| Sweep-5 | 5 | $0.491$ | $0.262$ | $3.1$ | 570 |
| Sweep-10 | 10 | $0.414$ | $0.189$ | $5.2$ | 920 |
| Sweep-20 | 20 | $0.379$ | $0.155$ | $7.3$ | 1,310 |
| Sweep-40 | 40 | $0.357$ | $0.120$ | $9.5$ | 1,780 |
| Sweep-80 | 80 | $0.323$ | $0.050$ | $14.0$ | 2,463 |
| Sweep-120 | 120 | $0.320$ | $0.050$ | $14.0$ | 2,463 |
| Selective | ~10 | $0.479$ | $0.224$ | $2.2$ | 373 |
| OOB | ~5 | $0.489$ | $0.275$ | $2.9$ | 524 |

### Table 5: Channel Strategy Results — 0.8B (n=3, mean)

| Condition | $Q_{content}$ | Coverage | Wall-Clock (s) |
|-----------|---------------|----------|----------------|
| SIG | $0.464$ | $0.181$ | $2.5$ |
| Selective | $0.489$ | $0.256$ | $1.9$ |
| OOB | $0.492$ | $0.273$ | $2.3$ |

**Key findings**:
- Benefit window ≤5 tokens/step for 2B ($\Delta = +0.030$)
- 0.8B more receptive to Selective/OOB ($\Delta = +0.025$–$+0.028$)
- Full injection (80+ tok/step) consistently degrades quality

---

## 7. EXP-10: Noisy Kitchen (2B)

**Method**: Same as EXP-3 but with 15% tool failure rate (fixed failure steps: 7, 14, 21, 28).

### Table 6: Noisy Kitchen Results (2B, n=3, mean)

| Condition | $Q_{content}$ | Coverage | Failures | Wall-Clock (s) |
|-----------|---------------|----------|----------|----------------|
| SIG | $\mathbf{0.462}$ | $0.170$ | $4.0$ | $3.1$ |
| SIG+SECM-H (full) | $0.323$ | $0.063$ | $4.0$ | $14.0$ |
| SIG+SECM-H (selective) | $0.314$ | $0.072$ | $4.0$ | $2.2$ |

**Key findings**:
- Noise does not change the qualitative pattern: SIG still wins
- Full injection degrades by $\Delta = -0.141$ (same as clean)
- Selective achieves lowest latency but not highest quality

---

## 8. EXP-11: Agent-Driven Tool Selection — Clean (2B)

**Method**: Model autonomously selects which tool to invoke. Primary metric: tool selection accuracy.

### Table 7: Agent-Driven Clean Results (2B, n=3, mean)

| Condition | Tool Accuracy | Args Accuracy | $Q_{content}$ | Coverage | Wall-Clock (s) | Gen Tokens |
|-----------|--------------|---------------|---------------|----------|----------------|------------|
| SIG | $\mathbf{1.000}$ | $\mathbf{0.429}$ | $0.596$ | $0.391$ | $\mathbf{12.0}$ | 2,280 |
| SIG+SECM-H (full) | $0.971$ | $0.400$ | $0.626$ | $0.451$ | $16.9$ | 3,939 |
| SIG+SECM-H (selective) | $0.886$ | $0.343$ | $\mathbf{0.718}$ | $\mathbf{0.550}$ | $\mathbf{10.4}$ | 2,409 |

**Key findings**:
- SIG achieves 100% tool accuracy in clean scenarios
- Selective achieves highest $Q_{content}$ ($0.718$) but lowest tool accuracy ($88.6\%$)
- Tension: $Q_{content}$ rewards verbose, coverage-rich generation; may not reflect task execution quality

---

## 9. EXP-12: Agent-Driven Tool Selection — Noisy (2B)

**Method**: Same as EXP-11 but with 15% tool failure rate.

### Table 8: Agent-Driven Noisy Results (2B, n=3, mean)

| Condition | Tool Accuracy | Args Accuracy | $Q_{content}$ | Coverage | Wall-Clock (s) |
|-----------|--------------|---------------|---------------|----------|----------------|
| SIG | $0.943$ | $0.400$ | $0.535$ | $0.170$ | $\mathbf{11.4}$ |
| SIG+SECM-H (full) | $\mathbf{0.971}$ | $\mathbf{0.400}$ | $\mathbf{0.636}$ | $\mathbf{0.451}$ | $17.0$ |
| SIG+SECM-H (selective) | $0.914$ | $0.371$ | $0.661$ | $0.550$ | $11.0$ |

**Key findings**:
- **SECM-H-full surpasses SIG under noise**: 97.1% vs 94.3% tool accuracy
- Confidence tracker guides model away from recently failed tools
- $Q_{content}$ improvement: $\Delta = +0.101$ (full), $+0.126$ (selective)
- H3 conditionally confirmed in agent-driven noisy scenarios

---

## 10. EXP-13: Path A/B Disentanglement (2B)

**Method**: Force model to use ground-truth tool (eliminates Path A), inject SECM-H state (isolates Path B).

### Table 9: Path A/B Disentanglement (2B, n=3, mean)

| Condition | Tool Accuracy | $Q_{content}$ | Coverage | Wall-Clock (s) | Gen Tokens |
|-----------|--------------|---------------|----------|----------------|------------|
| ForcedSIG | $\mathbf{1.000}$ | $0.614$ | $0.430$ | $13.3$ | 2,486 |
| ForcedSECMH | $0.914$ | $\mathbf{0.664}$ | $\mathbf{0.512}$ | $\mathbf{9.8}$ | 1,743 |

**Key findings**:
- Path B confirmed: $\Delta Q_{content} = +0.050$, $\Delta$ Coverage $= +0.082$
- SECM-H changes generation behavior even when tool selection is identical
- ForcedSECMH generates 30% fewer tokens (1,743 vs 2,486) — state injection focuses generation

---

## 11. EXP-14: Natural Language State Rendering (2B)

**Method**: Compare template format vs natural language format for SECM-H state.

### Template Format
```
[SECM-H] Modules: 18 available, 12 invoked, 6 pending. Budget: 50/2048. Top: set_oven(0.92). Patterns: 3.
```

### Natural Language Format
```
You have 18 tools available, 12 already used. Most reliable tool: set_oven (confidence 0.92). You have 50 of 2048 budget remaining. 3 cognitive patterns cached.
```

### Table 10: Format Interference Results (2B, n=3, mean)

| Rendering | Tool Accuracy | $Q_{content}$ | Coverage | Wall-Clock (s) | Gen Tokens |
|-----------|--------------|---------------|----------|----------------|------------|
| Template | $\mathbf{0.971}$ | $0.626$ | $0.435$ | $17.1$ | 3,191 |
| Natural language | $0.943$ | $\mathbf{0.739}$ | $\mathbf{0.610}$ | $\mathbf{11.5}$ | 2,092 |

**Key findings**:
- NL rendering: $\Delta Q_{content} = +0.113$, $\Delta$ Coverage $= +0.175$
- Format interference is a significant component of attention disruption
- NL reduces latency by 33% ($11.5$s vs $17.1$s) and tokens by 34%

---

## 12. EXP-15: 0.8B Agent-Driven

**Method**: Replicate EXP-11/12 with Qwen3.5-0.8B-Q4_K_M.

### Table 11: 0.8B Agent-Driven Clean (n=3, mean)

| Condition | Tool Accuracy | Args Accuracy | $Q_{content}$ | Wall-Clock (s) |
|-----------|--------------|---------------|---------------|----------------|
| SIG | $\mathbf{0.743}$ | $\mathbf{0.371}$ | $0.630$ | $\mathbf{7.5}$ |
| SIG+SECM-H (full) | $0.543$ | $0.171$ | $\mathbf{0.647}$ | $7.7$ |
| SIG+SECM-H (selective) | $0.714$ | $0.343$ | $0.589$ | $7.7$ |

### Table 12: 0.8B Agent-Driven Noisy (n=3, mean)

| Condition | Tool Accuracy | Args Accuracy | $Q_{content}$ | Wall-Clock (s) |
|-----------|--------------|---------------|---------------|----------------|
| SIG | $\mathbf{0.714}$ | $\mathbf{0.343}$ | $0.589$ | $\mathbf{7.7}$ |
| SIG+SECM-H (full) | $0.371$ | $0.114$ | $\mathbf{0.636}$ | $7.8$ |
| SIG+SECM-H (selective) | $0.571$ | $0.343$ | $0.456$ | $5.6$ |

**Key findings**:
- 0.8B struggles with tool selection (74.3% clean vs 2B's 100%)
- SECM-H degrades 0.8B tool accuracy ($-20$pp clean, $-34$pp noisy)
- Content quality benefit marginal ($\Delta = +0.017$ clean, $+0.047$ noisy)
- Supports Mechanism 4 (Harness-1 boundary violation) for small models

---

## 13. Hypothesis Assessment Summary

| Hypothesis | Claim | Status | Key Evidence |
|-----------|-------|--------|-------------|
| **H1** | $\geq 50\%$ functions externalizable | **Confirmed** | 76.5% (13/17) |
| **H2** | State renders in $\leq$100 tokens | **Confirmed** | Avg 77.3 tokens |
| **H3** | $Q_{SECM-H} - Q_{SIG} \geq 0.03$ | **Conditionally confirmed** | Pre-scripted: refuted ($\Delta = -0.141$). Agent-driven noisy: confirmed ($\Delta = +0.101$) |
| **H4** | Per-step overhead $\leq$5ms | **Confirmed (harness)** | <0.1 ms/step. Refuted (system): 3.7× wall-clock |
| **H5** | Monotonic scaling with ecosystem size | **Refuted** | Non-monotonic: negative at 15, positive at 30/50 |

---

## 14. Core Open Questions

### 14.1 Path A/B Causal Decoupling

The $Q_{content}$ improvement in agent-driven experiments has two components:
- **Path A**: Better tool selection (+2.8pp accuracy)
- **Path B**: Changed generation behavior (+0.082 Coverage from EXP-13)

These are disproportionate. If Path B dominates, SECM-H may function as an expensive prompt engineering technique rather than a genuine module management layer.

### 14.2 Q_content vs ToolAcc Tension

Selective injection achieves highest $Q_{content}$ ($0.718$) but lowest tool accuracy ($88.6\%$) in clean scenarios. The evaluator may reward verbose, coverage-rich generation over precise tool execution.

### 14.3 0.8B: Two Competing Explanations

1. **Model-size limitation** (Mechanism 4): larger models benefit more → monotonic improvement
2. **Generation stabilizer** (§7.6): benefit peaks at intermediate size → inverted U-curve

These predict contradictory futures. Testing across 4B/7B/14B models needed.

---

## 15. Reproduction

```bash
# Run all experiments (requires Qwen3.5-2B-Q4_K_M.gguf in models/)
python exp8_state_decomposition.py
python exp8_kitchen_benchmark.py --n-runs 5
python exp8_v2_channel_strategies.py --task all --n-runs 3
python exp8_v3_agent_driven.py --task all --n-runs 3 --n-steps 35

# Run 0.8B experiments (requires Qwen3.5-0.8B-Q4_K_M.gguf)
python exp8_v3_agent_driven.py --task exp15 --n-runs 3 --n-steps 35 --model models/Qwen3.5-0.8B-Q4_K_M.gguf
```

**Seed**: All experiments use `random.seed(42)` for determinism. With identical model files and library versions, results should be exactly reproducible.

**Runtime**: Full experiment suite takes approximately 20 minutes on a system with RTX 4090 (24GB VRAM).
