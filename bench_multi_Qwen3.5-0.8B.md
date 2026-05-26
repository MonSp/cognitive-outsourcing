# Qwen3.5-0.8B Multi-Run Results

> **Date**: 2026-05-26 10:37
> **Runs**: 5 per task, 30 OK / 0 FAIL

## Qwen3.5-0.8B Multi-Run Results

### Kitchen (5/5 valid runs)

| Baseline | Wall-Clock | Completed | Probe F1 | Gen Time | Prefill Time | vs SIG |
|----------|-----------|-----------|----------|----------|-------------|--------|
| SIG | 2.2±0.1s | 32/? | 50.0±0.0% | 1.5s | 0.1s | 1.0× |
| AppLoop | 2.0±0.1s | 32/? | 50.0±0.0% | 0.8s | 1.0s | 0.9× |
| AppLoop-PC | 4.4±0.0s | 17/? | 0.0±0.0% | 3.4s | 0.0s | 2.0× |
| AppLoop-Sliding | 4.1s | 32/? | 50.0±0.0% | 1.6s | 2.0s | 1.8× |
| SIG-Hybrid | 0.0s | 0/? | 0.0±0.0% | 0.0s | 0.0s | 0.0× |

### R15: Hybrid Scheduling (5/5 valid runs)

| Metric | Value |
|--------|-------|
| Pure SIG | 2.2±0.1s |
| Pure AppLoop | 2.1±0.0s |
| SIG/AppLoop Speedup | 0.9x |

### R16: Multi-Sequence (5/5 valid runs)

| Metric | Value |
|--------|-------|
| Total Wall-Clock | 8.4s |
| Avg Switch Latency | 122.6±0.6ms |

### R17: Context Compression (5/5 valid runs)

| Strategy | Wall-Clock | Probe F1 | Cache Tokens |
|----------|-----------|----------|-------------|
| None | 2.6±0.0s | 100.0±0.0% | 2320.0 |
| Drop-25% | 2.1±0.1s | 50.0±0.0% | 956.0 |
| Drop-50% | 2.0s | 50.0±0.0% | 353.0 |
| Recent-30 | 2.5±0.1s | 50.0±0.0% | 1659.0 |

### R18: Pipeline Separation (5/5 valid runs)

| Metric | Value |
|--------|-------|
| SIG Wall-Clock | 1.9±0.0s |
| AppLoop Wall-Clock | 1.6s |
| SIG/AppLoop Speedup | 0.8x |
| SIG Prefill | 0.1s |
| AppLoop Prefill | 0.8s |
| Overlap Potential | 5.6±0.1% |

### R19: Fragment Routing (5/5 valid runs)

| Metric | Value |
|--------|-------|
| Total Tokens | 257.0 |
| Re-encode Equivalent | 18.8±0.4ms |
