# Qwen3.5-4B-CPU Multi-Run Results

> **Date**: 2026-05-26 12:53
> **Runs**: 5 per task, 30 OK / 0 FAIL

## Qwen3.5-4B-CPU Multi-Run Results

### Kitchen (5/5 valid runs)

| Baseline | Wall-Clock | Completed | Probe F1 | Gen Time | Prefill Time | vs SIG |
|----------|-----------|-----------|----------|----------|-------------|--------|
| SIG | 36.1±0.2s | 32/? | 50.0±0.0% | 18.6s | 0.8s | 1.0× |
| AppLoop | 152.8±0.5s | 32/? | 50.0±0.0% | 101.5s | 41.9s | 4.2× |
| AppLoop-PC | 91.5±0.2s | 17/? | 0.0±0.0% | 72.3s | 0.0s | 2.5× |
| AppLoop-Sliding | 88.9±0.3s | 13/? | 0.0±0.0% | 71.4s | 14.2s | 2.5× |
| SIG-Hybrid | 0.0s | 0/? | 0.0±0.0% | 0.0s | 0.0s | 0.0× |

### R15: Hybrid Scheduling (5/5 valid runs)

| Metric | Value |
|--------|-------|
| Pure SIG | 36.4±0.8s |
| Pure AppLoop | 154.2±1.7s |
| SIG/AppLoop Speedup | 4.3±0.1x |

### R16: Multi-Sequence (5/5 valid runs)

| Metric | Value |
|--------|-------|
| Total Wall-Clock | 209.2±0.2s |
| Avg Switch Latency | 3605.7±6.5ms |

### R17: Context Compression (5/5 valid runs)

| Strategy | Wall-Clock | Probe F1 | Cache Tokens |
|----------|-----------|----------|-------------|
| None | 4.2s | 50.0±0.0% | 2087.0 |
| Drop-25% | 4.7s | 100.0±0.0% | 932.0 |
| Drop-50% | 4.2s | 0.0±0.0% | 347.0 |
| Recent-30 | 4.2s | 50.0±0.0% | 1436.0 |

### R18: Pipeline Separation (5/5 valid runs)

| Metric | Value |
|--------|-------|
| SIG Wall-Clock | 3.1s |
| AppLoop Wall-Clock | 13.5±0.1s |
| SIG/AppLoop Speedup | 4.3x |
| SIG Prefill | 0.1s |
| AppLoop Prefill | 4.6s |
| Overlap Potential | 3.5±0.1% |

### R19: Fragment Routing (5/5 valid runs)

| Metric | Value |
|--------|-------|
| Total Tokens | 257.0 |
| Re-encode Equivalent | 19.0±0.7ms |
