# Qwen3.5-4B Multi-Run Results

> **Date**: 2026-05-26 15:57
> **Runs**: 3 per task, 15 OK / 0 FAIL

## Qwen3.5-4B Multi-Run Results

### Kitchen (3/3 valid runs)

| Baseline | Wall-Clock | Completed | Probe F1 | Gen Time | Prefill Time | vs SIG |
|----------|-----------|-----------|----------|----------|-------------|--------|
| SIG | 6.2s | 32/? | 0.0±0.0% | 4.7s | 0.1s | 1.0× |
| AppLoop | 15.8±0.1s | 32/? | 0.0±0.0% | 9.0s | 5.5s | 2.5× |
| AppLoop-PC | 23.5±0.1s | 32/? | 0.0±0.0% | 15.5s | 0.0s | 3.8× |
| AppLoop-Sliding | 15.9±0.1s | 32/? | 0.0±0.0% | 9.1s | 5.5s | 2.6× |
| SIG-Hybrid | 3.8s | 32/? | 0.0±0.0% | 2.8s | 0.0s | 0.6× |

### R15: Hybrid Scheduling (3/3 valid runs)

| Metric | Value |
|--------|-------|
| Pure SIG | 6.2s |
| Pure AppLoop | 15.9s |
| SIG/AppLoop Speedup | 2.5x |

### R16: Multi-Sequence (3/3 valid runs)

| Metric | Value |
|--------|-------|
| Total Wall-Clock | 18.2±0.1s |
| Avg Switch Latency | 246.0±0.2ms |

### R17: Context Compression (3/3 valid runs)

| Strategy | Wall-Clock | Probe F1 | Cache Tokens |
|----------|-----------|----------|-------------|
| None | 6.6s | 0.0±0.0% | 2336.0 |
| Drop-25% | 6.9s | 50.0±0.0% | 1006.0 |
| Drop-50% | 6.7±0.1s | 0.0±0.0% | 386.0 |
| Recent-30 | 9.4s | 0.0±0.0% | 1602.0 |

### R18: Pipeline Separation (3/3 valid runs)

| Metric | Value |
|--------|-------|
| SIG Wall-Clock | 3.4s |
| AppLoop Wall-Clock | 13.6±0.1s |
| SIG/AppLoop Speedup | 4.0x |
| SIG Prefill | 0.1s |
| AppLoop Prefill | 4.6±0.1s |
| Overlap Potential | 3.3±0.1% |
