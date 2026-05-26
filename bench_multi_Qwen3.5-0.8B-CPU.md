# Qwen3.5-0.8B-CPU Multi-Run Results

> **Date**: 2026-05-26 11:27
> **Runs**: 5 per task, 30 OK / 0 FAIL

## Qwen3.5-0.8B-CPU Multi-Run Results

### Kitchen (5/5 valid runs)

| Baseline | Wall-Clock | Completed | Probe F1 | Gen Time | Prefill Time | vs SIG |
|----------|-----------|-----------|----------|----------|-------------|--------|
| SIG | 13.6±0.2s | 32/? | 50.0±0.0% | 8.6s | 0.3s | 1.0× |
| AppLoop | 10.6±0.1s | 32/? | 0.0±0.0% | 2.5s | 7.0s | 0.8× |
| AppLoop-PC | 21.2±0.1s | 18/? | 0.0±0.0% | 15.8s | 0.0s | 1.6× |
| AppLoop-Sliding | 12.3±0.1s | 28/? | 0.0±0.0% | 4.7s | 7.3s | 0.9× |
| SIG-Hybrid | 0.0s | 0/? | 0.0±0.0% | 0.0s | 0.0s | 0.0× |

### R15: Hybrid Scheduling (5/5 valid runs)

| Metric | Value |
|--------|-------|
| Pure SIG | 13.5±0.2s |
| Pure AppLoop | 10.5±0.1s |
| SIG/AppLoop Speedup | 0.8x |

### R16: Multi-Sequence (5/5 valid runs)

| Metric | Value |
|--------|-------|
| Total Wall-Clock | 63.4±0.4s |
| Avg Switch Latency | 1017.7±8.1ms |

### R17: Context Compression (5/5 valid runs)

| Strategy | Wall-Clock | Probe F1 | Cache Tokens |
|----------|-----------|----------|-------------|
| None | 2.6±0.1s | 100.0±0.0% | 2320.0 |
| Drop-25% | 2.1±0.1s | 50.0±0.0% | 956.0 |
| Drop-50% | 2.0s | 50.0±0.0% | 353.0 |
| Recent-30 | 2.5s | 50.0±0.0% | 1659.0 |

### R18: Pipeline Separation (5/5 valid runs)

| Metric | Value |
|--------|-------|
| SIG Wall-Clock | 1.9±0.1s |
| AppLoop Wall-Clock | 1.6s |
| SIG/AppLoop Speedup | 0.8x |
| SIG Prefill | 0.1s |
| AppLoop Prefill | 0.8s |
| Overlap Potential | 5.7±0.1% |

### R19: Fragment Routing (5/5 valid runs)

| Metric | Value |
|--------|-------|
| Total Tokens | 257.0 |
| Re-encode Equivalent | 19.0ms |
