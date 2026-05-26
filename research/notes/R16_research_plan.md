# R16 Research Plan: 多序列并发——SIG 对多任务持久状态的高效隔离

## 1. Core Research Question

**Can SIG maintain multiple independent persistent KV sequences within a single inference instance, serving different tasks or users with near-zero switching overhead?**

Edge deployments (smart homes, multi-robot systems) serve multiple concurrent agents. Running separate inference instances wastes VRAM through redundant model weights. SIG's persistent KV cache model offers a natural path to multi-tenancy: one model instance, multiple isolated KV sequences.

### Sub-Questions

1. **Switching latency**: What is the overhead of switching between N active KV sequences vs N independent AppLoop instances?
2. **Memory scaling**: How does total VRAM grow with sequence count under SIG vs independent instances?
3. **Shared prefix reuse**: Can common prefixes (system prompts, shared context) be shared across sequences without cross-contamination?
4. **Isolation guarantees**: Does cross-sequence attention leakage occur? How to enforce strict KV isolation?

---

## 2. Experimental Design

### Platform
- llama.cpp multi-sequence API (`llama_decode` with per-sequence `llama_batch`)
- Models: Qwen3.5-0.8B-Q4_K_M, Qwen3.5-4B-Q4_K_M
- Hardware: RTX 4070 SUPER (12 GB VRAM)

### Task Design
- 3 concurrent EdgeAgent-Kitchen sessions, each representing a different household:
  - Household A: Italian cuisine preferences, nut allergies, 4-person family
  - Household B: Vegan diet, gluten-free, 1-person household
  - Household C: Standard diet, no allergies, 2-person household
- Each household runs an independent long tool chain (30+ steps)
- Sequences are interleaved round-robin (1 step per household per round)

### Baselines
| Baseline | Description |
|----------|-------------|
| SIG-MultiSeq | Single llama.cpp instance, 3 KV sequences, round-robin |
| AppLoop-MultiInstance | 3 independent AppLoop processes (separate model loads) |
| AppLoop-Sequential | Single instance, re-encode full context per switch |

### Metrics
| Metric | Description |
|--------|-------------|
| Per-step latency | Wall-clock time per tool-call step (including switch) |
| VRAM peak | Maximum GPU memory during entire session |
| KV cache isolation F1 | Cross-contamination detection: inject household-A-specific facts into B's tool results, measure if B's responses leak A's information |
| Throughput | Steps completed per minute across all sequences |

---

## 3. Expected Findings

1. **VRAM advantage**: SIG-MultiSeq uses ~1× model weights + 3× KV cache, vs AppLoop-MultiInstance requiring 3× model weights. For 4B model (~4.7 GB weights + ~1 GB per KV cache), SIG saves ~9 GB VRAM.
2. **Switching latency < 1 ms**: llama.cpp sequence switching is a pointer change, not a data copy.
3. **Perfect isolation**: No cross-sequence attention leakage when using proper batch separation.
4. **Shared prefix bonus**: System prompt KV can be computed once and referenced by all sequences, saving additional VRAM.

---

## 4. Recommendation

- **SIG-MultiSeq is the recommended architecture** for any edge deployment serving ≥2 concurrent agent tasks.
- **Round-robin scheduling** with configurable time slices provides natural fairness.
- **Priority preemption**: High-priority sequences (e.g., safety-critical cooking alerts) can interrupt low-priority ones without state loss.

---

## 5. Future Work

- Dynamic sequence creation/destruction for ephemeral agents
- Cross-sequence KV fragment transfer (one agent's discovered knowledge shared to others)
- Integration with R19 for distributed multi-sequence across edge devices
- Admission control policies when VRAM budget is exhausted
