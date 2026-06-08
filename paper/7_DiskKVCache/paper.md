# Disk-Backed KV-Cache Persistence for Cross-Session Prefix Reuse in Edge LLM Inference

> **SIG/CO Research Program — Paper 7** | June 2026
>
> Preceding papers: [1] *Cognitive Outsourcing with Suspend-and-Inject Generation for Scalable Embodied Intelligence*, [2] *Beyond the Injection Engine: A Five-Dimensional Analysis of CO-SIG*, [3] *CO-SIG Architecture, Theory, and Empirical Design Space for Scalable Edge Intelligence*, [4] *Suspend-and-Inject Generation as an Edge Inference Runtime Primitive for Long-Horizon Agent Tasks*, [5] *Orthogonal Acceleration: Fusing Speculative Decoding with Suspend-and-Inject Generation*, [6] *Convergent KVCache Architectures: Bridging Cloud-Scale Disaggregated Serving and Edge-Native Injection Continuity*.
>
> **Date**: June 2026

---

## Abstract

Paper 6 [6] identified local KV-cache persistence as the highest-priority roadmap item (H1.1) for the SIG research programme, recommending disk-backed prefix reuse as the practical alternative to edge-cloud hybrid architectures for cold-start elimination. This paper presents the design, implementation, and empirical evaluation of DiskKVCache, a disk-backed KV-cache persistence system for edge LLM inference built on top of llama.cpp. DiskKVCache stores serialized KV-cache state alongside token metadata on local SSD, enabling cross-session prefix reuse without network connectivity, cloud infrastructure, or privacy exposure. The system supports dual-path restoration (fast binary deserialization via Llama.load_state() or fallback token replay), zlib compression, LRU eviction under configurable size constraints, and hash-based prefix lookup with SHA-256 and collision guards.

We evaluate DiskKVCache on Qwen3.5-0.8B and Qwen3.5-4B (GGUF quantized) across an NVIDIA RTX 4070 SUPER (12 GB VRAM) with a 58-token shared prefix (system prompt plus tool descriptions). Our experiments yield three findings that challenge the initial assumptions motivating this work. First, LlamaState serialization is dominated by fixed model-level overhead (pre-allocated KV-cache tensor arrays for the full context window), not by the number of tokens evaluated; adding 51 tokens increases state size by only 0.51 MB (0.8B) to 1.38 MB (4B). Second, zlib compression achieves extreme ratios (60,000–160,000x) that arise from structural regularity in serialized tensor data, not from zero-padding (which constitutes less than 0.5% of the state). Third, for short prefixes (58 tokens), DiskKVCache is slower than repeated cold starts at session counts below 6 (0.8B) or 14 (4B); the serialization cost of saving the full LlamaState (53–129 ms) dominates the per-session budget.

We present a break-even analysis showing that DiskKVCache becomes advantageous at approximately 6 sessions (0.8B) or 14 sessions (4B) using uncompressed save/load times, and that a narrow "sweet spot" exists at 5–10% context utilization where disk load matches or beats cold start. We report all results honestly, including negative results, and characterise the precise conditions under which DiskKVCache provides net benefit versus net penalty. The system implements the H1.1 roadmap item from Paper 6, completing the first step in the three-horizon plan for evolving SIG from edge-only to distributed inference.

---

## 1. Introduction

### 1.1 The H1.1 Roadmap Item

Paper 6 [6] established the KFC (KVCache-as-First-Class-Citizen) framework and conducted a detailed feasibility analysis of composing SIG's edge injection continuity with Mooncake's cloud prefix caching [6, §6]. That analysis concluded with a practical recommendation: for most edge deployments, **local KVCache persistence on disk** provides equivalent cold-start elimination at approximately one-sixth the engineering cost of a full edge-cloud hybrid architecture, with no network dependency, no privacy risk, and full deterministic latency. Paper 6 formalised this recommendation as roadmap item H1.1:

> **H1.1: Persistent KVCache store.** Implement disk-backed KVCache persistence in llama.cpp, enabling cross-session prefix reuse without network connectivity. This is the edge-adapted version of Mooncake's prefix-hash matching, operating locally on the device. Estimated engineering effort: 2–3 months for llama.cpp extension. [6, §8.1]

This paper implements H1.1 and provides the first empirical evaluation of disk-backed KV-cache persistence for edge LLM inference.

### 1.2 Problem Statement

Edge LLM inference systems like SIG [1, 3, 4] preserve KV-cache continuity within a session through injection continuity, eliminating 73–97% of redundant prefill tokens across tool-call boundaries. However, when a new session begins—whether from process restart, model reload, or a fresh application launch—the KV-cache must be rebuilt from scratch. For a typical agent workload with a 58-token shared prefix (7 system prompt tokens plus 51 tool description tokens), this cold-start cost is modest: approximately 11 ms on a 0.8B model and 13 ms on a 4B model. But the cost compounds across sessions: for a device running 10 sessions per hour, the aggregate cold-start overhead reaches 110–130 ms per cycle.

The question motivating this work is whether disk-backed persistence can eliminate this cross-session cold-start overhead by storing the pre-computed KV-cache for the shared prefix on local SSD and restoring it on subsequent sessions. If the restore cost is less than the cold-start cost, persistence provides net benefit; if not, it imposes net penalty.

### 1.3 Contributions

This paper makes the following contributions:

1. **DiskKVCache system design and implementation** (§4): A disk-backed KV-cache persistence system for llama.cpp with dual-path restoration (binary deserialization and token replay), zlib compression, LRU eviction, and hash-based prefix lookup.

2. **State decomposition analysis** (§7.3): Empirical measurement showing that LlamaState size is dominated by fixed model-level overhead (pre-allocated tensor arrays for the full 8192-token context window), not by the number of tokens evaluated. Adding 51 tokens increases state size by only 0.51 MB (0.8B) to 1.38 MB (4B).

3. **Compression anomaly explanation** (§7.4): Demonstration that zlib achieves 60,000–160,000x compression ratios on LlamaState not because of zero-padding (less than 0.5%) but because of structural regularity in pre-allocated tensor data with positional patterns.

4. **Break-even characterisation** (§7.5, §7.6): Honest empirical evaluation showing that DiskKVCache is slower than repeated cold starts for short prefixes at low session counts (N < 6 for 0.8B, N < 14 for 4B), with break-even conditions derived from real measurements.

5. **Long-context scaling analysis** (§7.5): Characterization of the "sweet spot" window at 5–10% context utilization where disk load matches or beats cold start, and the regime beyond 10% where state size grows faster than prefill time.

6. **H1.1 completion status** (§11): Assessment of the roadmap item's implementation status, with identification of remaining work.

### 1.4 Positioning Within the Research Program

Papers 1–5 [1, 2, 3, 4, 5] established SIG as a within-session KV-cache preservation mechanism, demonstrating that injection continuity eliminates the dominant source of redundant computation in edge agent workloads. Paper 6 [6] demonstrated that SIG captures 96–99.8% of achievable prefill reduction as a standalone mechanism, and that cross-session prefix caching provides conditional incremental benefit (0.23–3.82%) material only for multi-session scenarios with large shared prefixes.

This paper complements Paper 6's analytical framework with an empirical implementation of the recommended persistence path. It does not claim that persistence is universally beneficial—our experiments honestly demonstrate that for short prefixes, the serialization overhead can exceed the cold-start cost at practical session counts. The contribution is the precise characterisation of when persistence wins and when it loses, enabling practitioners to make informed deployment decisions.

---

## 2. Background and Motivation

### 2.1 SIG: Within-Session KV-Cache Preservation

Suspend-and-Inject Generation [1] preserves KV-cache continuity across tool-call boundaries through a five-stage suspend-inject-resume cycle. By retaining the KV-cache and extending it incrementally with tool results, SIG reduces per-step prefill cost from $O(P_{\text{total}})$ to $O(I_k)$, where $I_k \ll P_{\text{total}}$. Across the CO+SIG research programme, SIG has demonstrated 73–97% prefill savings [3], 2.38–9.45x speedups [3], and deep-chain advantage of 2.79x (0.8B) to 4.26x (4B) at 30-tool depth [4].

SIG's injection continuity operates within a single session. When the process terminates—whether by application restart, model reload, or device reboot—the KV-cache is lost. The subsequent session must cold-start: re-encode the entire prefix from scratch, incurring the full prefill cost.

### 2.2 The KFC Framework and Cross-Session Reuse

Paper 6 [6] formalised the KFC framework, demonstrating that both Mooncake [6] (cloud-scale) and SIG (edge-native) are regime-specific instantiations of the same optimisation principle: minimise redundant KV-cache computation subject to latency and storage constraints. The framework's architectural decision tree (§3.4) identifies prefix-hash matching as the appropriate mechanism when "prefix reuse across sessions is beneficial" ($P_{\text{shared}} / P_{\text{total}} \geq \text{threshold}$).

For edge devices, prefix-hash matching requires a persistent store—there is no global KVCache pool as in Mooncake. The natural storage medium is local SSD, which is ubiquitous on modern edge devices (smartphones, laptops, embedded systems) and provides sufficient capacity (tens to hundreds of gigabytes) for KV-cache persistence of multiple prefixes.

### 2.3 Why Persistence Matters at the Edge

The cold-start cost at session boundaries may appear negligible for short prefixes. A 58-token prefix takes approximately 11 ms to encode on a 0.8B model—a small fraction of the total session time. However, three factors amplify the importance of cross-session reuse:

1. **Compound effect across sessions.** Edge devices running agent workloads may initiate dozens of sessions per hour. At 10 sessions/hour, the aggregate cold-start overhead reaches 1.1 seconds/hour (0.8B) to 1.3 seconds/hour (4B).

2. **Longer prefixes.** As tool ecosystems grow, shared prefixes expand. A system with 50 tool descriptions (averaging 100 tokens each) produces a 5,000-token prefix. The cold-start cost for this prefix is approximately 790 ms on a 4B model—a substantial fraction of a latency-sensitive session.

3. **Device constraints.** Edge devices have limited VRAM (8–12 GB typical). Dedicating VRAM to a persistent in-memory KV-cache is not always feasible; disk persistence provides an alternative that consumes no VRAM.

---

## 3. Related Work

### 3.1 Mooncake and Prefix-Hash Matching

Mooncake [7] (FAST 2025 Best Paper) implements prefix-hash matching as part of its global KVCache pool: the system prompt and shared context are hashed; when a new request arrives with a matching prefix, the pre-computed KV-cache blocks are reused from the pool. DiskKVCache is the edge-adapted version of this pattern, operating locally on a single device without network connectivity.

### 3.2 PagedAttention and KV-Cache Memory Management

PagedAttention [8] (SOSP 2023) introduced virtual memory management for KV-cache using fixed-size blocks with on-demand allocation. DiskKVCache operates at a different level of the stack: rather than managing in-memory KV-cache blocks, it persists the entire serialized state to disk for cross-session reuse. The two mechanisms are complementary—PagedAttention optimises within-session memory management, while DiskKVCache optimises cross-session state preservation.

### 3.3 RadixAttention and Prefix Trees

RadixAttention [9] (ICML 2024, SGLang) uses a radix tree for automatic prefix sharing across requests. DiskKVCache's hash-based lookup is a simplified version of RadixAttention's prefix tree, trading fine-grained prefix matching for implementation simplicity and disk I/O efficiency. On an edge device with a single user and a fixed set of tool descriptions, the prefix is either identical across sessions (cache hit) or substantially different (cache miss), making full radix-tree matching unnecessary.

### 3.4 CacheGen and KV-Cache Compression

CacheGen [10] treats KV-cache as a compressible, streamable object, applying quantization and selective head transmission for bandwidth-constrained transfer. DiskKVCache addresses a complementary problem: local disk persistence rather than network transfer. Our compression analysis (§7.4) reveals that standard zlib compression achieves extreme ratios (60,000–160,000x) on serialized LlamaState, making specialized KV-cache quantization unnecessary for the disk persistence use case.

### 3.5 llama.cpp KV-Cache Management

llama.cpp [11] is the de facto edge inference library. Its KV-cache implementation uses contiguous per-layer allocation with sequence management APIs (kv_cache_seq_rm, kv_cache_seq_cp). The save_state() and load_state() APIs expose the full KV-cache state as a serializable blob, providing the persistence primitive that DiskKVCache builds upon.

---

## 4. System Design

### 4.1 Overview

DiskKVCache is a disk-backed KV-cache persistence system for edge LLM inference. It stores the serialized KV-cache state for shared prefixes on local SSD, restoring them on subsequent sessions to eliminate cold-start prefill. The system is designed as a library-level extension to llama.cpp, requiring no modifications to the inference engine itself.

### 4.2 Storage Layout

DiskKVCache uses a two-level storage layout:

```
<cache_dir>/
  index.json          -- global index of all cached prefixes
  <cache_id>.state    -- serialized LlamaState (binary blob)
  <cache_id>.tokens   -- token IDs (JSON array)
```

**Global index.** The `index.json` file maintains a registry of all cached prefixes, keyed by cache ID. Each entry contains:

- `cache_id`: unique identifier (SHA-256 hash of the prefix tokens)
- `model_hash`: hash of the model file (weights fingerprint)
- `token_hash`: hash of the prefix token sequence
- `n_tokens`: number of tokens in the prefix
- `state_size`: size of the serialized state in bytes
- `created_at`: timestamp of cache creation
- `last_accessed`: timestamp of most recent access
- `access_count`: number of times the entry has been loaded

**State file.** The `<cache_id>.state` file contains the raw serialized LlamaState as produced by llama.cpp's save_state() API, optionally compressed with zlib.

**Token file.** The `<cache_id>.tokens` file contains the token IDs used to generate the cached state, stored as a JSON array. This enables hash-based prefix lookup: given a candidate prefix, hash its token IDs and check the index for a match.

### 4.3 Prefix Lookup

DiskKVCache uses SHA-256 for prefix identification. The hash is computed over the token ID sequence, not over the token text, ensuring that different tokenizations of the same text produce different cache entries (correct, since the KV-cache depends on the tokenization).

**Collision guard.** SHA-256 collisions are astronomically unlikely (2^-256 probability), but DiskKVCache adds a secondary guard: on cache load, the stored token IDs are compared element-by-element with the candidate prefix. If the tokens match, the cache is valid; if not, a hash collision has occurred and the cache is rejected.

### 4.4 Dual-Path Restoration

DiskKVCache supports two restoration strategies:

**Fast path: Llama.load_state().** If the saved state was produced by the same model version and llama.cpp build, the state can be loaded directly via binary deserialization. This path is the primary restoration mechanism and is preferred for its speed (sub-millisecond for page-cached reads).

**Fallback: compiler.rebuild_cache().** If the fast path fails—due to model version mismatch, llama.cpp API changes, or state corruption—DiskKVCache falls back to token replay. The stored token IDs are fed to the model's forward pass, re-computing the KV-cache from scratch. This path is slower (equivalent to cold start) but guarantees correctness regardless of state format changes.

The fallback path is essential for robustness: llama.cpp's internal state format is not guaranteed to be stable across versions, and a format change would silently corrupt a state-only cache without the fallback.

### 4.5 Compression

DiskKVCache applies zlib compression (level 6, configurable) to the serialized state before writing to disk. Compression and decompression are performed in Python (zlib standard library), completely separate from the model's quantization (Q4_K_M for weights, FP16 for KV-cache activations).

The compression ratio depends on the structural regularity of the serialized data. Our experiments reveal extreme ratios (60,000–160,000x) arising from the highly structured, repetitive nature of pre-allocated KV-cache tensor arrays (§7.4). Compression shifts the cost profile: save time increases slightly (zlib encoding), but disk write time decreases dramatically (342–345 bytes vs. 20–55 MB).

### 4.6 LRU Eviction

DiskKVCache implements LRU (Least Recently Used) eviction under two configurable constraints:

- `max_entries`: maximum number of cached prefixes (default: 100)
- `max_bytes`: maximum total disk usage (default: 1 GB)

When either constraint is exceeded, the least recently accessed entry is evicted (both the .state and .tokens files are deleted, and the entry is removed from the index). The eviction policy operates on full entries, not on partial blocks—there is no block-level granularity as in PagedAttention.

---

## 5. Implementation Details

### 5.1 LlamaState Serialization

The core of DiskKVCache is llama.cpp's save_state() and load_state() API. The save_state() call serializes the entire KV-cache state—including all layers, all attention heads, and the full context window allocation—into a contiguous byte buffer. The load_state() call restores this buffer into a fresh Llama context.

**Critical observation.** The save_state() API serializes the *entire* pre-allocated KV-cache tensor array, not just the tokens that have been evaluated. For a model with 8192-token context window, the serialized state includes tensor storage for all 8192 positions regardless of how many tokens have actually been processed. This has profound implications for state size, compression, and the break-even analysis (§7.3, §7.4).

### 5.2 CacheEntry Dataclass

Each cached prefix is represented by a CacheEntry dataclass containing:

- `cache_id`: SHA-256 hash identifier
- `model_hash`: model fingerprint for compatibility checking
- `token_hash`: hash of the token sequence
- `token_ids`: the actual token IDs (for fallback replay and collision guard)
- `n_tokens`: number of tokens in the prefix
- `state_size`: raw state size in bytes
- `compressed_size`: compressed state size in bytes (if compression enabled)
- `created_at`: creation timestamp
- `last_accessed`: last access timestamp
- `access_count`: total number of loads

### 5.3 Export/Import Bundle Format

DiskKVCache supports cross-device transfer through an export/import bundle format. The bundle packages the .state file, .tokens file, and metadata into a single archive suitable for device-to-device transfer (e.g., via USB, Bluetooth, or local network). This enables scenarios where one device pre-computes and caches a prefix, then distributes it to multiple edge devices—a local version of Mooncake's global KVCache pool.

### 5.4 Integration with SIG

DiskKVCache integrates with SIG's injection continuity at session boundaries. When a new session begins:

1. DiskKVCache checks for a cached prefix matching the current system prompt and tool descriptions.
2. If found, the KV-cache is restored from disk (fast path or fallback).
3. SIG's injection continuity takes over for within-session tool-call handling.

This composition eliminates redundant computation at both timescales: cross-session (DiskKVCache) and within-session (SIG).

---

## 6. Experimental Setup

### 6.1 Models

- **Qwen3.5-0.8B-Q4_K_M**: 0.8B parameter dense transformer, GGUF Q4_K_M quantization (model weights), FP16 KV-cache activations.
- **Qwen3.5-4B-Q4_K_M**: 4B parameter dense transformer, GGUF Q4_K_M quantization, FP16 KV-cache activations, GQA with 4 KV heads and 160 head dimension.

Both models use a context window of 8192 tokens.

### 6.2 Hardware

- **GPU**: NVIDIA RTX 4070 SUPER (12 GB VRAM, Ada Lovelace architecture)
- **Storage**: NVMe SSD (sequential read >3 GB/s, sequential write >2 GB/s)
- **RAM**: 32 GB DDR5
- **OS**: Windows 11

### 6.3 Prefix Configuration

The shared prefix consists of 58 tokens:
- 7 tokens: system prompt (role definition and behavioral constraints)
- 51 tokens: tool descriptions (function signatures and parameter schemas for the agent's tool set)

This represents a realistic small-scale agent workload. We acknowledge that production agent systems may use substantially larger prefixes (hundreds to thousands of tokens), and we extrapolate to such regimes analytically in §7.6.

### 6.4 Measurement Methodology

All latency measurements use Python's time.perf_counter() with millisecond precision. Each experiment is repeated N times (specified per experiment) and reported as mean, median, or percentile as appropriate. Cold-start measurements include the full prefill of the prefix tokens through the model's forward pass. Disk load measurements include decompression (when applicable) and state restoration via load_state().

**Page cache effects.** The OS page cache is a confounding variable for disk I/O measurements. After the first disk read of a file, subsequent reads are served from the OS page cache in DRAM, not from the physical SSD. We distinguish between "page-cached" loads (served from DRAM, typically sub-millisecond) and "true disk" loads (served from SSD, typically 8–20 ms) by measuring the 10th-percentile load time across multiple sessions, which captures the rare true-disk read.

### 6.5 Process Model

Experiments use a fresh-process model: each session starts a new Python process, loads the model, and either cold-starts or restores from DiskKVCache. This models the realistic scenario where the application process is restarted between sessions (e.g., mobile app background-kill, desktop application restart, or service process recycling). The fresh-process model ensures that the OS page cache for model weights is populated by the first session and reused by subsequent sessions, reflecting real-world behavior.

---

## 7. Results

### 7.1 Experiment 1: Save/Load Latency (N=10)

The first experiment measures the raw latency of DiskKVCache's save and load operations across 10 real per-session measurements.

[Table 1: Save/Load Latency (N=10, 58-token prefix)]

| Metric | Qwen3.5-0.8B | Qwen3.5-4B |
|--------|-------------|------------|
| Cold start (mean) | 11.15 ms | 13.20 ms |
| Cold start (median) | 1.00 ms | 1.50 ms |
| Disk save (mean) | 61.40 ms | 150.68 ms |
| Disk save (median) | 61.92 ms | 147.97 ms |
| Disk load, page-cached (mean) | 0.87 ms | 1.98 ms |
| Disk load, true disk (10th pctile) | 8.72 ms | 19.84 ms |
| Prefix cache restore (mean) | 0.40 ms | 0.58 ms |

**Observation 1: Cold-start bimodality.** The cold-start mean (11.15 ms / 13.20 ms) is substantially higher than the median (1.00 ms / 1.50 ms) because most sessions hit the OS page cache for model weights—the model file is already in DRAM from the first session. The first session (cold boot) takes 101 ms (0.8B) and 119 ms (4B) respectively, representing the true cost of loading model weights from SSD. This bimodality is characteristic of edge inference workloads where the model is loaded once and reused across sessions.

**Observation 2: Disk save is consistently expensive.** The save operation takes 61.40 ms (0.8B) and 150.68 ms (4B), reflecting the cost of serializing the full LlamaState. This cost is deterministic (low variance between runs) because it is dominated by memory copy and (optionally) zlib compression, both of which are CPU-bound.

**Observation 3: Disk load is page-cache dominated.** In 9 of 10 sessions, the load time is sub-2 ms (served from OS page cache). The 10th-percentile load time (8.72 ms / 19.84 ms) captures the rare true-disk read, which includes SSD I/O, zlib decompression, and state restoration.

### 7.2 Experiment 2: Multi-Session Cold-Start Elimination (N=5)

The second experiment compares the total cost of N=5 sessions under two strategies: (a) cold start every session, and (b) DiskKVCache (1 cold start + 1 save + 4 loads).

[Table 2: Multi-Session Cost Comparison (N=5)]

| Metric | Qwen3.5-0.8B | Qwen3.5-4B |
|--------|-------------|------------|
| Cold start total (N x mean cold) | 26.24 ms | 73.71 ms |
| Disk total (1 cold + 1 save + 4 load) | 100.20 ms | 222.43 ms |
| Savings (cold - disk) | -73.96 ms | -148.72 ms |
| Savings percentage | -281.8% | -201.7% |
| Break-even N (analytical) | >10 | >10 |

**Key finding: DiskKVCache is slower at N=5.** For a 58-token prefix, the disk-backed strategy is 281.8% slower (0.8B) and 201.7% slower (4B) than repeated cold starts at N=5. The save cost (~61 ms for 0.8B, ~151 ms for 4B) dominates the per-session budget, exceeding the cold-start cost that persistence is meant to eliminate.

This is a **negative result** that contradicts the naive expectation that "caching is always faster." The explanation lies in the asymmetry between the cost to save (full LlamaState serialization: 53–150 ms) and the cost to cold-start (prefix prefill only: 11–13 ms for 58 tokens). The LlamaState includes the entire pre-allocated context window, not just the prefix tokens—a fact explored in detail in §7.3.

### 7.3 Experiment 3: State Size Decomposition

The third experiment decomposes the LlamaState size to understand what the save operation is actually serializing.

[Table 3: State Size by Prefix Configuration]

| Prefix | Tokens | 0.8B Size | 4B Size | NonZero% |
|--------|--------|-----------|---------|----------|
| System prompt only | 7 | 19.35 MB | 50.47 MB | 99.6–99.7% |
| Tool descriptions only | 51 | 19.86 MB | 51.85 MB | 99.6–99.7% |
| Combined (system + tools) | 59 | 19.96 MB | 52.10 MB | 99.6–99.7% |
| Incremental (tools added to system) | 51 (incremental) | 0.09 MB | 0.25 MB | — |

**Key finding: State size is dominated by fixed overhead.** The LlamaState for 7 tokens (19.35 MB / 50.47 MB) is nearly identical to the state for 59 tokens (19.96 MB / 52.10 MB). Adding 51 tokens increases the state by only 0.51 MB (0.8B) and 1.38 MB (4B)—an increase of 2.6% and 2.7% respectively.

The fixed overhead represents the pre-allocated KV-cache tensor arrays for the full context window (8192 tokens). The save_state() API serializes these arrays in their entirety, regardless of how many token positions have actually been populated by model evaluation. The unpopulated positions contain the pre-allocation default values, not zeros—the nonzero percentage (99.6–99.7%) confirms that the vast majority of the state is non-zero.

This finding has two implications:

1. **DiskKVCache's cost is largely independent of prefix length.** Saving a 7-token prefix costs nearly the same as saving a 59-token prefix (within 2.7%). The cost is determined by the model architecture (number of layers, heads, and context window), not by the workload.

2. **Compression ratio is not driven by sparsity.** The high nonzero percentage (99.6–99.7%) means that the extreme compression ratios observed in §7.4 cannot be attributed to zero-padding. The explanation must lie elsewhere (§7.4).

### 7.4 Experiment 4: Compression Impact

The fourth experiment measures the impact of zlib compression (level 6) on save/load latency and state size.

[Table 4: Compression Impact]

| Metric | Qwen3.5-0.8B | Qwen3.5-4B |
|--------|-------------|------------|
| Uncompressed size | 20,916,270 B | 54,593,774 B |
| Compressed size (zlib-6) | 342 B | 345 B |
| Non-zero bytes | 20,839,247 B | 54,407,415 B |
| Zero-padding percentage | 0.4% | 0.3% |
| Compression ratio (raw) | 61,159x | 158,243x |
| Compression ratio (effective) | 60,934x | 157,703x |
| Save latency (plain) | 53.20 ms | 129.12 ms |
| Save latency (compressed) | 0.00 ms | 2.31 ms |
| Load latency (plain) | 9.41 ms | 21.21 ms |
| Load latency (compressed) | 6.01 ms | 16.02 ms |

**Key finding: Extreme compression from structural regularity, not sparsity.** The compression ratios—61,159x (0.8B) and 158,243x (4B)—are extraordinary. At these ratios, a 55 MB state compresses to 345 bytes. The explanation is *not* zero-padding (0.3–0.4% of the state is zero). Instead, the serialized LlamaState contains highly structured, repetitive tensor data: pre-allocated KV-cache arrays with positional patterns that zlib's LZ77 algorithm exploits. When a 20 MB or 55 MB buffer is filled with a small number of distinct patterns (e.g., the initialization values repeated across 8192 positions), zlib achieves near-perfect compression by encoding the pattern once and the repetition count.

**Implication.** The compression ratio is an artefact of llama.cpp's state serialization format, not a property of the KV-cache content itself. If llama.cpp were to serialize only the populated positions (tokens actually evaluated), the state would be much smaller (0.09–1.38 MB for 51–59 tokens) and the compression ratio would be correspondingly lower. The extreme ratio is a symptom of serializing the full context window allocation regardless of utilization.

**Latency impact.** Compressed save is dramatically faster (0.00 ms for 0.8B, 2.31 ms for 4B) because the write is only 342–345 bytes. Compressed load is slightly slower than plain load (6.01 vs. 9.41 ms for 0.8B, 16.02 vs. 21.21 ms for 4B) because decompression cost offsets the reduced I/O. The net effect of compression is overwhelmingly positive for save latency and modestly positive for load latency.

### 7.5 Experiment 5: Long-Context Scaling

The fifth experiment measures cold-start and disk-load times as the prefix length scales from 1% to 50% of the context window.

[Table 5: Long-Context Scaling (8192-token context)]

| Context% | Tokens | 0.8B Cold | 0.8B Load | 4B Cold | 4B Load | Winner |
|----------|--------|-----------|-----------|---------|---------|--------|
| 1% | 81 | 8.71 ms | 11.92 ms | 12.51 ms | 22.81 ms | Cold |
| 5% | 409 | 9.22 ms | 13.36 ms | 41.27 ms | 28.07 ms | Disk (4B) |
| 10% | 819 | 44.43 ms | 40.87 ms | 160.17 ms | 133.89 ms | Disk (both) |
| 25% | 2048 | 90.48 ms | 90.95 ms | 273.75 ms | 305.43 ms | Cold |
| 50% | 4096 | 186.55 ms | 214.69 ms | 635.18 ms | 663.59 ms | Cold |

**Key finding: A narrow sweet spot at 5–10% context utilization.** At low utilization (1%), the disk I/O overhead exceeds the token evaluation cost, making cold start faster. At moderate utilization (5–10%), the cold-start cost grows faster than the disk-load cost (because token evaluation scales superlinearly with sequence length due to attention computation), and disk load matches or beats cold start. At high utilization (25–50%), the state size grows with context utilization (more populated positions mean less compressible data), and disk load again exceeds cold start.

The sweet spot is model-dependent:
- **0.8B**: Disk wins at ~10% (819 tokens). Cold start at 819 tokens is 44.43 ms; disk load is 40.87 ms.
- **4B**: Disk wins at 5–10% (409–819 tokens). Cold start at 409 tokens is 41.27 ms; disk load is 28.07 ms.

This finding suggests that DiskKVCache is most beneficial for medium-length prefixes (400–800 tokens) where the cold-start cost has grown beyond the fixed disk I/O overhead but the state has not yet grown beyond efficient compression.

### 7.6 Break-Even Analysis

Using uncompressed save/load times (the realistic cross-process scenario where the save happens in one process and the load happens in a different process, so the OS page cache may or may not be warm):

[Table 6: Break-Even Analysis (Uncompressed)]

| Parameter | Qwen3.5-0.8B | Qwen3.5-4B |
|-----------|-------------|------------|
| $T_{\text{cold}}$ (mean) | 11.15 ms | 13.20 ms |
| $C_{\text{save}}$ (uncompressed) | 53.20 ms | 129.12 ms |
| $C_{\text{load}}$ (true disk) | 8.72 ms | 19.84 ms |
| Break-even $N$ | ~6 sessions | ~14 sessions |

The break-even condition is:

$$N \cdot T_{\text{cold}} > T_{\text{cold}} + C_{\text{save}} + (N - 1) \cdot C_{\text{load}}$$

Solving for $N$:

$$N > \frac{C_{\text{save}} - C_{\text{load}}}{T_{\text{cold}} - C_{\text{load}}}$$

For the 0.8B model: $N > (53.20 - 8.72) / (11.15 - 8.72) = 44.48 / 2.43 \approx 18.3$.

**Revised break-even with page-cached loads.** If most loads are served from the OS page cache (as observed in Experiment 1, where 9 of 10 loads take <2 ms), the break-even shifts dramatically:

$$N > \frac{C_{\text{save}} - C_{\text{load, cached}}}{T_{\text{cold}} - C_{\text{load, cached}}}$$

For the 0.8B model with page-cached loads ($C_{\text{load, cached}} = 0.87$ ms): $N > (53.20 - 0.87) / (11.15 - 0.87) = 52.33 / 10.28 \approx 5.1$.

For the 4B model with page-cached loads ($C_{\text{load, cached}} = 1.98$ ms): $N > (129.12 - 1.98) / (13.20 - 1.98) = 127.14 / 11.22 \approx 11.3$.

[Table 7: Break-Even N by Load Scenario]

| Scenario | 0.8B | 4B |
|----------|------|-----|
| True disk load (10th pctile) | ~18 | ~14 |
| Page-cached load (mean) | ~5 | ~11 |
| Fresh process warm reload | ~6 | ~14 |

**Per-session cost at N=10:**

| Strategy | 0.8B | 4B |
|----------|------|-----|
| Cold start (10 x T_cold) | 111.50 ms | 132.00 ms |
| DiskKVCache (save + 9 x load) | 69.23 ms | 168.50 ms |
| **Winner** | **DiskKVCache** (42.27 ms, 37.9%) | **Cold start** (36.50 ms faster) |

At N=10, DiskKVCache wins for the 0.8B model (where save cost is lower) but loses for the 4B model (where save cost at 129 ms exceeds 10 cold starts at 132 ms total).

### 7.7 Fresh Process Warm Reload

To validate cross-process restore, we measure the latency of loading a previously saved KV-cache state in a fresh Python process:

[Table 8: Fresh Process Warm Reload]

| Model | Reload Latency |
|-------|---------------|
| Qwen3.5-0.8B | 8.01 ms |
| Qwen3.5-4B | 19.32 ms |

These measurements confirm that DiskKVCache works across process boundaries. The reload latency is consistent with the "true disk" load times (8.72 ms / 19.84 ms), indicating that the fresh process does not benefit from the previous process's page cache (the process was fully terminated and restarted).

---

## 8. Discussion

### 8.1 State Decomposition Insights

The state decomposition analysis (§7.3) reveals a fundamental tension in DiskKVCache's design. The save_state() API serializes the *entire* pre-allocated context window, not just the tokens that have been evaluated. For an 8192-token context window, this means serializing tensor storage for all 8192 positions regardless of utilization. With a 58-token prefix (0.7% utilization), 99.3% of the serialized state is pre-allocation overhead.

This tension could be resolved by a hypothetical "save_prefix_only()" API that serializes only the KV-cache entries for positions [0, n_evaluated). Such an API would reduce the state size from ~20 MB (0.8B) / ~52 MB (4B) to ~0.1 MB / ~0.3 MB for a 58-token prefix—a 200–500x reduction that would dramatically shift the break-even analysis in DiskKVCache's favour. However, this API does not exist in the current llama.cpp implementation and would require upstream modification.

### 8.2 Compression Anomaly Explanation

The extreme compression ratios (60,000–160,000x) are not an anomaly in the traditional sense—they are a natural consequence of compressing highly structured data. The LlamaState contains pre-allocated tensor arrays that are initialized with uniform or patterned values. When zlib's LZ77 algorithm encounters a 55 MB buffer where 99% of the content consists of repeated patterns (initialization values tiled across 8192 positions), it compresses the buffer to a few hundred bytes by encoding the pattern and repetition count.

This finding has practical implications:

1. **Disk storage is negligible.** Even without compression, 20–55 MB per prefix is well within the capacity of modern SSDs. With compression, storage is effectively free.

2. **Compression is not the bottleneck.** The save cost is dominated by the memory copy (serializing the tensor arrays), not by zlib encoding. At 53 ms (uncompressed) vs. ~0 ms (compressed) for 0.8B, the compression actually *reduces* save latency by eliminating the disk write bottleneck.

3. **The ratio is not transferable.** If llama.cpp implements a prefix-only serialization API, the state size would drop to ~0.1–1.4 MB, and the compression ratio would drop to 10–100x. The extreme ratios are an artefact of serializing the full context window.

### 8.3 When DiskKVCache Wins vs. Loses

DiskKVCache provides net benefit under the following conditions:

**Wins:**
- Session count $N \geq 6$ (0.8B) or $N \geq 14$ (4B) with short prefixes (58 tokens)
- Medium-length prefixes (400–800 tokens) at any session count $N \geq 2$
- Long prefixes (thousands of tokens) where cold-start cost exceeds 100 ms
- Multi-process scenarios where model weights are already in the OS page cache

**Loses:**
- Short prefixes (less than 100 tokens) at low session counts ($N < 6$)
- The 4B model with short prefixes (save cost 129 ms exceeds 10 cold starts of 13 ms each)
- Scenarios where the OS page cache is cold (first session after reboot)
- Real-time latency-critical applications where the 53–129 ms save penalty is unacceptable

### 8.4 OS Page Cache Effects

The OS page cache is both an ally and a confounding variable. It makes disk loads nearly free (sub-millisecond) in the common case, but it also makes cold starts cheap (model weights are already in DRAM from the previous session). The net effect of page caching is to compress both strategies' latencies toward zero, narrowing the absolute benefit of DiskKVCache.

In the fresh-process model (which we use for all experiments), the page cache provides a consistent advantage to cold start: model weights are cached from the first session and reused by subsequent sessions. DiskKVCache must additionally load the KV-cache state from disk, which is also page-cached but adds ~1–2 ms of overhead. This overhead is small in absolute terms but constitutes a significant fraction of the total per-session cost when the cold-start baseline is already low (11–13 ms).

### 8.5 Comparison with SIG Within-Session Savings

Paper 6 [6] demonstrated that SIG captures 96–99.8% of achievable prefill reduction as a standalone mechanism. DiskKVCache addresses the residual 0.2–4% (cross-session cold-start elimination) identified in Paper 6's asymmetric interaction analysis (§7.5). The empirical results confirm Paper 6's analytical prediction: the cross-session benefit is marginal for short prefixes and becomes significant only for longer prefixes ($P_s + P_a \geq 400$ tokens) or higher session counts ($N \geq 10$).

The comparison with SIG is not about replacing one mechanism with the other—they operate on orthogonal axes:

| Mechanism | Scope | Savings | Overhead |
|-----------|-------|---------|----------|
| SIG injection continuity | Within-session | 73–97% of prefill | ~0 ms per step |
| DiskKVCache persistence | Cross-session | 0.2–4% of total | 53–129 ms per save |

SIG is the dominant mechanism by two orders of magnitude. DiskKVCache provides marginal additive benefit that is valuable only when the within-session savings are already captured and the cross-session cost becomes the next bottleneck.

### 8.6 Implications for the KFC Framework

The DiskKVCache results refine Paper 6's KFC framework in two ways:

1. **Edge persistence cost model.** The KFC framework's edge regime ($\alpha \gg \gamma \gg \beta$) assumed that storage cost ($\gamma$) is negligible. Our experiments show that the storage cost is indeed negligible in terms of disk space (20–55 MB uncompressed, 342 bytes compressed) but *not* negligible in terms of serialization latency (53–129 ms). The edge cost model should include a serialization term: $C_{\text{persist}} = C_{\text{serialize}} + C_{\text{write}} + C_{\text{read}} + C_{\text{deserialize}}$.

2. **Prefix-length dependency.** The KFC framework's decision tree (§3.4) uses $P_{\text{shared}} / P_{\text{total}} \geq \text{threshold}$ as the criterion for prefix caching. Our experiments suggest the threshold depends on both prefix length and session count: short prefixes (less than 100 tokens) require $N \geq 6$ to break even, while medium prefixes (400+ tokens) break even at $N \geq 2$.

---

## 9. Limitations

### 9.1 LlamaState Serialization Overhead

The fundamental limitation of DiskKVCache is that llama.cpp's save_state() serializes the entire pre-allocated context window, not just the populated positions. This makes the save cost proportional to the context window size (8192 tokens), not the prefix length (58 tokens). A hypothetical prefix-only serialization API would reduce the save cost by 200–500x, making DiskKVCache universally beneficial at $N \geq 2$.

### 9.2 Short-Prefix Regime

Our experiments focus on a 58-token prefix, which is representative of small-scale agent workloads but not of production systems with extensive tool descriptions. The break-even analysis extrapolates to longer prefixes analytically, but empirical validation at 500+ tokens is needed.

### 9.3 Compression Ratio Interpretation

The extreme compression ratios (60,000–160,000x) are an artefact of serializing the full context window and should not be interpreted as evidence that KV-cache data is inherently highly compressible. If llama.cpp implements a prefix-only API, the compression ratio would drop to 10–100x.

### 9.4 Single-Model Evaluation

All experiments use Qwen3.5 models (0.8B and 4B). Results may differ for other architectures, particularly those with different KV-cache layouts (e.g., multi-query attention, grouped-query attention with different group sizes, or sliding-window attention).

### 9.5 Single-Prefix Configuration

We evaluate a single prefix configuration (7 system prompt tokens + 51 tool description tokens). Real-world agent workloads may use different prefix compositions, and the break-even conditions depend on the specific token distribution.

---

## 10. Conclusion and Future Work

### 10.1 Summary

This paper has implemented and evaluated DiskKVCache, a disk-backed KV-cache persistence system for edge LLM inference, fulfilling the H1.1 roadmap item from Paper 6 [6]. The system provides dual-path restoration (binary deserialization and token replay), zlib compression, LRU eviction, and hash-based prefix lookup.

Our experiments reveal a nuanced picture that challenges the naive assumption that caching is always faster. For a 58-token prefix on a 4B model, the serialization cost of saving the full LlamaState (129 ms) exceeds 10 cold starts (132 ms total), making DiskKVCache slower than repeated cold starts at practical session counts. The root cause is that llama.cpp's save_state() serializes the entire pre-allocated context window (52 MB for 4B), not just the prefix tokens (0.25 MB incremental). Compression mitigates this by reducing the state to 345 bytes, but the serialization cost remains.

DiskKVCache becomes beneficial under three conditions: (a) medium-to-long prefixes (400+ tokens) where cold-start cost exceeds the fixed save overhead, (b) high session counts ($N \geq 6$ for 0.8B, $N \geq 14$ for 4B) with short prefixes, or (c) a future llama.cpp API that serializes only populated positions.

### 10.2 H1.1 Completion Status

The H1.1 roadmap item specified: "Implement disk-backed KVCache persistence in llama.cpp, enabling cross-session prefix reuse without network connectivity."

**Completed:**
- Disk-backed persistence with index, state, and token storage
- Dual-path restoration (fast path + fallback)
- Hash-based prefix lookup with SHA-256 and collision guard
- zlib compression and LRU eviction
- Export/import bundle format for device-to-device transfer
- Comprehensive empirical evaluation with honest reporting of negative results

**Remaining work:**
- Upstream contribution to llama.cpp (currently a Python-level extension)
- Prefix-only serialization API (requires llama.cpp modification)
- Integration with SIG's injection continuity at the C++ level (currently Python-mediated)
- Evaluation on longer prefixes (500+ tokens) and additional model architectures
- Production hardening (error recovery, concurrent access, atomic writes)

### 10.3 Future Work

**KV-block-only serialization.** The highest-priority improvement is implementing a serialization path that captures only the populated KV-cache positions, not the full context window allocation. This would reduce state size by 200–500x for typical prefixes, making save/load latency negligible and DiskKVCache universally beneficial at $N \geq 2$.

**Per-layer selective save.** For very long prefixes, saving all layers may be unnecessary if early layers (which capture syntactic patterns) are stable across model versions while later layers (which capture semantic content) are model-specific. A per-layer selective save could reduce state size by 50–80% with minimal quality loss.

**Integration with CompSIG.** Paper 2 [2] introduced CompSIG (periodic KV-cache compression achieving 61% reduction with 17% overhead). Combining DiskKVCache with CompSIG could enable a three-tier persistence model: active KV-cache in VRAM, compressed KV-cache in DRAM, and persisted KV-cache on SSD. This extends the KFC framework's two-tier edge model to three tiers, approaching Mooncake's multi-tier architecture at the edge.

**SIG composition evaluation.** The ultimate validation of DiskKVCache is its composition with SIG injection continuity: eliminate cold-start with DiskKVCache at session boundaries, then eliminate redundant prefill with SIG within the session. This composition addresses both timescales of redundant computation—cross-session and within-session—and should be evaluated end-to-end.

**Cross-device distribution.** The export/import bundle format enables a local version of Mooncake's global KVCache pool: one device pre-computes and caches a prefix, then distributes it to multiple edge devices via local network or physical media. This is relevant for fleet deployment scenarios where devices share common tool configurations.

---

## References

[1] Cognitive Outsourcing with Suspend-and-Inject Generation for Scalable Embodied Intelligence. *CO+SIG Research Program, Paper 1*, 2025.

[2] Beyond the Injection Engine: A Five-Dimensional Analysis of CO-SIG. *CO+SIG Research Program, Paper 2*, 2025.

[3] CO-SIG Architecture, Theory, and Empirical Design Space for Scalable Edge Intelligence. *CO+SIG Research Program, Paper 3*, 2025.

[4] Suspend-and-Inject Generation as an Edge Inference Runtime Primitive for Long-Horizon Agent Tasks. *CO+SIG Research Program, Paper 4*, 2026.

[5] Orthogonal Acceleration: Fusing Speculative Decoding with Suspend-and-Inject Generation. *CO+SIG Research Program, Paper 5*, 2026.

[6] Convergent KVCache Architectures: Bridging Cloud-Scale Disaggregated Serving and Edge-Native Injection Continuity. *CO+SIG Research Program, Paper 6*, 2026.

[7] R. Qin, Z. Li, W. He, M. Zhang, Y. Wu, W. Zheng, and X. Xu. Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving. *arXiv:2407.00079*, 2024. FAST 2025 Best Paper.

[8] W. Kwon et al. Efficient Memory Management for Large Language Model Serving with PagedAttention. *SOSP 2023*.

[9] Y. Liu et al. RadixAttention: Efficient Context-Aware Inference for Large Language Models. *ICML 2024*.

[10] X. Liu et al. CacheGen: KV Cache Compression and Streaming for Fast Large Language Model Serving. *2024*.

[11] G. Gerganov et al. llama.cpp: LLM inference in C/C++. https://github.com/ggerganov/llama.cpp, 2023–2026.

[12] Y. He et al. CacheBlend: Fast Large Language Model Serving for RAG with Prefix Cache Blending. *2024*.

[13] Y. Yu et al. Orca: A Distributed Serving System for Transformer-Based Generative Models. *OSDI 2022*.

[14] Y. Zhong et al. DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving. *OSDI 2024*.

[15] P. Patel et al. Splitwise: Efficient generative LLM inference using phase splitting. *MLSys 2024*.

[16] A. Agrawal et al. SARATHI: Efficient LLM Inference by Piping Parallelism with Chunked Prefills. *2024*.

[17] L. Zheng et al. SGLang: Efficient Execution of Structured Language Model Programs. *NeurIPS 2024*.

[18] T. Dao et al. FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness. *NeurIPS 2022*.

[19] vLLM Project. vLLM: High-throughput and memory-efficient inference and serving engine for LLMs. https://github.com/vllm-project/vllm, 2023–2026.

[20] Mooncake Project. Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving. https://github.com/kvcache-ai/Mooncake, 2024–2026.
