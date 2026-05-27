"""InjectionEngine — KV-cache-aware token injection for SIG pipeline.

Extracted from r2_benchmark.py (the more complete version with label
tracking, position_map, eviction, and cache_size property).

Review improvements:
  - In-place cache compaction via kv_cache_seq_shift (avoids costly rebuild)
  - Cache capacity management with automatic eviction
  - Cumulative compute tracking for fair SIG vs AppLoop comparison
  - Batch-aware dependency analysis for automatic batch size selection
"""

import time
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field

from .compiler import MeaningCompiler, SEQ_ID


@dataclass
class CacheStats:
    """Tracks KV-cache resource usage and cumulative compute."""
    total_injected_tokens: int = 0
    total_eval_time: float = 0.0
    total_generate_time: float = 0.0
    injection_count: int = 0
    generate_count: int = 0
    evict_count: int = 0
    compact_count: int = 0
    peak_cache_tokens: int = 0

    @property
    def cumulative_compute(self) -> float:
        return self.total_eval_time + self.total_generate_time

    @property
    def avg_inject_time(self) -> float:
        if self.injection_count == 0:
            return 0.0
        return self.total_eval_time / self.injection_count

    @property
    def avg_generate_time(self) -> float:
        if self.generate_count == 0:
            return 0.0
        return self.total_generate_time / self.generate_count

    def summary(self) -> Dict:
        return {
            "total_injected_tokens": self.total_injected_tokens,
            "total_eval_time_s": round(self.total_eval_time, 4),
            "total_generate_time_s": round(self.total_generate_time, 4),
            "cumulative_compute_s": round(self.cumulative_compute, 4),
            "injection_count": self.injection_count,
            "generate_count": self.generate_count,
            "evict_count": self.evict_count,
            "compact_count": self.compact_count,
            "peak_cache_tokens": self.peak_cache_tokens,
        }


@dataclass
class DependencyNode:
    """Represents a tool call with dependency information."""
    index: int
    tool_name: str
    tool_args: Dict
    depends_on: List[int] = field(default_factory=list)

    @property
    def is_independent(self) -> bool:
        return len(self.depends_on) == 0


class DependencyAnalyzer:
    """Analyzes tool call dependencies for automatic batch sizing.

    Classifies tool calls into independent (parallelizable) and
    sequential (dependent) groups to determine optimal batch size.
    """

    def __init__(self):
        self._nodes: List[DependencyNode] = []

    def add_node(self, tool_name: str, tool_args: Dict,
                 depends_on: Optional[List[int]] = None) -> int:
        idx = len(self._nodes)
        node = DependencyNode(
            index=idx,
            tool_name=tool_name,
            tool_args=tool_args,
            depends_on=depends_on or [],
        )
        self._nodes.append(node)
        return idx

    def classify(self) -> Tuple[List[List[int]], List[int]]:
        """Classify tool calls into independent groups and sequential chain.

        Returns:
            independent_groups: list of groups, each group is a list of indices
                               that can be batched together
            sequential_chain: list of indices that must be executed in order
        """
        independent_groups: List[List[int]] = []
        sequential_chain: List[int] = []

        for node in self._nodes:
            if node.is_independent:
                if not independent_groups:
                    independent_groups.append([node.index])
                else:
                    independent_groups[0].append(node.index)
            else:
                sequential_chain.append(node.index)

        return independent_groups, sequential_chain

    def recommend_batch_size(self, max_batch: int = 8) -> int:
        """Recommend optimal batch size based on dependency analysis.

        Returns batch size that maximizes parallelization while respecting
        dependencies. Returns 1 if all tools are sequential.
        """
        groups, sequential = self.classify()
        if not groups:
            return 1

        independent_count = sum(len(g) for g in groups)
        if independent_count == 0:
            return 1

        return min(independent_count, max_batch)

    def reset(self):
        self._nodes.clear()


class InjectionEngine:
    """Manages token injection into the MeaningCompiler's KV cache while
    tracking cached token IDs, position ranges, and timing metrics.

    Review improvements:
      - In-place compaction via kv_cache_seq_shift (O(shift) vs O(rebuild))
      - Cache capacity management with configurable max tokens
      - Cumulative compute tracking for fair SIG vs AppLoop comparison
      - Dependency analysis for automatic batch sizing

    Attributes:
        compiler:    the underlying MeaningCompiler instance.
        cached_ids:  ordered list of all token IDs currently in the KV cache.
        position_map: list of (start, end, label) tuples for labelled
                      injection segments.
        stats:       cumulative compute and cache statistics.
        max_cache_tokens: maximum KV-cache capacity (0 = unlimited).
    """

    def __init__(self, compiler: MeaningCompiler, max_cache_tokens: int = 0):
        self.compiler = compiler
        self.cached_ids: List[int] = []
        self.position_map: List[Tuple[int, int, str]] = []
        self.stats = CacheStats()
        self.max_cache_tokens = max_cache_tokens
        self._dep_analyzer = DependencyAnalyzer()

    def inject(self, token_ids: List[int], label: str = "") -> Tuple[int, float]:
        """Evaluate *token_ids* into the KV cache and return (n_tokens, elapsed_s).

        If *label* is non-empty the position range is recorded in
        ``position_map`` for later eviction or analysis.
        """
        if self.max_cache_tokens > 0:
            self._enforce_capacity(len(token_ids))

        t0 = time.time()
        self.compiler.eval(token_ids)
        elapsed = time.time() - t0

        start_pos = len(self.cached_ids)
        self.cached_ids = self.cached_ids + list(token_ids)
        end_pos = len(self.cached_ids)

        if label:
            self.position_map.append((start_pos, end_pos, label))

        self.stats.total_injected_tokens += len(token_ids)
        self.stats.total_eval_time += elapsed
        self.stats.injection_count += 1
        self.stats.peak_cache_tokens = max(
            self.stats.peak_cache_tokens, len(self.cached_ids)
        )
        return len(token_ids), elapsed

    def inject_and_track(
        self,
        token_ids: List[int],
        metrics: Dict,
        key_prefix: str = "total",
        label: str = "",
    ):
        """Inject tokens and accumulate prefill timing metrics."""
        n_tok, elapsed = self.inject(token_ids, label=label)
        metrics[f"{key_prefix}_prefill_tokens"] += n_tok
        metrics[f"{key_prefix}_prefill_time"] += elapsed

    def track_generate(self, elapsed: float, n_tokens: int = 1):
        """Track generation timing for cumulative compute calculation."""
        self.stats.total_generate_time += elapsed
        self.stats.generate_count += 1

    def update_cache(self, new_ids: List[int]):
        """Append *new_ids* to the tracked cache without evaluating them
        (they were already evaluated by the caller)."""
        self.cached_ids = self.cached_ids + list(new_ids)

    def rollback(self, target_ids: List[int]):
        """Rebuild the KV cache from scratch with *target_ids* and reset
        tracking state accordingly."""
        self.compiler.rebuild_cache(target_ids)
        self.cached_ids = list(target_ids)

    def reset(self):
        """Clear the KV cache and all tracking state."""
        self.compiler.reset_cache()
        self.cached_ids = []
        self.position_map = []

    def evict_range(self, p0: int, p1: int, use_compaction: bool = False):
        """Remove KV pairs in the position range [p0, p1).

        Two strategies:
          - use_compaction=True: shift subsequent entries left via
            kv_cache_seq_shift, then trim. O(shift) cost, no re-evaluation.
            WARNING: triggers C-level GGML_ASSERT (abort) on models whose
            position encoding does not support seq_shift (e.g. Qwen3.5).
            Only enable when the model is known to support n_pos_per_embd==1.
          - use_compaction=False (default): rebuild entire cache from remaining
            IDs. O(n) cost with full re-evaluation, but universally correct.
        """
        if p0 >= p1 or p0 < 0 or p1 > len(self.cached_ids):
            return

        remaining_ids = self.cached_ids[:p0] + self.cached_ids[p1:]

        if use_compaction:
            removed_count = p1 - p0
            try:
                self.compiler.kv_cache_seq_shift(SEQ_ID, p1, len(self.cached_ids), -removed_count)
                self.compiler.kv_cache_seq_rm(SEQ_ID, len(remaining_ids), -1)
                self.cached_ids = remaining_ids
                self.stats.compact_count += 1
            except Exception:
                self.compiler.rebuild_cache(remaining_ids)
                self.cached_ids = remaining_ids
                self.stats.evict_count += 1
        else:
            self.compiler.rebuild_cache(remaining_ids)
            self.cached_ids = remaining_ids
            self.stats.evict_count += 1

        self._rebuild_position_map()

    def _rebuild_position_map(self):
        """Rebuild position map after eviction to reflect new positions."""
        new_map: List[Tuple[int, int, str]] = []
        offset = 0
        for _, _, label in self.position_map:
            found = False
            for i, (s, e, l) in enumerate(self.position_map):
                if l == label and not any(m[2] == label for m in new_map):
                    length = e - s
                    new_map.append((offset, offset + length, label))
                    offset += length
                    found = True
                    break
            if not found:
                continue
        self.position_map = new_map

    def _enforce_capacity(self, incoming_tokens: int):
        """Evict oldest injections to make room for incoming tokens.

        Uses FIFO eviction: removes the earliest injected segments first.
        """
        if self.max_cache_tokens <= 0:
            return

        current = len(self.cached_ids)
        needed = current + incoming_tokens - self.max_cache_tokens

        if needed <= 0:
            return

        while needed > 0 and self.position_map:
            p0, p1, _ = self.position_map[0]
            self.evict_range(p0, p1, use_compaction=True)
            needed -= (p1 - p0)

    @property
    def cache_size(self) -> int:
        return len(self.cached_ids)

    @property
    def cache_utilization(self) -> float:
        """Fraction of max_cache_tokens currently used (0 if unlimited)."""
        if self.max_cache_tokens <= 0:
            return 0.0
        return len(self.cached_ids) / self.max_cache_tokens

    def register_tool_call(self, tool_name: str, tool_args: Dict,
                           depends_on: Optional[List[int]] = None) -> int:
        """Register a tool call for dependency analysis."""
        return self._dep_analyzer.add_node(tool_name, tool_args, depends_on)

    def recommend_batch_size(self, max_batch: int = 8) -> int:
        """Get recommended batch size based on registered tool dependencies."""
        return self._dep_analyzer.recommend_batch_size(max_batch)

    def reset_dependency_analysis(self):
        """Clear dependency analysis state."""
        self._dep_analyzer.reset()

    def get_stats(self) -> Dict:
        """Return comprehensive cache and compute statistics."""
        return self.stats.summary()
