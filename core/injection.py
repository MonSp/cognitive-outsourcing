"""InjectionEngine — KV-cache-aware token injection for SIG pipeline.

Extracted from r2_benchmark.py (the more complete version with label
tracking, position_map, eviction, and cache_size property).
"""

import time
from typing import List, Dict, Tuple

from .compiler import MeaningCompiler, SEQ_ID


class InjectionEngine:
    """Manages token injection into the MeaningCompiler's KV cache while
    tracking cached token IDs, position ranges, and timing metrics.

    Attributes:
        compiler:    the underlying MeaningCompiler instance.
        cached_ids:  ordered list of all token IDs currently in the KV cache.
        position_map: list of (start, end, label) tuples for labelled
                      injection segments.
    """

    def __init__(self, compiler: MeaningCompiler):
        self.compiler = compiler
        self.cached_ids: List[int] = []
        self.position_map: List[Tuple[int, int, str]] = []

    def inject(self, token_ids: List[int], label: str = "") -> Tuple[int, float]:
        """Evaluate *token_ids* into the KV cache and return (n_tokens, elapsed_s).

        If *label* is non-empty the position range is recorded in
        ``position_map`` for later eviction or analysis.
        """
        t0 = time.time()
        self.compiler.eval(token_ids)
        elapsed = time.time() - t0
        start_pos = len(self.cached_ids)
        self.cached_ids = self.cached_ids + list(token_ids)
        end_pos = len(self.cached_ids)
        if label:
            self.position_map.append((start_pos, end_pos, label))
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

    def evict_range(self, p0: int, p1: int):
        """Remove KV pairs in the position range [p0, p1) and compact
        by rebuilding the cache from the remaining token IDs.

        Uses rebuild_cache (reset + re-eval) instead of in-place shift to
        guarantee correctness across all model architectures.
        """
        self.cached_ids = self.cached_ids[:p0] + self.cached_ids[p1:]
        self.compiler.rebuild_cache(self.cached_ids)

    @property
    def cache_size(self) -> int:
        return len(self.cached_ids)
