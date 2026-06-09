"""ConfidenceTracker — Per-module confidence scores via EMA (C_t)."""

import threading


class ConfidenceTracker:
    """Per-module confidence scores via EMA (C_t).

    Maintains an exponential moving average of quality scores for each
    module, producing a confidence value in [0, 1] that reflects
    historical invocation quality.
    """

    def __init__(self, alpha: float = 0.3):
        self._lock = threading.Lock()
        self._alpha = alpha
        self._scores: dict = {}

    def update(self, module: str, quality: float):
        quality = max(0.0, min(1.0, quality))
        with self._lock:
            prev = self._scores.get(module, quality)
            self._scores[module] = self._alpha * quality + (1 - self._alpha) * prev

    def get(self, module: str) -> float:
        with self._lock:
            return self._scores.get(module, 0.5)

    def get_top(self, n: int = 3) -> list:
        with self._lock:
            ranked = sorted(self._scores.items(), key=lambda x: -x[1])
            return ranked[:n]

    def rank_modules(self) -> list:
        with self._lock:
            return sorted(self._scores.items(), key=lambda x: -x[1])

    def summary(self) -> str:
        with self._lock:
            if not self._scores:
                return "Top: (none)"
            ranked = sorted(self._scores.items(), key=lambda x: -x[1])[:3]
        parts = [f"{name}({score:.2f})" for name, score in ranked]
        return "Top: " + ", ".join(parts)
