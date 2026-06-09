"""SECMHarness — State-Externalizing Cognitive Module Harness orchestrator."""

import threading

from .registry import ModuleRegistry
from .history import InvocationHistory
from .confidence import ConfidenceTracker
from .dependency import DependencyGraph
from .pattern_cache import PatternCache
from .budget import BudgetTracker
from .renderer import StateRenderer


_DEFAULT_ABLATION = {
    "R_t": True,
    "H_t": True,
    "C_t": True,
    "D_t": True,
    "P_t": True,
    "B_t": True,
}


def _estimate_quality(result: str) -> float:
    length = len(result)
    if length <= 0:
        return 0.0
    raw = min(1.0, length / 200.0)
    return max(0.1, raw)


class SECMHarness:
    """State-Externalizing Cognitive Module Harness — orchestrator.

    Coordinates six state components (R_t, H_t, C_t, D_t, P_t, B_t)
    and a compact renderer.  ``ablation_config`` disables individual
    components by setting their key to False.
    """

    def __init__(self, tool_descriptions: dict, total_budget: int = 1024,
                 ablation_config: dict = None):
        self._lock = threading.Lock()
        self._ablation = dict(_DEFAULT_ABLATION)
        if ablation_config:
            self._ablation.update(ablation_config)

        self.registry = ModuleRegistry(tool_descriptions)
        self.history = InvocationHistory(max_entries=100)
        self.confidence = ConfidenceTracker(alpha=0.3)
        self.dependency = DependencyGraph()
        self.pattern_cache = PatternCache(max_patterns=20)
        self.budget = BudgetTracker(total_budget=total_budget)
        self.renderer = StateRenderer(max_tokens=100)

        self.pattern_cache.auto_seed(tool_descriptions)

    def pre_invoke(self, module: str, args: dict) -> None:
        with self._lock:
            if self._ablation.get("D_t", True):
                if self.dependency.has_unmet_deps(
                        module, self.history.get_invoked_modules()):
                    pass

    def post_invoke(self, module: str, args: dict, result: str,
                    latency_ms: float, success: bool,
                    quality: float = 0.0) -> None:
        if quality <= 0.0:
            quality = _estimate_quality(result)

        with self._lock:
            if self._ablation.get("H_t", True):
                self.history.record(module, args, result, latency_ms,
                                    success, quality)
            if self._ablation.get("C_t", True):
                self.confidence.update(module, quality)
            if self._ablation.get("B_t", True):
                token_est = max(1, len(result) // 4)
                self.budget.allocate(module, token_est)

    def render_state(self, tokenizer=None) -> str:
        with self._lock:
            return self.renderer.render(
                self.registry, self.history, self.confidence,
                self.dependency, self.pattern_cache, self.budget,
                tokenizer=tokenizer, ablation=self._ablation)

    def get_state_summary(self) -> dict:
        with self._lock:
            return {
                "modules_total": len(self.registry),
                "modules_invoked": len(self.history.get_invoked_modules()),
                "invocations": len(self.history),
                "top_confidence": self.confidence.get_top(3),
                "budget_used": int(self.budget.get_utilization()
                                   * self.budget._total),
                "budget_total": self.budget._total,
                "patterns_cached": len(self.pattern_cache.get_all()),
                "failure_rate": self.history.get_failure_rate(),
            }

    def seed_patterns(self, patterns: list) -> None:
        for p in patterns:
            pid = p.get("id", p.get("pattern_id", ""))
            steps = p.get("steps", [])
            importance = p.get("importance", 0.5)
            if pid and steps:
                self.pattern_cache.add_pattern(pid, steps, importance)

    def reset(self):
        with self._lock:
            self.history = InvocationHistory(max_entries=100)
            self.confidence = ConfidenceTracker(alpha=0.3)
            self.budget = BudgetTracker(
                total_budget=self.budget._total)
