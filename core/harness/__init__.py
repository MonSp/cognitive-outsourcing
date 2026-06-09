"""core.harness — SECM-H (State-Externalizing Cognitive Module Harness) package.

Re-exports all harness components for convenient importing::

    from core.harness import SECMHarness, ModuleRegistry, ...
"""

from .registry import ModuleRegistry
from .history import InvocationHistory
from .confidence import ConfidenceTracker
from .dependency import DependencyGraph
from .pattern_cache import PatternCache
from .budget import BudgetTracker
from .renderer import StateRenderer
from .harness import SECMHarness

__all__ = [
    "SECMHarness",
    "ModuleRegistry",
    "InvocationHistory",
    "ConfidenceTracker",
    "DependencyGraph",
    "PatternCache",
    "BudgetTracker",
    "StateRenderer",
]
