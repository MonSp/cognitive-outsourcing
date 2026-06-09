"""BudgetTracker — Context budget and token allocation (B_t)."""

import threading


class BudgetTracker:
    """Context budget and token allocation (B_t).

    Tracks how many tokens have been allocated to each module and
    enforces a hard total budget.  Allocation requests that would
    exceed the budget are rejected (return False).
    """

    def __init__(self, total_budget: int = 1024):
        self._lock = threading.Lock()
        self._total = total_budget
        self._allocations: dict = {}
        self._used = 0

    def allocate(self, module: str, tokens: int) -> bool:
        with self._lock:
            if self._used + tokens > self._total:
                return False
            self._allocations[module] = self._allocations.get(module, 0) + tokens
            self._used += tokens
            return True

    def get_remaining(self) -> int:
        with self._lock:
            return max(0, self._total - self._used)

    def get_utilization(self) -> float:
        with self._lock:
            if self._total <= 0:
                return 0.0
            return self._used / self._total

    def get_allocation(self, module: str) -> int:
        with self._lock:
            return self._allocations.get(module, 0)

    def get_all_allocations(self) -> dict:
        with self._lock:
            return dict(self._allocations)

    def reset(self):
        with self._lock:
            self._allocations.clear()
            self._used = 0

    def summary(self) -> str:
        with self._lock:
            return f"Budget: {self._used}/{self._total} tokens used"
