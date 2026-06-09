"""InvocationHistory — Invocation history log (H_t)."""

import threading
from collections import deque


class InvocationHistory:
    """Invocation history log (H_t).

    Records each tool invocation with module name, arguments, result,
    latency, success flag, and quality score.  Provides recent-history
    retrieval and failure-rate analytics.
    """

    def __init__(self, max_entries: int = 100):
        self._lock = threading.Lock()
        self._max = max_entries
        self._entries: deque = deque(maxlen=max_entries)

    def record(self, module: str, args: dict, result: str,
               latency_ms: float, success: bool, quality: float = 0.0):
        with self._lock:
            self._entries.append({
                "module": module,
                "args": args,
                "result": result,
                "latency_ms": latency_ms,
                "success": success,
                "quality": quality,
            })

    def get_recent(self, n: int = 3) -> list:
        with self._lock:
            entries = list(self._entries)
            return entries[-n:]

    def get_all(self) -> list:
        with self._lock:
            return list(self._entries)

    def get_module_count(self) -> int:
        with self._lock:
            return len({e["module"] for e in self._entries})

    def get_failure_rate(self, module: str = None) -> float:
        with self._lock:
            entries = list(self._entries)
        if module is not None:
            entries = [e for e in entries if e["module"] == module]
        if not entries:
            return 0.0
        failures = sum(1 for e in entries if not e["success"])
        return failures / len(entries)

    def get_invoked_modules(self) -> set:
        with self._lock:
            return {e["module"] for e in self._entries}

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)

    def summary(self) -> str:
        with self._lock:
            recent = list(self._entries)[-3:]
        if not recent:
            return "Recent: (none)"
        parts = []
        for e in recent:
            status = "OK" if e["success"] else "FAIL"
            parts.append(f"{e['module']}→{status}")
        return "Recent: " + ", ".join(parts)
