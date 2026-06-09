"""ModuleRegistry — Module registry and capability map (R_t)."""

import threading


class ModuleRegistry:
    """Module registry and capability map (R_t).

    Stores tool descriptions parsed from the KitchenToolRegistry
    and provides lookup, enumeration, and summary methods.
    """

    def __init__(self, tool_descriptions: dict):
        self._lock = threading.Lock()
        with self._lock:
            self._modules: dict = {}
            for name, info in tool_descriptions.items():
                if isinstance(info, str):
                    self._modules[name] = {"name": name, "description": info}
                elif isinstance(info, dict):
                    self._modules[name] = {"name": name, **info}
                else:
                    self._modules[name] = {"name": name, "description": str(info)}

    def get_module(self, name: str) -> dict:
        with self._lock:
            return dict(self._modules.get(name, {}))

    def get_all_modules(self) -> dict:
        with self._lock:
            return dict(self._modules)

    def get_pending_modules(self, invoked: set) -> list:
        with self._lock:
            return [name for name in self._modules if name not in invoked]

    def get_module_names(self) -> list:
        with self._lock:
            return list(self._modules.keys())

    def __len__(self) -> int:
        with self._lock:
            return len(self._modules)

    def summary(self) -> str:
        with self._lock:
            total = len(self._modules)
            names = sorted(self._modules.keys())
            if total <= 6:
                return f"Registry: {total} modules ({', '.join(names)})"
            return f"Registry: {total} modules ({', '.join(names[:4])}... +{total - 4} more)"
