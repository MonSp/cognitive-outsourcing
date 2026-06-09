"""DependencyGraph — Inter-module dependency DAG (D_t)."""

import threading


_KITCHEN_DEPS = [
    ("start_cooking", "set_oven"),
    ("next_step", "start_cooking"),
    ("check_ingredients", "get_recipe"),
    ("get_nutrition", "get_recipe"),
    ("start_cooking", "check_ingredients"),
    ("add_shopping_item", "check_pantry"),
    ("compare_prices", "add_shopping_item"),
    ("get_substitution", "get_recipe"),
    ("set_timer", "start_cooking"),
]


class DependencyGraph:
    """Inter-module dependency DAG (D_t).

    Stores directed edges ``from_module → to_module`` meaning
    ``from_module`` *requires* ``to_module``.  Ships with built-in
    Kitchen-domain dependencies and supports runtime additions.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._deps: dict = {}
        self._reverse: dict = {}
        for src, dst in _KITCHEN_DEPS:
            self._add_edge(src, dst)

    def _add_edge(self, from_module: str, to_module: str):
        self._deps.setdefault(from_module, set()).add(to_module)
        self._reverse.setdefault(to_module, set()).add(from_module)

    def add_dependency(self, from_module: str, to_module: str):
        with self._lock:
            self._add_edge(from_module, to_module)

    def get_prerequisites(self, module: str) -> set:
        with self._lock:
            return set(self._deps.get(module, set()))

    def get_dependents(self, module: str) -> set:
        with self._lock:
            return set(self._reverse.get(module, set()))

    def has_unmet_deps(self, module: str, completed: set) -> bool:
        with self._lock:
            prereqs = self._deps.get(module, set())
            return bool(prereqs - completed)

    def get_all_modules(self) -> set:
        with self._lock:
            return set(self._deps.keys()) | set(self._reverse.keys())

    def summary(self) -> str:
        with self._lock:
            if not self._deps:
                return "Deps: (none)"
            items = []
            for src, dsts in sorted(self._deps.items()):
                for dst in sorted(dsts):
                    items.append(f"{src} needs {dst}")
            if len(items) <= 2:
                return "Deps: " + "; ".join(items)
            return "Deps: " + "; ".join(items[:2]) + f" (+{len(items) - 2} more)"
