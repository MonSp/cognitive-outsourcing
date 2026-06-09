"""PatternCache — Cognitive pattern cache with importance tags (P_t, I_t)."""

import threading


_KITCHEN_PATTERNS = {
    "recipe_planning": {
        "steps": ["find_recipes", "get_recipe", "check_ingredients", "get_nutrition"],
        "importance": 0.8,
    },
    "cooking_guidance": {
        "steps": ["set_oven", "start_cooking", "next_step", "next_step",
                   "next_step", "set_timer", "get_oven_status"],
        "importance": 0.9,
    },
    "inventory_check": {
        "steps": ["check_pantry", "check_fridge", "check_ingredients",
                   "add_shopping_item", "get_shopping_list"],
        "importance": 0.6,
    },
}


class PatternCache:
    """Cognitive pattern cache with importance tags (P_t, I_t).

    Stores frequently-used tool invocation sequences as *patterns*,
    each tagged with an importance score.  Patterns can be queried by
    step overlap to suggest pre-computed cognitive scaffolds.
    """

    def __init__(self, max_patterns: int = 20):
        self._lock = threading.Lock()
        self._max = max_patterns
        self._patterns: dict = {}

    def add_pattern(self, pattern_id: str, steps: list,
                    importance: float = 0.5):
        importance = max(0.0, min(1.0, importance))
        with self._lock:
            if len(self._patterns) >= self._max and pattern_id not in self._patterns:
                worst = min(self._patterns.items(), key=lambda x: x[1]["importance"])
                del self._patterns[worst[0]]
            self._patterns[pattern_id] = {
                "steps": list(steps),
                "importance": importance,
            }

    def query(self, query_steps: list, top_k: int = 3) -> list:
        if not query_steps:
            return []
        query_set = set(query_steps)
        with self._lock:
            scored = []
            for pid, pat in self._patterns.items():
                overlap = len(query_set & set(pat["steps"]))
                if overlap > 0:
                    score = overlap / max(len(query_set), len(pat["steps"]))
                    scored.append((pid, score, pat["importance"]))
            scored.sort(key=lambda x: -(x[1] * x[2]))
            return [(pid, s, imp) for pid, s, imp in scored[:top_k]]

    def auto_seed(self, tool_descriptions: dict):
        with self._lock:
            available = set(tool_descriptions.keys())
        for pid, info in _KITCHEN_PATTERNS.items():
            valid_steps = [s for s in info["steps"] if s in available]
            if valid_steps:
                self.add_pattern(pid, valid_steps, info["importance"])

    def update_importance(self, pattern_id: str, delta: float):
        with self._lock:
            if pattern_id in self._patterns:
                cur = self._patterns[pattern_id]["importance"]
                self._patterns[pattern_id]["importance"] = max(0.0, min(1.0, cur + delta))

    def get_pattern(self, pattern_id: str) -> dict:
        with self._lock:
            return dict(self._patterns.get(pattern_id, {}))

    def get_all(self) -> dict:
        with self._lock:
            return dict(self._patterns)

    def summary(self) -> str:
        with self._lock:
            count = len(self._patterns)
            if count == 0:
                return "Patterns: 0 cached"
            top = max(self._patterns.items(),
                      key=lambda x: x[1]["importance"])
            return (f"Patterns: {count} cached, "
                    f"top: {top[0]}({top[1]['importance']:.1f})")
