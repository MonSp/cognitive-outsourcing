"""StateRenderer — Compact derived-state renderer (≤100 tokens per step)."""

import threading


class StateRenderer:
    """Compact derived-state renderer (≤100 tokens per step).

    Combines summaries from all six state components into a single
    compact text block suitable for injection into the MeaningCompiler
    context window.  The output is trimmed to ``max_tokens`` via a
    heuristic word-count approximation when a real tokenizer is absent.
    """

    def __init__(self, max_tokens: int = 100):
        self._lock = threading.Lock()
        self._max_tokens = max_tokens

    def render(self, registry, history, confidence, dependency,
               pattern_cache, budget, tokenizer=None, ablation=None) -> str:
        ablation = ablation or {}
        parts = ["[SECM-H]"]

        if ablation.get("R_t", True):
            total = len(registry)
            invoked = history.get_invoked_modules()
            pending = registry.get_pending_modules(invoked)
            parts.append(f"Modules: {total} available, {len(invoked)} invoked, "
                         f"{len(pending)} pending.")

        if ablation.get("C_t", True):
            top = confidence.get_top(2)
            if top:
                top_str = ", ".join(f"{n}({s:.2f})" for n, s in top)
                parts.append(f"Top: {top_str}.")

        if ablation.get("B_t", True):
            used = int(budget.get_utilization() * budget._total)
            parts.append(f"Budget: {used}/{budget._total}.")

        if ablation.get("H_t", True):
            recent = history.get_recent(2)
            if recent:
                entries = []
                for e in recent:
                    status = "OK" if e["success"] else "FAIL"
                    entries.append(f"{e['module']}→{status}")
                parts.append(f"Recent: {', '.join(entries)}.")

        if ablation.get("D_t", True):
            last_modules = [e["module"] for e in history.get_recent(1)]
            if last_modules:
                deps = dependency.get_prerequisites(last_modules[0])
                if deps:
                    parts.append(f"Deps: {last_modules[0]} needs "
                                 f"{', '.join(sorted(deps))}.")

        if ablation.get("P_t", True):
            count = len(pattern_cache.get_all())
            if count > 0:
                parts.append(f"Patterns: {count}.")

        text = " ".join(parts)

        with self._lock:
            text = self._trim_to_tokens(text, tokenizer)

        return text

    def _trim_to_tokens(self, text: str, tokenizer=None) -> str:
        token_count = self.render_token_count(text, tokenizer)
        if token_count <= self._max_tokens:
            return text
        ratio = self._max_tokens / max(token_count, 1)
        keep_chars = int(len(text) * ratio * 0.9)
        return text[:max(keep_chars, 20)].rstrip() + "..."

    def render_token_count(self, text: str, tokenizer=None) -> int:
        if tokenizer is not None:
            try:
                ids = tokenizer(text, add_bos=False)
                return len(ids)
            except Exception:
                pass
        return max(1, int(len(text.split()) * 1.35))
