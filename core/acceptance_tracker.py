"""AcceptanceRateTracker — Per-step acceptance rate tracking for EXP-2.

Records drafter acceptance rates correlated with SIG injection events,
enabling analysis of post-injection acceptance rate depression and recovery.

Since llama.cpp's SpecDec is incompatible with Qwen3.5, we use n-gram
matching as a proxy: for each generated token, we check if a simple n-gram
drafter would have predicted it correctly.
"""

import json
import csv
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Tuple


@dataclass
class AcceptanceRecord:
    """Single acceptance rate observation."""
    run_id: int
    step_idx: int
    injection_id: int
    steps_since_injection: int
    tokens_proposed: int
    tokens_accepted: int
    acceptance_rate: float
    injection_size_tokens: int
    injection_category: str
    wall_time_s: float


class NgramDrafterSimulator:
    """Simulates a drafter using n-gram matching on the token history.

    For each generation step, checks if the last N tokens appear earlier
    in the context and predicts the continuation. This simulates what a
    prompt-lookup drafter would predict, without requiring the actual
    llama.cpp SpecDec path.
    """

    def __init__(self, ngram_size: int = 3, num_pred: int = 3):
        self.ngram_size = ngram_size
        self.num_pred = num_pred
        self.token_history: List[int] = []

    def reset(self):
        self.token_history = []

    def add_tokens(self, tokens: List[int]):
        self.token_history.extend(tokens)

    def predict(self) -> List[int]:
        """Predict next tokens using n-gram matching on history."""
        if len(self.token_history) < self.ngram_size + 1:
            return []

        ngram = self.token_history[-self.ngram_size:]
        for i in range(len(self.token_history) - self.ngram_size - 1, -1, -1):
            if self.token_history[i:i + self.ngram_size] == ngram:
                start = i + self.ngram_size
                end = min(start + self.num_pred, len(self.token_history))
                if start < end:
                    return self.token_history[start:end]
        return []

    def check_acceptance(self, actual_token: int) -> Tuple[int, int]:
        """Check if actual token matches drafter prediction.

        Returns:
            (tokens_proposed, tokens_accepted) where tokens_proposed includes
            the base token, and tokens_accepted is how many sequential draft
            tokens matched (including the base).
        """
        predictions = self.predict()
        if not predictions:
            return (1, 1)  # Only base token, always accepted

        # Count consecutive matches from the start
        accepted = 1  # Base token always accepted
        for pred in predictions:
            if pred == actual_token:
                accepted += 1
            else:
                break

        return (1 + len(predictions), accepted)


class AcceptanceRateTracker:
    """Records per-step acceptance rates correlated with injection events."""

    SIZE_THRESHOLDS = {"small": 50, "medium": 200}

    def __init__(self, run_id: int = 0):
        self.run_id = run_id
        self.records: List[AcceptanceRecord] = []
        self._injection_count = 0
        self._steps_since_injection = 0
        self._last_injection_size = 0

    def reset(self):
        self.records = []
        self._injection_count = 0
        self._steps_since_injection = 0
        self._last_injection_size = 0

    def _categorize_injection(self, n_tokens: int) -> str:
        if n_tokens <= 0:
            return "none"
        if n_tokens < self.SIZE_THRESHOLDS["small"]:
            return "small"
        if n_tokens < self.SIZE_THRESHOLDS["medium"]:
            return "medium"
        return "large"

    def on_injection(self, n_tokens: int):
        self._injection_count += 1
        self._steps_since_injection = 0
        self._last_injection_size = n_tokens

    def on_generation_step(
        self,
        tokens_proposed: int,
        tokens_accepted: int,
        wall_time_s: float = 0.0,
    ):
        self.records.append(AcceptanceRecord(
            run_id=self.run_id,
            step_idx=len(self.records),
            injection_id=self._injection_count,
            steps_since_injection=self._steps_since_injection,
            tokens_proposed=tokens_proposed,
            tokens_accepted=tokens_accepted,
            acceptance_rate=tokens_accepted / max(1, tokens_proposed),
            injection_size_tokens=self._last_injection_size,
            injection_category=self._categorize_injection(self._last_injection_size),
            wall_time_s=wall_time_s,
        ))
        self._steps_since_injection += 1

    def get_baseline_rate(self, min_steps_since_injection: int = 10) -> float:
        far_records = [
            r for r in self.records
            if r.steps_since_injection >= min_steps_since_injection
        ]
        if not far_records:
            return 0.0
        return np.mean([r.acceptance_rate for r in far_records])

    def get_recovery_curve(self) -> Dict[int, List[float]]:
        curves: Dict[int, List[float]] = {}
        for r in self.records:
            key = r.steps_since_injection
            if key not in curves:
                curves[key] = []
            curves[key].append(r.acceptance_rate)
        return curves

    def get_recovery_stats(self) -> Dict[int, Dict[str, float]]:
        curves = self.get_recovery_curve()
        stats = {}
        for t, rates in sorted(curves.items()):
            stats[t] = {
                "mean": float(np.mean(rates)),
                "std": float(np.std(rates)) if len(rates) > 1 else 0.0,
                "count": len(rates),
            }
        return stats

    def get_injection_size_breakdown(self) -> Dict[str, Dict[str, float]]:
        by_category: Dict[str, List[float]] = {}
        for r in self.records:
            cat = r.injection_category
            if cat not in by_category:
                by_category[cat] = []
            by_category[cat].append(r.acceptance_rate)

        result = {}
        for cat, rates in by_category.items():
            result[cat] = {
                "mean": float(np.mean(rates)),
                "std": float(np.std(rates)) if len(rates) > 1 else 0.0,
                "count": len(rates),
            }
        return result

    def to_dicts(self) -> List[Dict]:
        return [asdict(r) for r in self.records]

    def save_csv(self, path: str):
        if not self.records:
            return
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fieldnames = list(asdict(self.records[0]).keys())
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in self.records:
                writer.writerow(asdict(r))

    def save_json(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        data = {
            "run_id": self.run_id,
            "total_records": len(self.records),
            "baseline_rate": self.get_baseline_rate(),
            "recovery_stats": self.get_recovery_stats(),
            "injection_size_breakdown": self.get_injection_size_breakdown(),
            "records": self.to_dicts(),
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load_csv(cls, path: str) -> "AcceptanceRateTracker":
        tracker = cls(run_id=0)
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                tracker.records.append(AcceptanceRecord(
                    run_id=int(row["run_id"]),
                    step_idx=int(row["step_idx"]),
                    injection_id=int(row["injection_id"]),
                    steps_since_injection=int(row["steps_since_injection"]),
                    tokens_proposed=int(row["tokens_proposed"]),
                    tokens_accepted=int(row["tokens_accepted"]),
                    acceptance_rate=float(row["acceptance_rate"]),
                    injection_size_tokens=int(row["injection_size_tokens"]),
                    injection_category=row["injection_category"],
                    wall_time_s=float(row["wall_time_s"]),
                ))
        return tracker


def fit_recovery_model(
    tracker: AcceptanceRateTracker,
    min_steps: int = 0,
    max_steps: int = 20,
) -> Dict[str, float]:
    """Fit exponential recovery model to acceptance rate data.

    Model: alpha(t) = alpha_baseline * (1 - delta * exp(-t / tau))
    """
    try:
        from scipy.optimize import curve_fit
    except ImportError:
        return {}

    stats = tracker.get_recovery_stats()
    if not stats:
        return {}

    t_data = []
    alpha_data = []
    for t, s in sorted(stats.items()):
        if min_steps <= t <= max_steps and s["count"] >= 2:
            t_data.append(t)
            alpha_data.append(s["mean"])

    if len(t_data) < 3:
        return {}

    t_arr = np.array(t_data, dtype=float)
    alpha_arr = np.array(alpha_data, dtype=float)

    def recovery_model(t, alpha_baseline, delta, tau):
        return alpha_baseline * (1 - delta * np.exp(-t / tau))

    try:
        popt, pcov = curve_fit(
            recovery_model, t_arr, alpha_arr,
            p0=[0.6, 0.3, 3.0],
            bounds=([0, 0, 0.1], [1, 1, 50]),
            maxfev=5000,
        )
        alpha_baseline, delta, tau = popt
        residuals = alpha_arr - recovery_model(t_arr, *popt)
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((alpha_arr - np.mean(alpha_arr))**2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        perr = np.sqrt(np.diag(pcov))

        return {
            "alpha_baseline": float(alpha_baseline),
            "delta": float(delta),
            "tau": float(tau),
            "r_squared": float(r_squared),
            "alpha_baseline_ci95": float(1.96 * perr[0]),
            "delta_ci95": float(1.96 * perr[1]),
            "tau_ci95": float(1.96 * perr[2]),
            "n_points": len(t_data),
        }
    except Exception:
        return {}
