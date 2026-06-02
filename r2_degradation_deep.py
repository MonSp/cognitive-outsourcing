#!/usr/bin/env python3
"""R2 Degradation Curve — Deep Validation.

Extended R2 experiment that probes recall degradation over 32-64 rounds of
fact injection.  Each round inserts a "city information card" containing
3-5 correlated facts (weather, population, landmark, specialty, language).
Every 4 rounds, two recall probes are issued:
  - short-term recall : last injected city card
  - long-term recall  : city card from round 0

After the run, three candidate degradation hypotheses are fitted against
the (m, R) data via scipy.optimize.curve_fit:
  H1 (logarithmic) : R(k, m) = a - b * ln(1 + m)
  H2 (linear)      : R(k, m) = a - b * m
  H3 (phase change): R(k, m) = a            if m < m_crit
                                a * exp(-c * (m - m_crit))  otherwise

R^2, AIC, and BIC are reported for each hypothesis and the best one
(lowest BIC) is selected.

Usage:
  python r2_degradation_deep.py --model models/Qwen3.5-0.8B-Q4_K_M.gguf \\
      --rounds 32 --n-ctx 16384 --n-gpu-layers 99

Output:
  data/r2_degradation_deep_<model>_<rounds>.json
"""

import argparse
import json
import math
import os
import re
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

from core import MeaningCompiler, InjectionEngine, ToolRegistry, GPUMonitor
from core.metrics import mean_std
from core.info_theory import shannon_entropy

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

try:
    from scipy.optimize import curve_fit
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


# ======================================================================
# City card definitions — 5 correlated facts per card
# ======================================================================
CITY_CARDS: Dict[str, Dict[str, str]] = {
    "paris": {
        "weather":    "Partly cloudy, 18C",
        "population": "Population: 2.2 million",
        "landmark":   "Landmark: Eiffel Tower",
        "specialty":  "Specialty: Croissants",
        "language":   "Language: French",
    },
    "tokyo": {
        "weather":    "Rain, 22C",
        "population": "Population: 13.9 million",
        "landmark":   "Landmark: Tokyo Skytree",
        "specialty":  "Specialty: Sushi",
        "language":   "Language: Japanese",
    },
    "rome": {
        "weather":    "Sunny, 26C",
        "population": "Population: 2.8 million",
        "landmark":   "Landmark: Colosseum",
        "specialty":  "Specialty: Pizza al Taglio",
        "language":   "Language: Italian",
    },
    "london": {
        "weather":    "Overcast, 15C",
        "population": "Population: 8.9 million",
        "landmark":   "Landmark: Big Ben",
        "specialty":  "Specialty: Fish and Chips",
        "language":   "Language: English",
    },
    "newyork": {
        "weather":    "Clear, 22C",
        "population": "Population: 8.4 million",
        "landmark":   "Landmark: Statue of Liberty",
        "specialty":  "Specialty: Bagels",
        "language":   "Language: English",
    },
    "sydney": {
        "weather":    "Sunny, 24C",
        "population": "Population: 5.3 million",
        "landmark":   "Landmark: Opera House",
        "specialty":  "Specialty: Pavlova",
        "language":   "Language: English",
    },
    "beijing": {
        "weather":    "Smog, 28C",
        "population": "Population: 21.5 million",
        "landmark":   "Landmark: Great Wall",
        "specialty":  "Specialty: Peking Duck",
        "language":   "Language: Mandarin",
    },
    "dubai": {
        "weather":    "Hot, 38C",
        "population": "Population: 3.5 million",
        "landmark":   "Landmark: Burj Khalifa",
        "specialty":  "Specialty: Shawarma",
        "language":   "Language: Arabic",
    },
    "seoul": {
        "weather":    "Clear, 20C",
        "population": "Population: 9.7 million",
        "landmark":   "Landmark: Gyeongbokgung",
        "specialty":  "Specialty: Kimchi",
        "language":   "Language: Korean",
    },
    "bangkok": {
        "weather":    "Humid, 33C",
        "population": "Population: 10.5 million",
        "landmark":   "Landmark: Grand Palace",
        "specialty":  "Specialty: Pad Thai",
        "language":   "Language: Thai",
    },
    "istanbul": {
        "weather":    "Mild, 21C",
        "population": "Population: 15.5 million",
        "landmark":   "Landmark: Hagia Sophia",
        "specialty":  "Specialty: Baklava",
        "language":   "Language: Turkish",
    },
    "cairo": {
        "weather":    "Hot, 35C",
        "population": "Population: 9.5 million",
        "landmark":   "Landmark: Pyramids of Giza",
        "specialty":  "Specialty: Koshari",
        "language":   "Language: Arabic",
    },
}

CITY_ORDER: List[str] = list(CITY_CARDS.keys())

FACT_KEYS: List[str] = ["weather", "population", "landmark", "specialty", "language"]
FACT_TO_QUESTION: Dict[str, str] = {
    "weather":    "What is the weather in {city}?",
    "population": "What was the population of {city}?",
    "landmark":   "What is the landmark in {city}?",
    "specialty":  "What is the specialty of {city}?",
    "language":   "What language is spoken in {city}?",
}


# ======================================================================
# Recall scoring
# ======================================================================
def _fact_keywords(fact_value: str) -> List[str]:
    """Extract the discriminative tokens from a fact string.

    The recall score counts how many of these tokens appear in the
    model response, normalised by the number of tokens.  Pure stop
    words are filtered but domain-specific terms (numbers, proper
    nouns, temperature values) are preserved.
    """
    STOP = {
        "the", "a", "an", "of", "in", "is", "and", "or", "to", "for",
        "with", "on", "at", "by", "from", "as", "be", "it", "this",
        "that", "was", "are", "were",
    }
    text = fact_value.lower()
    text = re.sub(r"[^a-z0-9.\-°c%]", " ", text)
    tokens = [t for t in text.split() if t and t not in STOP and len(t) >= 1]
    return tokens


def recall_score(response: str, fact_value: str) -> float:
    """Compute recall score in [0, 1] for a single fact.

    Returns the fraction of discriminative tokens in *fact_value* that
    appear (as substrings) in *response*.
    """
    if not response or not fact_value:
        return 0.0
    kws = _fact_keywords(fact_value)
    if not kws:
        return 0.5
    resp_lower = response.lower()
    hits = sum(1 for kw in kws if kw in resp_lower)
    return min(1.0, hits / max(len(kws), 1))


def card_recall_score(response: str, card: Dict[str, str]) -> float:
    """Aggregate recall over all facts in a city card.

    Returns the mean per-fact recall score, in [0, 1].
    """
    if not card:
        return 0.0
    per_fact = [recall_score(response, card.get(k, "")) for k in FACT_KEYS]
    valid = [s for s, v in zip(per_fact, FACT_KEYS) if card.get(v)]
    if not valid:
        return 0.0
    return sum(valid) / len(valid)


# ======================================================================
# Degradation hypothesis models
# ======================================================================
def _h1_log(m, a: float, b: float):
    """H1: logarithmic decay.  R = a - b * ln(1 + m)."""
    m_arr = np.asarray(m, dtype=float)
    return a - b * np.log1p(m_arr)


def _h2_linear(m, a: float, b: float):
    """H2: linear decay.  R = a - b * m."""
    m_arr = np.asarray(m, dtype=float)
    return a - b * m_arr


def _h3_phase(m, a: float, m_crit: float, c: float):
    """H3: phase-change decay (vectorised).

    R = a                    if m < m_crit
        a * exp(-c*(m-m_crit))   otherwise
    """
    m_arr = np.asarray(m, dtype=float)
    above = m_arr >= m_crit
    decay = a * np.exp(-c * (m_arr - m_crit))
    return np.where(above, decay, a)


def _h3_smooth(m, a: float, m_crit: float, c: float):
    """Smoothed phase transition used as a curve_fit target.

    The smoothed form is differentiable everywhere, which lets
    ``scipy.optimize.curve_fit`` (Levenberg-Marquardt) converge; the
    hard step version above is used to score the resulting parameters
    on the original fit quality.
    """
    m_arr = np.asarray(m, dtype=float)
    gate = 1.0 / (1.0 + np.exp(-(m_arr - m_crit)))
    decay = a * np.exp(-c * np.maximum(m_arr - m_crit, 0.0))
    return a * (1.0 - gate) + decay * gate


def _sse(y_obs: List[float], y_pred: List[float]) -> float:
    return sum((o - p) ** 2 for o, p in zip(y_obs, y_pred))


def _r_squared(y_obs: List[float], y_pred: List[float]) -> float:
    if not y_obs:
        return 0.0
    mean_y = sum(y_obs) / len(y_obs)
    ss_tot = sum((y - mean_y) ** 2 for y in y_obs)
    ss_res = _sse(y_obs, y_pred)
    if ss_tot <= 0:
        return 0.0
    return 1.0 - ss_res / ss_tot


def _aic_bic(n: int, k: int, ss_res: float) -> Tuple[float, float]:
    """Compute Akaike / Bayesian information criteria for least-squares fit."""
    if n <= k or ss_res <= 0:
        return float("inf"), float("inf")
    sigma2 = ss_res / n
    aic = n * math.log(sigma2) + 2 * k
    bic = n * math.log(sigma2) + k * math.log(n)
    return aic, bic


def _to_float_list(y_pred) -> List[float]:
    """Convert numpy array / iterable / scalar predictions to a list of floats."""
    if hasattr(y_pred, "__len__") and not isinstance(y_pred, (str, bytes)):
        arr = np.asarray(y_pred, dtype=float).ravel()
        return [float(x) for x in arr]
    return [float(y_pred)]


def fit_h1(m_vals: List[float], r_vals: List[float]) -> Optional[Dict[str, Any]]:
    """Fit logarithmic decay model H1."""
    if not SCIPY_AVAILABLE or len(m_vals) < 3:
        return None
    try:
        p0 = [max(0.9, max(r_vals) if r_vals else 1.0), 0.05]
        bounds = ([-5.0, 0.0], [5.0, 5.0])
        popt, _ = curve_fit(_h1_log, m_vals, r_vals, p0=p0, bounds=bounds, maxfev=2000)
        a, b = popt
        y_pred = _to_float_list(_h1_log(np.asarray(m_vals, dtype=float), a, b))
        ss_res = _sse(r_vals, y_pred)
        n = len(m_vals)
        k = 2
        aic, bic = _aic_bic(n, k, ss_res)
        r2 = _r_squared(r_vals, y_pred)
        return {
            "model": "H1_logarithmic",
            "formula": "R(m) = a - b * ln(1 + m)",
            "params": {"a": float(a), "b": float(b)},
            "r_squared": float(r2),
            "aic": float(aic),
            "bic": float(bic),
            "ss_res": float(ss_res),
            "n_points": n,
            "k_params": k,
            "predictions": y_pred,
        }
    except Exception as e:
        return {"model": "H1_logarithmic", "error": str(e)}


def fit_h2(m_vals: List[float], r_vals: List[float]) -> Optional[Dict[str, Any]]:
    """Fit linear decay model H2."""
    if not SCIPY_AVAILABLE or len(m_vals) < 3:
        return None
    try:
        p0 = [max(0.9, max(r_vals) if r_vals else 1.0), 0.01]
        bounds = ([-5.0, 0.0], [5.0, 5.0])
        popt, _ = curve_fit(_h2_linear, m_vals, r_vals, p0=p0, bounds=bounds, maxfev=2000)
        a, b = popt
        y_pred = _to_float_list(_h2_linear(np.asarray(m_vals, dtype=float), a, b))
        ss_res = _sse(r_vals, y_pred)
        n = len(m_vals)
        k = 2
        aic, bic = _aic_bic(n, k, ss_res)
        r2 = _r_squared(r_vals, y_pred)
        return {
            "model": "H2_linear",
            "formula": "R(m) = a - b * m",
            "params": {"a": float(a), "b": float(b)},
            "r_squared": float(r2),
            "aic": float(aic),
            "bic": float(bic),
            "ss_res": float(ss_res),
            "n_points": n,
            "k_params": k,
            "predictions": y_pred,
        }
    except Exception as e:
        return {"model": "H2_linear", "error": str(e)}


def fit_h3(m_vals: List[float], r_vals: List[float]) -> Optional[Dict[str, Any]]:
    """Fit phase-change decay model H3.

    Uses a smooth approximation to the discontinuity so that gradient-based
    curve_fit can converge.  During evaluation, predictions are computed
    with the original step function.
    """
    if not SCIPY_AVAILABLE or len(m_vals) < 4:
        return None

    try:
        r_max = max(r_vals) if r_vals else 1.0
        m_max = max(m_vals) if m_vals else 1.0
        p0 = [max(0.5, r_max), m_max / 2.0, 0.05]
        bounds = ([-5.0, 0.0, 0.0], [5.0, m_max * 1.5, 5.0])
        popt, _ = curve_fit(
            _h3_smooth, m_vals, r_vals, p0=p0, bounds=bounds, maxfev=4000
        )
        a, m_crit, c = popt
        # Use the original (non-smooth) step function for the final
        # predictions and residuals to honestly report fit quality.
        y_pred = _to_float_list(_h3_phase(np.asarray(m_vals, dtype=float), a, m_crit, c))
        ss_res = _sse(r_vals, y_pred)
        n = len(m_vals)
        k = 3
        aic, bic = _aic_bic(n, k, ss_res)
        r2 = _r_squared(r_vals, y_pred)
        return {
            "model": "H3_phase_change",
            "formula": "R(m) = a if m < m_crit else a * exp(-c * (m - m_crit))",
            "params": {"a": float(a), "m_crit": float(m_crit), "c": float(c)},
            "r_squared": float(r2),
            "aic": float(aic),
            "bic": float(bic),
            "ss_res": float(ss_res),
            "n_points": n,
            "k_params": k,
            "predictions": y_pred,
        }
    except Exception as e:
        return {"model": "H3_phase_change", "error": str(e)}


def select_best_model(fits: Dict[str, Optional[Dict[str, Any]]]) -> str:
    """Return the name of the model with the lowest BIC among valid fits."""
    valid = {
        name: fit
        for name, fit in fits.items()
        if fit and "bic" in fit and math.isfinite(fit["bic"])
    }
    if not valid:
        return "none"
    return min(valid, key=lambda n: valid[n]["bic"])


# ======================================================================
# Card injection helpers
# ======================================================================
def _detect_chat_format(model_path: str) -> str:
    """Detect which chat format to use based on model filename."""
    ml = model_path.lower()
    if "gemma" in ml:
        return "gemma"
    return "qwen"


def build_card_text(round_idx: int, city: str, card: Dict[str, str], chat_fmt: str = "qwen") -> str:
    """Render a city card as a single injection block.

    All five facts are concatenated into one block so the model sees
    a coherent, correlated chunk of information each round.
    """
    lines = [f"[Round {round_idx + 1}] City: {city.title()}"]
    for key in FACT_KEYS:
        val = card.get(key, "")
        if val:
            lines.append(f"  - {val}")
    body = "\n".join(lines) + "\n"
    if chat_fmt == "gemma":
        return f"<start_of_turn>user\nHere is a city information card. Memorize all facts.\n{body}<end_of_turn>\n<start_of_turn>model\nUnderstood, I have memorized the information.<end_of_turn>\n"
    return f"<|im_start|>user\nHere is a city information card. Memorize all facts.\n{body}<|im_end|>\n<|im_start|>assistant\nUnderstood, I have memorized the information.<|im_end|>\n"


def build_probe_prompt(fact_key: str, city: str, chat_fmt: str = "qwen") -> str:
    """Build a probe question that asks about a specific fact.

    For Qwen3.5 models the prompt includes a pre-filled empty
    <think></think> block so the model skips its reasoning mode and
    answers directly.  This mirrors the chat-template behaviour when
    enable_thinking=False (see Qwen3.5 chat template lines 149-152).
    """
    template = FACT_TO_QUESTION[fact_key]
    q = template.format(city=city.title())
    if chat_fmt == "gemma":
        return f"<start_of_turn>user\n{q}<end_of_turn>\n<start_of_turn>model\n"
    return f"<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


# ======================================================================
# Main experiment
# ======================================================================
def run_experiment(args: argparse.Namespace) -> Dict[str, Any]:
    """Run the deep R2 degradation experiment and return the result dict."""
    print("=" * 80)
    print("  R2 Degradation Curve — Deep Validation")
    print(f"  Model: {args.model}")
    print(f"  Rounds: {args.rounds} | Probe every: {args.probe_interval} rounds")
    print(f"  Context: n_ctx={args.n_ctx}, n_gpu_layers={args.n_gpu_layers}")
    print("=" * 80)

    if not SCIPY_AVAILABLE:
        print("  WARNING: scipy is not available; fits will be skipped.")

    chat_fmt = _detect_chat_format(args.model)
    print(f"  Chat format: {chat_fmt}")

    gpu = GPUMonitor()
    compiler = MeaningCompiler(
        model_path=args.model,
        n_ctx=args.n_ctx,
        n_threads=args.n_threads,
        n_gpu_layers=args.n_gpu_layers,
    )
    module = ToolRegistry()
    engine = InjectionEngine(compiler)

    # ---- prefill system prompt ----------------------------------------
    if chat_fmt == "gemma":
        system_text = (
            "<start_of_turn>user\nYou are a helpful travel assistant. "
            "You will be given a stream of city information cards. "
            "Memorize all facts.<end_of_turn>\n<start_of_turn>model\n"
            "Understood, I will memorize all city facts.<end_of_turn>\n"
        )
    else:
        system_text = (
            "<|im_start|>system\nYou are a helpful travel assistant. "
            "You will be given a stream of city information cards. "
            "Memorize all facts.<|im_end|>\n"
        )
    sys_ids = list(compiler.tokenize(system_text, add_bos=False))
    t0 = time.time()
    compiler.eval(sys_ids)
    prefill_sys_time = time.time() - t0
    engine.update_cache(sys_ids)
    print(f"  System prefill: {prefill_sys_time:.2f}s, cache={engine.cache_size} tok")

    # ---- per-round records --------------------------------------------
    round_records: List[Dict[str, Any]] = []
    probe_records: List[Dict[str, Any]] = []
    city_at_round: List[str] = []
    card_at_round: List[Dict[str, str]] = []

    # ---- main loop ----------------------------------------------------
    for r in range(args.rounds):
        city = CITY_ORDER[r % len(CITY_ORDER)]
        card = CITY_CARDS[city]
        city_at_round.append(city)
        card_at_round.append(card)

        # Pick a fact slot for short-term probe (rotate through facts so
        # we test multiple discriminative dimensions across runs).
        short_fact_key = FACT_KEYS[r % len(FACT_KEYS)]

        t_round_start = time.time()
        card_text = build_card_text(r, city, card, chat_fmt=chat_fmt)
        card_ids = list(compiler.tokenize(card_text, add_bos=False))
        pf_t0 = time.time()
        compiler.eval(card_ids)
        pf_elapsed = time.time() - pf_t0
        engine.update_cache(card_ids)
        round_t = time.time() - t_round_start

        record = {
            "round": r + 1,
            "city": city,
            "fact_key": short_fact_key,
            "card_text": card_text.strip(),
            "card_tokens": len(card_ids),
            "prefill_time_ms": pf_elapsed * 1000.0,
            "cache_tokens_after": engine.cache_size,
            "round_time_ms": round_t * 1000.0,
            "card_entropy_bits": shannon_entropy(card_text),
        }
        round_records.append(record)

        # ---- probe every N rounds ------------------------------------
        if (r + 1) % args.probe_interval == 0:
            probe_t0 = time.time()
            short_expected = card[short_fact_key]
            short_score, short_resp = _probe(
                compiler, engine,
                build_probe_prompt(short_fact_key, city, chat_fmt=chat_fmt),
                short_expected,
                args.max_new_probe,
            )
            engine.update_cache([])

            # Long-term probe: city from round 0
            long_city = city_at_round[0]
            long_card = card_at_round[0]
            long_fact_key = FACT_KEYS[0]  # weather — most distinctive round-0 fact
            long_score, long_resp = _probe(
                compiler, engine,
                build_probe_prompt(long_fact_key, long_city, chat_fmt=chat_fmt),
                long_card[long_fact_key],
                args.max_new_probe,
            )
            engine.update_cache([])
            probe_t = time.time() - probe_t0

            # Also probe a different fact on long-term to test
            # discrimination across the card, not just one slot.
            long_fact_alt = FACT_KEYS[2]  # landmark
            long_score_alt, long_resp_alt = _probe(
                compiler, engine,
                build_probe_prompt(long_fact_alt, long_city, chat_fmt=chat_fmt),
                long_card[long_fact_alt],
                args.max_new_probe,
            )
            engine.update_cache([])

            # Long-term card-aggregate score: average over all five facts.
            long_card_scores: List[float] = []
            long_card_resps: Dict[str, str] = {}
            for fk in FACT_KEYS:
                s, resp = _probe(
                    compiler, engine,
                    build_probe_prompt(fk, long_city, chat_fmt=chat_fmt),
                    long_card[fk],
                    args.max_new_probe,
                )
                long_card_scores.append(s)
                long_card_resps[fk] = resp.strip()
                engine.update_cache([])
            long_card_score = sum(long_card_scores) / len(long_card_scores)

            # Short-term card-aggregate over previous round's city.
            prev_round = r - 1
            if prev_round >= 0:
                prev_city = city_at_round[prev_round]
                prev_card = card_at_round[prev_round]
                short_card_scores: List[float] = []
                for fk in FACT_KEYS:
                    s, resp = _probe(
                        compiler, engine,
                        build_probe_prompt(fk, prev_city, chat_fmt=chat_fmt),
                        prev_card[fk],
                        args.max_new_probe,
                    )
                    short_card_scores.append(s)
                    engine.update_cache([])
                short_card_score = sum(short_card_scores) / len(short_card_scores)
            else:
                short_card_score = short_score
                prev_city = city

            probe_record = {
                "round": r + 1,
                "cache_tokens": engine.cache_size,
                "short_term": {
                    "city": prev_city,
                    "fact_key": short_fact_key,
                    "single_fact_score": float(short_score),
                    "card_aggregate_score": float(short_card_score),
                    "response": short_resp.strip()[:200],
                },
                "long_term": {
                    "city": long_city,
                    "fact_key_primary": long_fact_key,
                    "fact_key_alt": long_fact_alt,
                    "primary_score": float(long_score),
                    "alt_score": float(long_score_alt),
                    "card_aggregate_score": float(long_card_score),
                    "primary_response": long_resp.strip()[:200],
                    "alt_response": long_resp_alt.strip()[:200],
                    "per_fact_scores": {
                        fk: float(s) for fk, s in zip(FACT_KEYS, long_card_scores)
                    },
                },
                "probe_time_ms": probe_t * 1000.0,
            }
            probe_records.append(probe_record)

            print(
                f"  R{r + 1:3d} | cache={engine.cache_size:6d} tok | "
                f"short={short_card_score:.2f} | long={long_card_score:.2f} "
                f"| long_primary={long_score:.2f} | long_alt={long_score_alt:.2f} "
                f"| probe={probe_t:.1f}s"
            )

    # ---- fit degradation hypotheses ----------------------------------
    print("\n  Fitting degradation hypotheses (using long-term card aggregate)...")
    m_vals = [float(p["round"]) for p in probe_records]
    r_short = [float(p["short_term"]["card_aggregate_score"]) for p in probe_records]
    r_long = [float(p["long_term"]["card_aggregate_score"]) for p in probe_records]
    r_long_primary = [float(p["long_term"]["primary_score"]) for p in probe_records]
    cache_tokens = [float(p["cache_tokens"]) for p in probe_records]

    fits_short = {
        "H1": fit_h1(m_vals, r_short),
        "H2": fit_h2(m_vals, r_short),
        "H3": fit_h3(m_vals, r_short),
    }
    fits_long = {
        "H1": fit_h1(m_vals, r_long),
        "H2": fit_h2(m_vals, r_long),
        "H3": fit_h3(m_vals, r_long),
    }
    fits_long_primary = {
        "H1": fit_h1(m_vals, r_long_primary),
        "H2": fit_h2(m_vals, r_long_primary),
        "H3": fit_h3(m_vals, r_long_primary),
    }

    best_short = select_best_model(fits_short)
    best_long = select_best_model(fits_long)
    best_long_primary = select_best_model(fits_long_primary)

    # ---- summary ------------------------------------------------------
    short_mean, short_std = mean_std(r_short)
    long_mean, long_std = mean_std(r_long)
    long_primary_mean, long_primary_std = mean_std(r_long_primary)
    cache_mean, cache_std = mean_std(cache_tokens)

    summary = {
        "n_probe_points": len(probe_records),
        "short_term": {
            "mean": float(short_mean),
            "std": float(short_std),
            "min": float(min(r_short)) if r_short else 0.0,
            "max": float(max(r_short)) if r_short else 0.0,
        },
        "long_term": {
            "mean": float(long_mean),
            "std": float(long_std),
            "min": float(min(r_long)) if r_long else 0.0,
            "max": float(max(r_long)) if r_long else 0.0,
        },
        "long_term_primary": {
            "mean": float(long_primary_mean),
            "std": float(long_primary_std),
            "min": float(min(r_long_primary)) if r_long_primary else 0.0,
            "max": float(max(r_long_primary)) if r_long_primary else 0.0,
        },
        "cache_tokens": {
            "mean": float(cache_mean),
            "std": float(cache_std),
            "final": float(cache_tokens[-1]) if cache_tokens else 0.0,
        },
    }

    print("\n  === Hypothesis Comparison (long-term card aggregate) ===")
    print(f"  {'Model':<22} {'R^2':<10} {'AIC':<14} {'BIC':<14}")
    print(f"  {'-' * 22} {'-' * 10} {'-' * 14} {'-' * 14}")
    for name in ["H1", "H2", "H3"]:
        f = fits_long.get(name)
        if f and "r_squared" in f:
            print(
                f"  {name + '_' + f.get('model', ''):<22} "
                f"{f['r_squared']:<10.4f} {f['aic']:<14.3f} {f['bic']:<14.3f}"
            )
    print(f"  Best model (lowest BIC): {best_long}")

    print("\n  === Hypothesis Comparison (short-term card aggregate) ===")
    print(f"  {'Model':<22} {'R^2':<10} {'AIC':<14} {'BIC':<14}")
    print(f"  {'-' * 22} {'-' * 10} {'-' * 14} {'-' * 14}")
    for name in ["H1", "H2", "H3"]:
        f = fits_short.get(name)
        if f and "r_squared" in f:
            print(
                f"  {name + '_' + f.get('model', ''):<22} "
                f"{f['r_squared']:<10.4f} {f['aic']:<14.3f} {f['bic']:<14.3f}"
            )
    print(f"  Best model (lowest BIC): {best_short}")

    result = {
        "metadata": {
            "model": args.model,
            "n_ctx": args.n_ctx,
            "n_threads": args.n_threads,
            "n_gpu_layers": args.n_gpu_layers,
            "rounds": args.rounds,
            "probe_interval": args.probe_interval,
            "cities_used": list({c for c in city_at_round}),
            "timestamp": time.time(),
        },
        "round_records": round_records,
        "probe_records": probe_records,
        "fits": {
            "short_term": fits_short,
            "long_term": fits_long,
            "long_term_primary": fits_long_primary,
        },
        "best_model": {
            "short_term": best_short,
            "long_term": best_long,
            "long_term_primary": best_long_primary,
        },
        "summary": summary,
    }

    gpu.shutdown()
    return result


def _probe(
    compiler: MeaningCompiler,
    engine: InjectionEngine,
    prompt: str,
    expected_fact: str,
    max_new: int,
) -> Tuple[float, str]:
    """Issue a single probe and return (recall_score, response_text).

    The prompt is appended to the existing KV cache; the model is asked
    to complete the answer; the generated text is decoded and scored
    against *expected_fact* via ``recall_score``.

    Handles Qwen3.5 <think>...</think> blocks by:
    1. Generating a generous first pass (max_new tokens)
    2. Stripping any <think>...</think> blocks from the output
    3. Taking the first non-empty line after stripping as the answer
    """
    ids = list(compiler.tokenize(prompt, add_bos=False))
    compiler.eval(ids)
    engine.update_cache(ids)
    text, _, _ = compiler.generate_until_any(
        ["\n", "</think>"],
        max_new=max_new,
        rep_threshold=3,
    )
    if "<think>" in text:
        remaining, _, _ = compiler.generate_until_any(
            ["</think>", "\n\n"],
            max_new=max_new * 4,
            rep_threshold=3,
        )
        text = text + remaining
        after, _ = compiler.generate_until_str(
            "\n", max_new=max_new, rep_threshold=3,
        )
        text = text + after
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    text = lines[0] if lines else ""
    score = recall_score(text, expected_fact)
    return float(score), text


def main() -> int:
    parser = argparse.ArgumentParser(
        description="R2 Degradation Curve — Deep Validation (32-64 rounds, "
                    "3-5 fact cards, 3 hypothesis fits)."
    )
    parser.add_argument("--model", type=str, required=True,
                        help="Path to GGUF model file.")
    parser.add_argument("--n-ctx", type=int, default=16384,
                        help="Context size (default 16384).")
    parser.add_argument("--n-threads", type=int, default=4)
    parser.add_argument("--n-gpu-layers", type=int, default=99)
    parser.add_argument("--rounds", type=int, default=32,
                        help="Number of injection rounds (32 or 64).")
    parser.add_argument("--probe-interval", type=int, default=4,
                        help="Probe recall every N rounds.")
    parser.add_argument("--max-new-probe", type=int, default=40,
                        help="Max new tokens per probe answer.")
    parser.add_argument("--output-dir", type=str, default="data",
                        help="Directory for the output JSON.")
    args = parser.parse_args()

    if args.rounds < args.probe_interval:
        parser.error("--rounds must be >= --probe-interval")

    result = run_experiment(args)

    # ---- write output -------------------------------------------------
    model_tag = os.path.splitext(os.path.basename(args.model))[0]
    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(
        out_dir, f"r2_degradation_deep_{model_tag}_{args.rounds}.json"
    )
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\n  Results written to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
