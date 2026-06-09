"""Task completion quality evaluators for CO/SIG benchmarks.

Evaluates whether model outputs achieve task goals, not just lexical overlap.
Designed to address the reviewer concern: "If the output is shorter, is the
task actually completed correctly?"

Evaluators:
  KitchenQualityEvaluator  — recipe completeness, inventory correctness, step compliance
  TravelQualityEvaluator   — city coverage, weather accuracy, flight completeness
  DevQualityEvaluator      — code correctness, test pass count, bug identification
  SemanticScorer           — TF-IDF based semantic similarity (no external deps)
"""

import re
import math
from typing import Dict, List, Tuple, Optional
from collections import Counter


def _extract_number(s: str) -> Optional[float]:
    m = re.search(r'[-+]?\d+\.?\d*', s.replace(",", ""))
    return float(m.group()) if m else None


def _tokenize(text: str) -> List[str]:
    normalized = text.replace("_", " ")
    return re.findall(r'[a-zA-Z0-9_]+', normalized.lower())


def _keyword_hit(keywords: List[str], text: str) -> float:
    text_lower = text.lower()
    hits = sum(1 for kw in keywords if kw.lower() in text_lower)
    return hits / max(len(keywords), 1)


def _overlap_score(expected_set: set, actual_set: set) -> float:
    if not expected_set:
        return 1.0
    return len(expected_set & actual_set) / len(expected_set)


class SemanticScorer:
    """TF-IDF based semantic similarity scorer (no external model needed).

    Computes cosine similarity between expected and actual text by
    constructing weighted token vectors.  More robust than pure keyword
    matching: handles underscore/spacing differences (spaghetti_bolognese
    vs "spaghetti bolognese"), partial matches, and token-level weighting.

    Usage::

        scorer = SemanticScorer(stopwords=False)
        score = scorer.similarity("spaghetti_bolognese chicken_stir_fry",
                                   "I made spaghetti bolognese and chicken")
    """

    def __init__(self, stopwords: bool = True, idf_smoothing: float = 0.5):
        self._use_stopwords = stopwords
        self._idf_smoothing = idf_smoothing
        self._doc_freq: Counter = Counter()

    def _compute_tf(self, tokens: List[str]) -> Dict[str, float]:
        if not tokens:
            return {}
        counts = Counter(tokens)
        return {t: c / len(tokens) for t, c in counts.items()}

    def _compute_idf(self, all_token_sets: List[List[str]]) -> Dict[str, float]:
        n_docs = len(all_token_sets)
        if n_docs == 0:
            return {}
        for tokens in all_token_sets:
            for t in set(tokens):
                self._doc_freq[t] += 1
        return {
            t: math.log((n_docs + 1) / (df + self._idf_smoothing)) + 1
            for t, df in self._doc_freq.items()
        }

    def _vectorize(self, tokens: List[str], idf: Dict[str, float]) -> Dict[str, float]:
        tf = self._compute_tf(tokens)
        return {t: tf[t] * idf.get(t, 1.0) for t in tf}

    def _dot(self, a: Dict[str, float], b: Dict[str, float]) -> float:
        return sum(a.get(k, 0) * b.get(k, 0) for k in set(a) | set(b))

    def _norm(self, v: Dict[str, float]) -> float:
        return math.sqrt(sum(val * val for val in v.values()))

    def similarity(self, expected_text: str, actual_text: str) -> float:
        """Compute TF-IDF cosine similarity between two text strings.

        Returns a score in [0, 1] where 1.0 = semantically identical.
        """
        if not expected_text or not actual_text:
            return 0.0
        et = _tokenize(expected_text)
        at = _tokenize(actual_text)
        if not et or not at:
            return 0.0
        idf = self._compute_idf([et, at])
        ev = self._vectorize(et, idf)
        av = self._vectorize(at, idf)
        dot = self._dot(ev, av)
        nrm = self._norm(ev) * self._norm(av)
        if nrm == 0:
            return 0.0
        return dot / nrm


def _semantic_hit(keywords: List[str], text: str,
                  scorer: Optional[SemanticScorer] = None) -> float:
    """Hybrid semantic scoring: keyword substring match + TF-IDF cosine.

    Falls back to pure keyword matching when *scorer* is None.
    """
    if scorer is None:
        return _keyword_hit(keywords, text)
    kws_text = " ".join(keywords)
    kw_score = _keyword_hit(keywords, text)
    sem_score = scorer.similarity(kws_text, text)
    return 0.4 * kw_score + 0.6 * sem_score


class KitchenQualityEvaluator:
    """Evaluates EdgeAgent-Kitchen task completion quality.

    Metrics:
      recipe_complete  — fraction of expected recipe steps mentioned
      inventory_match  — whether pantry/fridge state assertions match ground truth
      shopping_match   — whether shopping list matches expected items
      allergen_aware   — whether allergen constraints are respected
      tool_chain_follow — whether the correct sequence of tools was invoked
    """

    def __init__(self, ground_truth: Dict):
        self.gt = ground_truth
        self._scorer = SemanticScorer()

    def evaluate(self, final_answer: str, tool_call_log: List[Dict],
                 tool_results: List[str]) -> Dict[str, float]:
        scores = {}

        scores["recipe_mentioned"] = self._eval_recipe_mention(final_answer)
        scores["allergen_aware"] = self._eval_allergen(final_answer)
        scores["inventory_entities"] = self._eval_inventory(final_answer, tool_results)
        scores["shopping_list_items"] = self._eval_shopping(final_answer)
        scores["tool_execution_rate"] = self._eval_tool_chain(tool_call_log)

        scores["recipe_mentioned_kw"] = self._eval_recipe_mention_kw(final_answer)
        scores["inventory_entities_kw"] = self._eval_inventory_kw(final_answer, tool_results)
        scores["shopping_list_items_kw"] = self._eval_shopping_kw(final_answer)

        weights = {
            "recipe_mentioned": 0.25,
            "allergen_aware": 0.15,
            "inventory_entities": 0.20,
            "shopping_list_items": 0.15,
            "tool_execution_rate": 0.25,
        }
        composite = sum(scores[k] * weights[k] for k in weights)
        scores["composite"] = composite
        return scores

    def _eval_recipe_mention(self, answer: str) -> float:
        expected = self.gt.get("expected_recipes", [])
        if not expected:
            return 0.5
        return _semantic_hit(expected, answer, self._scorer)

    def _eval_recipe_mention_kw(self, answer: str) -> float:
        expected = self.gt.get("expected_recipes", [])
        if not expected:
            return 0.5
        return _keyword_hit(expected, answer)

    def _eval_allergen(self, answer: str) -> float:
        allergens = self.gt.get("allergens", [])
        if not allergens:
            return 1.0
        mentioned = any(a.lower() in answer.lower() for a in allergens)
        avoided = all(
            kw not in answer.lower()
            for kw in self.gt.get("forbidden_foods", [])
        )
        return (0.5 if mentioned else 0.0) + (0.5 if avoided else 0.0)

    def _eval_inventory(self, answer: str, tool_results: List[str]) -> float:
        expected_items = self.gt.get("inventory_items", [])
        if not expected_items:
            return 0.5
        all_text = answer + " " + " ".join(tool_results)
        return _semantic_hit(expected_items, all_text, self._scorer)

    def _eval_inventory_kw(self, answer: str, tool_results: List[str]) -> float:
        expected_items = self.gt.get("inventory_items", [])
        if not expected_items:
            return 0.5
        all_text = answer + " " + " ".join(tool_results)
        return _keyword_hit(expected_items, all_text)

    def _eval_shopping(self, answer: str) -> float:
        expected = self.gt.get("shopping_items", [])
        if not expected:
            return 1.0
        return _semantic_hit(expected, answer, self._scorer)

    def _eval_shopping_kw(self, answer: str) -> float:
        expected = self.gt.get("shopping_items", [])
        if not expected:
            return 1.0
        return _keyword_hit(expected, answer)

    def _eval_tool_chain(self, tool_log: List[Dict]) -> float:
        expected_tools = self.gt.get("expected_tool_sequence", [])
        if not expected_tools:
            return len(tool_log) > 0
        actual_names = [t.get("tool", "") for t in tool_log]
        hits = sum(1 for et in expected_tools if et in actual_names)
        return hits / len(expected_tools)


class TravelQualityEvaluator:
    """Evaluates CO travel scenario task completion quality.

    Metrics:
      city_coverage    — fraction of target cities covered in answer
      weather_accuracy — fraction of cities where weather matches tool output
      flight_info      — whether flight details appear
      attraction_count — how many specific attractions were named
    """

    def __init__(self, ground_truth: Dict):
        self.gt = ground_truth

    def evaluate(self, final_answer: str, tool_results: Dict[str, str]) -> Dict[str, float]:
        scores = {}
        target_cities = self.gt.get("target_cities", [])
        target_weather = self.gt.get("target_weather", {})
        target_flights = self.gt.get("target_flights", [])
        target_attractions = self.gt.get("target_attractions", [])

        if target_cities:
            scores["city_coverage"] = _keyword_hit(target_cities, final_answer)
        else:
            scores["city_coverage"] = 0.5

        if target_weather:
            hits = sum(
                1 for city, wx in target_weather.items()
                if city.lower() in final_answer.lower() and wx.lower() in final_answer.lower()
            )
            scores["weather_accuracy"] = hits / max(len(target_weather), 1)
        else:
            scores["weather_accuracy"] = 0.5

        if target_flights:
            scores["flight_info"] = _keyword_hit(target_flights, final_answer)
        else:
            scores["flight_info"] = 0.5

        if target_attractions:
            scores["attraction_count"] = _keyword_hit(target_attractions, final_answer)
        else:
            scores["attraction_count"] = 0.5

        weights = {
            "city_coverage": 0.30,
            "weather_accuracy": 0.25,
            "flight_info": 0.20,
            "attraction_count": 0.25,
        }
        composite = sum(scores[k] * weights[k] for k in weights)
        scores["composite"] = composite
        return scores


class DevQualityEvaluator:
    """Evaluates code debugging scenario task completion quality.

    Metrics:
      bug_identified   — whether the bug is named
      fix_suggested    — whether a fix is proposed
      test_count_match — whether test results are correctly reported
    """

    def __init__(self, ground_truth: Dict):
        self.gt = ground_truth

    def evaluate(self, final_answer: str) -> Dict[str, float]:
        expected_bug = self.gt.get("expected_bug", "")
        expected_fix = self.gt.get("expected_fix", "")
        expected_test_count = self.gt.get("expected_test_count", {})

        bug_hit = 1.0 if not expected_bug else (
            1.0 if expected_bug.lower() in final_answer.lower() else 0.0)
        fix_hit = 1.0 if not expected_fix else (
            1.0 if expected_fix.lower() in final_answer.lower() else 0.0)

        test_hit = 0.5
        if expected_test_count:
            pass_count = expected_test_count.get("passed", 0)
            fail_count = expected_test_count.get("failed", 0)
            found_pass = str(pass_count) in final_answer
            found_fail = str(fail_count) in final_answer
            test_hit = (0.5 if found_pass else 0.0) + (0.5 if found_fail else 0.0)

        return {
            "bug_identified": bug_hit,
            "fix_suggested": fix_hit,
            "test_count_match": test_hit,
            "composite": 0.40 * bug_hit + 0.40 * fix_hit + 0.20 * test_hit,
        }


def build_kitchen_ground_truth(scenario_steps) -> Dict:
    """Extract ground truth for Kitchen quality evaluation from scenario steps."""
    recipes_seen = set()
    shopping_items = set()
    inventory_items = set()
    tool_sequence = []
    allergens = set()

    for step in scenario_steps:
        tool_sequence.append(step.tool_name)
        if step.tool_name == "get_recipe":
            recipes_seen.add(step.tool_args.get("recipe_id", ""))
        elif step.tool_name == "add_shopping_item":
            shopping_items.add(step.tool_args.get("ingredient", ""))
        elif step.tool_name == "add_to_pantry" or step.tool_name == "add_to_fridge":
            inventory_items.add(step.tool_args.get("ingredient", ""))
        elif step.tool_name == "set_user_profile":
            al = step.tool_args.get("allergies", "")
            if al and al != "none":
                allergens.add(al)

    return {
        "expected_recipes": list(recipes_seen),
        "shopping_items": list(shopping_items),
        "inventory_items": list(inventory_items),
        "expected_tool_sequence": tool_sequence,
        "allergens": list(allergens),
        "forbidden_foods": [],
    }


class ContentQualityEvaluator:
    """Evaluates content-level quality of agent responses.

    Measures three dimensions that the simple tool_execution_rate metric
    cannot capture:
    1. Information Coverage — fraction of tool result facts reflected in
       the agent's generated responses.
    2. Response Quality — whether generated text is useful, non-repetitive,
       and appropriately concise.
    3. Context Utilisation — whether the model leverages injected context
       (tool results, harness state) effectively.
    """

    def __init__(self, scenario_steps):
        self._scorer = SemanticScorer()
        self._steps = scenario_steps

    def evaluate(self, gen_texts: List[str], tool_results: List[str]) -> Dict[str, float]:
        n = min(len(gen_texts), len(tool_results), len(self._steps))
        if n == 0:
            return {"information_coverage": 0.0, "response_quality": 0.0,
                    "context_utilisation": 0.0, "semantic_adequacy": 0.0,
                    "information_density": 0.0, "content_composite": 0.0}

        coverage_scores = []
        quality_scores = []
        adequacy_scores = []
        density_scores = []

        for i in range(n):
            gt = gen_texts[i] if i < len(gen_texts) else ""
            tr = tool_results[i] if i < len(tool_results) else ""
            step = self._steps[i] if i < len(self._steps) else None

            coverage_scores.append(self._information_coverage(gt, tr, step))
            quality_scores.append(self._response_quality(gt))
            adequacy_scores.append(self._semantic_adequacy(gt, tr))
            density_scores.append(self._information_density(gt))

        avg_coverage = sum(coverage_scores) / len(coverage_scores)
        avg_quality = sum(quality_scores) / len(quality_scores)
        context_util = self._context_utilisation(gen_texts, tool_results)
        avg_adequacy = sum(adequacy_scores) / len(adequacy_scores)
        avg_density = sum(density_scores) / len(density_scores)

        composite = (0.30 * avg_coverage + 0.20 * avg_quality
                     + 0.15 * context_util + 0.20 * avg_adequacy
                     + 0.15 * avg_density)

        return {
            "information_coverage": round(avg_coverage, 4),
            "response_quality": round(avg_quality, 4),
            "context_utilisation": round(context_util, 4),
            "semantic_adequacy": round(avg_adequacy, 4),
            "information_density": round(avg_density, 4),
            "content_composite": round(composite, 4),
            "coverage_detail": [round(s, 4) for s in coverage_scores],
        }

    def _information_coverage(self, gen_text: str, tool_result: str,
                              step=None) -> float:
        if not gen_text.strip() or not tool_result.strip():
            return 0.0

        result_tokens = set(_tokenize(tool_result))
        gen_tokens = set(_tokenize(gen_text))

        content_words = {w for w in result_tokens
                         if len(w) > 2 and w not in _STOPWORDS}
        gen_content = {w for w in gen_tokens
                       if len(w) > 2 and w not in _STOPWORDS}

        if not content_words:
            return 0.5

        overlap = content_words & gen_content
        coverage = len(overlap) / len(content_words)

        return min(1.0, coverage)

    def _response_quality(self, gen_text: str) -> float:
        text = gen_text.strip()
        if not text:
            return 0.0

        score = 0.0

        tokens = _tokenize(text)
        word_count = len(tokens)
        if word_count >= 3:
            score += 0.3
        elif word_count >= 1:
            score += 0.15

        if tokens:
            unique_ratio = len(set(tokens)) / len(tokens)
            score += 0.3 * min(1.0, unique_ratio * 1.5)

        if not self._has_repetition(text):
            score += 0.2
        else:
            score += 0.05

        density = self._information_density(gen_text)
        if 3 <= word_count <= 30 and density >= 0.4:
            score += 0.2
        elif density >= 0.35:
            score += 0.15
        elif density >= 0.25:
            score += 0.1
        else:
            score += 0.02

        return min(1.0, score)

    def _semantic_adequacy(self, gen_text: str, tool_result: str) -> float:
        """TF-IDF cosine similarity between tool result and response.

        Unlike coverage (which is set-based), this measures directional
        semantic alignment — how much of the response's meaning aligns
        with the tool result."""
        if not gen_text.strip() or not tool_result.strip():
            return 0.0
        return self._scorer.similarity(tool_result, gen_text)

    def _information_density(self, gen_text: str) -> float:
        """Ratio of content words to total words. Higher = more concise.

        Content words are non-stopwords with length > 2."""
        tokens = _tokenize(gen_text)
        if not tokens:
            return 0.0
        content = [t for t in tokens if len(t) > 2 and t not in _STOPWORDS]
        return len(content) / len(tokens)

    def _context_utilisation(self, gen_texts: List[str],
                             tool_results: List[str]) -> float:
        n = min(len(gen_texts), len(tool_results))
        if n == 0:
            return 0.0

        tool_vocab = set()
        for tr in tool_results:
            tool_vocab.update(_tokenize(tr))
        tool_vocab = {w for w in tool_vocab if len(w) > 2 and w not in _STOPWORDS}

        if not tool_vocab:
            return 0.5

        gen_vocab = set()
        for gt in gen_texts:
            gen_vocab.update(_tokenize(gt))
        gen_vocab = {w for w in gen_vocab if len(w) > 2 and w not in _STOPWORDS}

        if not gen_vocab:
            return 0.0

        overlap = tool_vocab & gen_vocab
        return min(1.0, len(overlap) / len(tool_vocab) * 1.2)

    def _has_repetition(self, text: str) -> bool:
        words = text.lower().split()
        if len(words) < 6:
            return False
        for pat_len in range(3, min(15, len(words) // 3)):
            for i in range(len(words) - pat_len * 2 + 1):
                pat = words[i:i + pat_len]
                rest = words[i + pat_len:]
                count = 0
                for j in range(len(rest) - pat_len + 1):
                    if rest[j:j + pat_len] == pat:
                        count += 1
                if count >= 2:
                    return True
        return False


_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "need", "must", "to", "of",
    "in", "for", "on", "with", "at", "by", "from", "as", "into", "about",
    "like", "through", "after", "over", "between", "out", "against",
    "during", "without", "before", "under", "around", "among", "and",
    "but", "or", "nor", "not", "so", "yet", "both", "either", "neither",
    "each", "every", "all", "any", "few", "more", "most", "other", "some",
    "such", "no", "only", "own", "same", "than", "too", "very", "just",
    "because", "if", "when", "where", "how", "what", "which", "who",
    "whom", "this", "that", "these", "those", "it", "its", "i", "me",
    "my", "we", "our", "you", "your", "he", "him", "his", "she", "her",
    "they", "them", "their", "here", "there", "then", "also", "still",
    "already", "now", "well", "back", "up", "down", "off", "above",
    "below", "again", "once", "further", "while", "until", "since",
}


def build_travel_ground_truth(scenario_turns: List[Dict]) -> Dict:
    """Extract ground truth for Travel quality evaluation from scenario turns."""
    target_cities = set()
    target_flights = []
    tool_results = {}

    for turn in scenario_turns:
        if not turn.get("tool"):
            continue
        args = turn.get("tool_args", {})
        for k, v in args.items():
            if v:
                target_cities.add(str(v).lower())
                if k in ("origin", "destination"):
                    target_flights.append(str(v))

    return {
        "target_cities": list(target_cities),
        "target_weather": {},
        "target_flights": target_flights,
        "target_attractions": sorted(target_cities),
    }
