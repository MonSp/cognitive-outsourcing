"""Benchmark metrics — initialization, fact extraction, answer quality, and averaging.

Merged from co_benchmark.py (the more complete version with per-turn TTF
tracking and tool/chat turn separation) and r2_benchmark.py.
"""

import re
from typing import List, Dict


def init_metrics() -> Dict:
    """Return a fresh metrics dictionary with all counters zeroed."""
    return {
        "total_ttf": 0.0,
        "total_gen_time": 0.0,
        "total_prefill_time": 0.0,
        "per_turn_ttf": [],
        "tool_turn_ttf": [],
        "chat_turn_ttf": [],
        "tool_calls_ok": 0,
        "total_tool_calls": 0,
        "final_answer": "",
        "peak_gpu_delta": 0.0,
        "total_gen_tokens": 0,
        "total_prefill_tokens": 0,
        "chain_depth": 0,
        "chain_total": 0,
        "tool_results_text": "",
    }


def extract_key_facts(tool_results_text: str) -> List[str]:
    """Extract key facts from assembled tool-result text.

    Parses ``[Result of tool_name(args)]: ...`` blocks and returns a flat
    list of result strings, proper-noun entities, numeric values, and
    argument values.
    """
    facts: List[str] = []
    for m in re.finditer(
        r"\[Result of (\w+)\(([^)]*)\)\]:\s*(.+?)(?=\n\[Result|\Z)",
        tool_results_text,
        re.DOTALL,
    ):
        tool_name = m.group(1)
        tool_args_str = m.group(2)
        result_text = m.group(3).strip()
        facts.append(result_text)
        for word in re.findall(r"[A-Z][a-zA-Z]+", result_text):
            if len(word) > 2 and word not in (
                "The",
                "This",
                "That",
                "And",
                "For",
                "With",
                "From",
                "Not",
                "But",
                "All",
                "Has",
                "Are",
                "Was",
                "Were",
                "Its",
                "Her",
            ):
                facts.append(word)
        for num in re.findall(r"\d+\.?\d*", result_text):
            facts.append(num)
        for arg_pair in tool_args_str.split(","):
            arg_pair = arg_pair.strip()
            if "=" in arg_pair:
                val = arg_pair.split("=", 1)[1].strip().strip('"').strip("'")
                if val:
                    facts.append(val)
    return facts


def evaluate_answer_quality(answer: str, tool_results_text: str) -> Dict:
    """Measure how well *answer* covers the key facts in *tool_results_text*.

    Returns a dict with:
        coverage:      fraction of unique facts found in the answer [0, 1]
        fact_count:    number of unique key facts
        matched_count: number of facts present in the answer
        answer_len:    character length of the answer
    """
    key_facts = extract_key_facts(tool_results_text) if tool_results_text else []
    unique_facts = list(set(key_facts))
    fact_count = len(unique_facts)

    if not answer or not unique_facts:
        return {
            "coverage": 0.0,
            "fact_count": fact_count,
            "matched_count": 0,
            "answer_len": len(answer),
        }

    answer_lower = answer.lower()
    matched = 0
    for fact in unique_facts:
        if fact.lower() in answer_lower:
            matched += 1

    coverage = matched / fact_count
    return {
        "coverage": coverage,
        "fact_count": fact_count,
        "matched_count": matched,
        "answer_len": len(answer),
    }


def average_metrics(runs: List[Dict]) -> Dict:
    """Average a list of per-run metrics dicts into a single summary.

    Computes means for scalar fields, standard deviations for timing
    fields, and takes the max for peak GPU delta.
    """
    if not runs:
        return {}
    if len(runs) == 1:
        return dict(runs[0])

    n = len(runs)
    avg: Dict = {}
    sum_fields = [
        "total_ttf",
        "total_gen_time",
        "total_prefill_time",
        "tool_calls_ok",
        "total_tool_calls",
        "total_gen_tokens",
        "total_prefill_tokens",
        "chain_depth",
        "chain_total",
    ]
    for f in sum_fields:
        avg[f] = sum(r.get(f, 0) for r in runs) / n

    std_fields = ["total_gen_time", "total_prefill_time"]
    for f in std_fields:
        mean = avg[f]
        if n > 1:
            variance = sum((r.get(f, 0) - mean) ** 2 for r in runs) / (n - 1)
            avg[f"{f}_std"] = variance ** 0.5
        else:
            avg[f"{f}_std"] = 0.0

    avg["peak_gpu_delta"] = max(r.get("peak_gpu_delta", 0) for r in runs)
    avg["final_answer"] = runs[0].get("final_answer", "")
    avg["tool_results_text"] = runs[0].get("tool_results_text", "")

    max_len = max(len(r.get("per_turn_ttf", [])) for r in runs)
    avg["per_turn_ttf"] = [
        sum(
            r.get("per_turn_ttf", [0] * max_len)[i]
            for r in runs
            if i < len(r.get("per_turn_ttf", []))
        )
        / sum(1 for r in runs if i < len(r.get("per_turn_ttf", [])))
        for i in range(max_len)
    ]

    for key in ["tool_turn_ttf", "chat_turn_ttf"]:
        vals: List[float] = []
        for r in runs:
            vals.extend(r.get(key, []))
        avg[key] = vals

    avg["correct_runs"] = n
    return avg
