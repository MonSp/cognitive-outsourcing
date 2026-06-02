#!/usr/bin/env python3
"""
R4 Teacher-Size Scan — Inverted-U Hypothesis Validation
======================================================

Scans multiple teacher model sizes (e.g., 1.5B / 4B / 7B) paired with a
fixed 0.8B student, to validate the hypothesis that an *optimal capability
gap* exists between teacher and student. The expectation is an inverted-U
shape: too small a teacher offers little extra signal, while too large a
teacher produces plans that overwhelm the student.

Per teacher-student pair, this script:
  1. Offline generates CoT plans for 9 benchmark scenarios using the
     teacher model (Llama / llama-cpp-python chat completion).
  2. Runs the 0.8B student in AppLoop mode against each plan and measures
     CoT comprehension rate.
  3. Runs the 0.8B student in SIG mode against each plan and measures
     final-answer quality via ``evaluate_answer_quality``.

Outputs JSON to ``data/r4_teacher_scan_results.json``.

Usage:
  python r4_teacher_scan.py \\
      --student models/Qwen3.5-0.8B-Q4_K_M.gguf \\
      --teachers models/Qwen3.5-1.5B-Q4_K_M.gguf \\
                  models/Qwen3.5-4B-Q4_K_M.gguf \\
      --runs 3

If a teacher model file is missing, that scale is skipped (with a recorded
reason) and the rest of the scan continues.
"""

import argparse
import json
import logging
import os
import re
import sys
import time
from typing import Dict, List, Optional, Tuple

from llama_cpp import Llama

from core import (
    MeaningCompiler,
    InjectionEngine,
    ToolRegistry,
    GPUMonitor,
    SYSTEM_PROMPT,
    SYSTEM_PROMPT_DEV,
    TOOL_DESCRIPTIONS_TRAVEL,
    TOOL_DESCRIPTIONS_DEV,
    TEACHER_PLANNING_PROMPT,
    NODE_PATTERN,
    evaluate_answer_quality,
    init_metrics,
    normalize_city,
)
from core.scenarios import (
    build_scenario1_long_sequence,
    build_scenario2_multi_tool_chain,
    build_scenario3_rapid_fire,
    build_scenario4_long_document,
    build_scenario5_mixed_conversation,
    build_scenario6_deep_tool_chain,
    build_scenario7_travel_planning_chain,
    build_scenario8_code_debugging_chain,
    build_scenario9_cross_reference_chain,
    BUILDERS,
)


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUTPUT = os.path.join(SCRIPT_DIR, "data", "r4_teacher_scan_results.json")

logger = logging.getLogger("r4_teacher_scan")


# ======================================================================
# Scenario configuration — (id, name, builder, is_conversation, system_prompt)
# ======================================================================
SCENARIO_CONFIG = [
    (1, "Long-seq",            build_scenario1_long_sequence,         True,  SYSTEM_PROMPT),
    (2, "Multi-tool",          build_scenario2_multi_tool_chain,      True,  SYSTEM_PROMPT),
    (3, "Rapid-fire",          build_scenario3_rapid_fire,            True,  SYSTEM_PROMPT),
    (4, "Long-doc",            build_scenario4_long_document,         True,  SYSTEM_PROMPT),
    (5, "Mixed",               build_scenario5_mixed_conversation,    True,  SYSTEM_PROMPT),
    (6, "Deep-chain",          build_scenario6_deep_tool_chain,       True,  SYSTEM_PROMPT),
    (7, "Travel-plan",         build_scenario7_travel_planning_chain, True,  SYSTEM_PROMPT),
    (8, "Code-debug",          build_scenario8_code_debugging_chain,  True,  SYSTEM_PROMPT_DEV),
    (9, "Cross-ref",           build_scenario9_cross_reference_chain, True,  SYSTEM_PROMPT),
]


# ======================================================================
# Plan parsing — copied from co_benchmark._parse_cot_plan
# ======================================================================
def _parse_cot_plan(content: str) -> dict:
    """Extract a {chain_of_thought, nodes} plan from raw LLM output."""
    json_match = re.search(r"\{[\s\S]*\}", content)
    if not json_match:
        return {"chain_of_thought": "", "nodes": {}}
    try:
        plan = json.loads(json_match.group())
    except json.JSONDecodeError:
        return {"chain_of_thought": "", "nodes": {}}
    cot = plan.get("chain_of_thought", plan.get("reasoning", ""))
    nodes = {}
    for key, val in plan.get("nodes", {}).items():
        name = val.get("tool") or val.get("name", "")
        args = val.get("arguments", {})
        if name and isinstance(args, dict):
            nodes[str(key)] = {"tool": name, "arguments": args}
    return {"chain_of_thought": cot, "nodes": nodes}


# ======================================================================
# Local teacher — uses Llama.chat_completion to generate CoT plans
# ======================================================================
class LocalTeacherPlanGenerator:
    """Generate CoT plans for benchmark scenarios using a local GGUF model."""

    def __init__(
        self,
        model_path: str,
        n_ctx: int = 4096,
        n_threads: int = 4,
        n_gpu_layers: int = 0,
        max_tokens: int = 2048,
        temperature: float = 0.0,
    ):
        logger.info("Loading teacher model: %s", model_path)
        self.model_path = model_path
        self.model = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_threads=n_threads,
            n_gpu_layers=n_gpu_layers,
            verbose=False,
        )
        self.max_tokens = max_tokens
        self.temperature = temperature

    def plan_tool_chain(self, query: str, scenario_id: int) -> dict:
        """Ask the teacher to produce a CoT plan for the given query."""
        tool_desc = (
            TOOL_DESCRIPTIONS_DEV if scenario_id == 8 else TOOL_DESCRIPTIONS_TRAVEL
        )
        system_prompt = TEACHER_PLANNING_PROMPT.format(tool_descriptions=tool_desc)
        try:
            output = self.model.create_chat_completion(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": query},
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )
            content = output["choices"][0]["message"]["content"].strip()
            return _parse_cot_plan(content)
        except Exception as exc:
            logger.warning("Teacher plan generation failed: %s", exc)
            return {"chain_of_thought": "", "nodes": {}}

    def close(self):
        try:
            del self.model
        except Exception:
            pass


# ======================================================================
# CoT assembly — mirror of co_benchmark.assemble_chain_of_thought
# ======================================================================
def assemble_chain_of_thought(
    cot: str,
    nodes: dict,
    module: ToolRegistry,
    expected_chain: list,
    metrics: dict,
    debug: bool = False,
) -> Tuple[str, int, int]:
    """Materialize tool results into a CoT template. Returns
    (assembled_cot, matched_count, total_calls)."""
    matched_count = 0
    matched_flags = [False] * len(expected_chain)
    total_calls = 0

    def replace_node(match):
        nonlocal matched_count, total_calls
        node_id = match.group(1)
        if node_id not in nodes:
            return f"[Node {node_id}: no tool specified]"
        node = nodes[node_id]
        tool_name = node["tool"]
        tool_args = node["arguments"]
        try:
            tool_result = module.execute(tool_name, tool_args)
        except Exception as exc:
            tool_result = f"[Error: {exc}]"
        matched = False
        for j, expected in enumerate(expected_chain):
            if matched_flags[j]:
                continue
            true_name = expected["tool"]
            true_args = expected["tool_args"]
            name_ok = (tool_name == true_name)
            args_ok = False
            if tool_args and true_args:
                normalized_parsed = {
                    k: normalize_city(str(v)) for k, v in tool_args.items()
                }
                normalized_true = {
                    k: normalize_city(str(v)) for k, v in true_args.items()
                }
                args_ok = normalized_parsed == normalized_true
            if name_ok and args_ok:
                matched_count += 1
                matched_flags[j] = True
                matched = True
                break
        total_calls += 1
        return (
            f"\n[Result of {tool_name}({', '.join(f'{k}={v}' for k, v in tool_args.items())})]: "
            f"{tool_result}\n"
        )

    assembled = NODE_PATTERN.sub(replace_node, cot)
    metrics["tool_calls_ok"] += matched_count
    metrics["total_tool_calls"] += total_calls
    return assembled, matched_count, total_calls


# ======================================================================
# Student-runner — AppLoop / SIG against a precomputed plan
# ======================================================================
def run_student_apploop(
    compiler: MeaningCompiler,
    module: ToolRegistry,
    plan: dict,
    expected_chain: list,
    query: str,
    system_prompt: str,
    max_new: int = 300,
) -> dict:
    """Run student in AppLoop mode: full re-prefill + generate."""
    metrics = init_metrics()
    metrics["chain_total"] = len(expected_chain)

    cot = plan.get("chain_of_thought", "")
    nodes = plan.get("nodes", {})

    assembled_cot, matched, _ = assemble_chain_of_thought(
        cot, nodes, module, expected_chain, metrics, debug=False
    )
    metrics["chain_depth"] = matched

    full_text = f"{system_prompt}\n\nUser: {query}\nAssistant:\n{assembled_cot}\n\nAnswer:"
    full_ids = list(compiler.tokenize(full_text, add_bos=False))
    compiler.reset_cache()
    pf_t0 = time.time()
    compiler.eval(full_ids)
    metrics["total_prefill_time"] += time.time() - pf_t0
    metrics["total_prefill_tokens"] += len(full_ids)

    cot_token_count = len(compiler.tokenize(assembled_cot, add_bos=False))
    gen_budget = max(max_new * 2 // 3, max_new - cot_token_count // 6)

    gen_t0 = time.time()
    gen_text, gen_ids = compiler.generate_until_str(
        "\nUser:", max_new=gen_budget, rep_threshold=3
    )
    metrics["total_gen_time"] += time.time() - gen_t0
    metrics["total_gen_tokens"] += len(gen_ids)

    metrics["total_ttf"] = metrics["total_gen_time"] + metrics["total_prefill_time"]
    metrics["final_answer"] = gen_text.strip()
    metrics["tool_results_text"] = assembled_cot
    return metrics


def run_student_sig(
    compiler: MeaningCompiler,
    module: ToolRegistry,
    plan: dict,
    expected_chain: list,
    query: str,
    system_prompt: str,
    max_new: int = 300,
) -> dict:
    """Run student in SIG mode: incremental cache injection + generate."""
    metrics = init_metrics()
    metrics["chain_total"] = len(expected_chain)

    engine = InjectionEngine(compiler)
    engine.reset()

    cot = plan.get("chain_of_thought", "")
    nodes = plan.get("nodes", {})

    assembled_cot, matched, _ = assemble_chain_of_thought(
        cot, nodes, module, expected_chain, metrics, debug=False
    )
    metrics["chain_depth"] = matched

    init_text = f"{system_prompt}\n\n"
    init_ids = list(compiler.tokenize(init_text, add_bos=False))
    compiler.reset_cache()
    pf_t0 = time.time()
    compiler.eval(init_ids)
    metrics["total_prefill_time"] += time.time() - pf_t0
    metrics["total_prefill_tokens"] += len(init_ids)
    engine.update_cache(init_ids)

    cot_block = f"User: {query}\nAssistant:\n{assembled_cot}\n\nAnswer:"
    cot_ids = list(compiler.tokenize(cot_block, add_bos=False))
    pf_t0 = time.time()
    compiler.eval(cot_ids)
    metrics["total_prefill_time"] += time.time() - pf_t0
    metrics["total_prefill_tokens"] += len(cot_ids)
    engine.update_cache(cot_ids)
    metrics["cache_injection_count"] += 1

    cot_token_count = len(compiler.tokenize(assembled_cot, add_bos=False))
    gen_budget = max(max_new * 2 // 3, max_new - cot_token_count // 6)

    gen_t0 = time.time()
    gen_text, gen_ids = compiler.generate_until_str(
        "\nUser:", max_new=gen_budget, rep_threshold=3
    )
    metrics["total_gen_time"] += time.time() - gen_t0
    metrics["total_gen_tokens"] += len(gen_ids)
    engine.update_cache(list(gen_ids))

    metrics["total_ttf"] = metrics["total_gen_time"] + metrics["total_prefill_time"]
    metrics["final_answer"] = gen_text.strip()
    metrics["tool_results_text"] = assembled_cot
    return metrics


# ======================================================================
# Helpers
# ======================================================================
def extract_user_query(turns_or_scenario) -> str:
    """Pull a user-facing query out of a scenario's turns list."""
    if isinstance(turns_or_scenario, tuple):
        _, turns = turns_or_scenario
    else:
        turns = turns_or_scenario
    for turn in turns:
        if turn.get("user"):
            return turn["user"]
    return turns[0].get("user", "") if turns else ""


def compute_comprehension_rate(metrics: dict) -> float:
    """CoT comprehension = tool_calls_ok / total_tool_calls in [0, 1]."""
    total = metrics.get("total_tool_calls", 0)
    if total <= 0:
        return 0.0
    return max(0.0, min(1.0, metrics.get("tool_calls_ok", 0) / total))


def compute_quality_composite(metrics: dict) -> float:
    """Use the existing evaluate_answer_quality coverage as the composite."""
    answer = metrics.get("final_answer", "")
    tool_text = metrics.get("tool_results_text", "")
    return evaluate_answer_quality(answer, tool_text).get("coverage", 0.0)


def _autonomous_apploop(
    compiler: MeaningCompiler,
    module: ToolRegistry,
    expected_chain: list,
    query: str,
    system_prompt: str,
    max_new: int = 300,
) -> dict:
    """Student autonomous (no CoT) AppLoop run — establishes the baseline."""
    metrics = init_metrics()
    metrics["chain_total"] = len(expected_chain)

    full_text = f"{system_prompt}\n\nUser: {query}\nAssistant:"
    full_ids = list(compiler.tokenize(full_text, add_bos=False))
    compiler.reset_cache()
    pf_t0 = time.time()
    compiler.eval(full_ids)
    metrics["total_prefill_time"] += time.time() - pf_t0
    metrics["total_prefill_tokens"] += len(full_ids)

    gen_t0 = time.time()
    gen_text, gen_ids = compiler.generate_until_str(
        "\nUser:", max_new=max_new, rep_threshold=3
    )
    metrics["total_gen_time"] += time.time() - gen_t0
    metrics["total_gen_tokens"] += len(gen_ids)

    metrics["total_ttf"] = metrics["total_gen_time"] + metrics["total_prefill_time"]
    metrics["final_answer"] = gen_text.strip()
    metrics["tool_results_text"] = ""
    return metrics


def build_scenario_payload(scenario_id: int) -> Tuple[str, list, str, str]:
    """Resolve scenario_id into (query, turns, system_prompt, description)."""
    for sid, name, builder, is_conversation, system_prompt in SCENARIO_CONFIG:
        if sid != scenario_id:
            continue
        if scenario_id == 4:
            sys_prompt, turns = builder()
        else:
            turns = builder()
            sys_prompt = system_prompt
        expected_chain = [t for t in turns if t.get("tool")]
        query = extract_user_query(turns)
        return name, expected_chain, sys_prompt, query
    raise KeyError(f"Unknown scenario_id={scenario_id}")


# ======================================================================
# Inverted-U test
# ======================================================================
def detect_inverted_u(xs: List[float], ys: List[float]) -> dict:
    """Test for an inverted-U shape in the (xs, ys) curve.

    Strategy:
      1. If fewer than 3 points, no conclusion.
      2. Find argmax index; check that interior point(s) exceed the
         endpoints and that the curve is not monotonically increasing or
         decreasing.
      3. Compute a simple "peakness" score = (peak - mean(endpoints)) /
         max(peak, 1e-9).
    """
    if len(xs) < 3 or len(ys) < 3:
        return {
            "inverted_u_detected": False,
            "reason": "need at least 3 data points",
            "peakness": 0.0,
            "peak_index": None,
        }

    peak_idx = max(range(len(ys)), key=lambda i: ys[i])
    peak_val = ys[peak_idx]

    if peak_idx == 0 or peak_idx == len(ys) - 1:
        return {
            "inverted_u_detected": False,
            "reason": "peak is at an endpoint — not an inverted U",
            "peakness": 0.0,
            "peak_index": int(peak_idx),
            "peak_x": float(xs[peak_idx]),
            "peak_y": float(peak_val),
        }

    left_endpoint = ys[0]
    right_endpoint = ys[-1]
    endpoint_mean = (left_endpoint + right_endpoint) / 2.0
    denom = max(abs(peak_val), 1e-9)
    peakness = (peak_val - endpoint_mean) / denom

    monotonic_inc = all(ys[i] <= ys[i + 1] for i in range(len(ys) - 1))
    monotonic_dec = all(ys[i] >= ys[i + 1] for i in range(len(ys) - 1))

    detected = (peakness > 0.05) and (not monotonic_inc) and (not monotonic_dec)

    return {
        "inverted_u_detected": bool(detected),
        "peakness": round(float(peakness), 4),
        "peak_index": int(peak_idx),
        "peak_x": float(xs[peak_idx]),
        "peak_y": float(peak_val),
        "left_endpoint_y": float(left_endpoint),
        "right_endpoint_y": float(right_endpoint),
        "monotonic_increasing": bool(monotonic_inc),
        "monotonic_decreasing": bool(monotonic_dec),
        "n_points": len(ys),
    }


# ======================================================================
# Per-teacher experiment
# ======================================================================
def run_teacher_scan(
    teacher_path: str,
    student_path: str,
    runs: int,
    max_new: int,
    n_ctx: int,
    n_threads: int,
    n_gpu_layers: int,
    debug: bool,
) -> dict:
    """Run the full per-teacher experiment. Returns the result dict."""
    teacher_label = os.path.basename(teacher_path)
    student_label = os.path.basename(student_path)
    print("\n" + "=" * 72)
    print(f"  R4 Teacher-Size Scan: teacher={teacher_label} | student={student_label}")
    print(f"  runs per scenario = {runs}, n_ctx = {n_ctx}")
    print("=" * 72)

    result: Dict = {
        "teacher_model": teacher_path,
        "student_model": student_path,
        "runs": runs,
        "per_scenario": {},
        "skipped": False,
        "skip_reason": None,
    }

    if not os.path.exists(teacher_path):
        result["skipped"] = True
        result["skip_reason"] = f"teacher model file not found: {teacher_path}"
        print(f"  SKIP: {result['skip_reason']}")
        return result

    print(f"\n[Phase 1/3] Loading teacher model for plan generation...")
    teacher = LocalTeacherPlanGenerator(
        model_path=teacher_path,
        n_ctx=n_ctx,
        n_threads=n_threads,
        n_gpu_layers=n_gpu_layers,
        max_tokens=2048,
        temperature=0.0,
    )

    print(f"\n[Phase 2/3] Teacher generates CoT plans for {len(SCENARIO_CONFIG)} scenarios...")
    plans_by_scenario: Dict[int, dict] = {}
    for sid, name, _, _, _ in SCENARIO_CONFIG:
        _, expected_chain, sys_prompt, query = build_scenario_payload(sid)
        if not query:
            print(f"  [S{sid} {name}] skipped: empty query")
            continue
        print(f"  [S{sid} {name}] generating plan...", end=" ", flush=True)
        plan = teacher.plan_tool_chain(query, sid)
        if not plan.get("chain_of_thought") or not plan.get("nodes"):
            print("FAILED")
            continue
        plan_node_count = len(plan["nodes"])
        plan_marker_count = len(re.findall(r"<<NODE:\d+>>", plan["chain_of_thought"]))
        print(f"OK (cot={len(plan['chain_of_thought'])}c, nodes={plan_node_count}, markers={plan_marker_count})")
        plans_by_scenario[sid] = plan

    teacher.close()

    if not plans_by_scenario:
        result["skipped"] = True
        result["skip_reason"] = "no teacher plans generated"
        return result

    print(f"\n[Phase 3/3] Loading student model: {student_path}")
    student = MeaningCompiler(
        model_path=student_path,
        n_ctx=n_ctx,
        n_threads=n_threads,
        n_gpu_layers=n_gpu_layers,
    )
    module = ToolRegistry()
    gpu = GPUMonitor()

    try:
        for sid, name, _, _, _ in SCENARIO_CONFIG:
            if sid not in plans_by_scenario:
                continue
            _, expected_chain, sys_prompt, query = build_scenario_payload(sid)
            plan = plans_by_scenario[sid]
            print(f"\n  --- Scenario {sid} {name} ({len(expected_chain)} tools) ---")

            apploop_runs: List[dict] = []
            sig_runs: List[dict] = []
            auto_runs: List[dict] = []

            for run_i in range(runs):
                app_m = run_student_apploop(
                    student, module, plan, expected_chain, query, sys_prompt, max_new=max_new
                )
                sig_m = run_student_sig(
                    student, module, plan, expected_chain, query, sys_prompt, max_new=max_new
                )
                auto_m = _autonomous_apploop(
                    student, module, expected_chain, query, sys_prompt, max_new=max_new
                )
                apploop_runs.append(app_m)
                sig_runs.append(sig_m)
                auto_runs.append(auto_m)
                if gpu is not None:
                    app_m["peak_gpu_delta"] = max(
                        app_m.get("peak_gpu_delta", 0), gpu.snapshot()["delta_mb"]
                    )

            app_comp = [compute_comprehension_rate(m) for m in apploop_runs]
            sig_qual = [compute_quality_composite(m) for m in sig_runs]
            auto_qual = [compute_quality_composite(m) for m in auto_runs]

            cot_comprehension_mean = sum(app_comp) / len(app_comp) if app_comp else 0.0
            sig_quality_mean = sum(sig_qual) / len(sig_qual) if sig_qual else 0.0
            auto_quality_mean = sum(auto_qual) / len(auto_qual) if auto_qual else 0.0

            perf_gain = 0.0
            if auto_quality_mean > 1e-9:
                perf_gain = (sig_quality_mean - auto_quality_mean) / auto_quality_mean

            print(
                f"  cot_comprehension={cot_comprehension_mean:.3f} | "
                f"sig_quality={sig_quality_mean:.3f} | "
                f"autonomous_quality={auto_quality_mean:.3f} | "
                f"perf_gain={perf_gain:+.3f}"
            )

            result["per_scenario"][sid] = {
                "name": name,
                "expected_tools": len(expected_chain),
                "plan_nodes": len(plan.get("nodes", {})),
                "cot_comprehension_rate": round(cot_comprehension_mean, 4),
                "cot_comprehension_runs": [round(v, 4) for v in app_comp],
                "sig_quality": round(sig_quality_mean, 4),
                "sig_quality_runs": [round(v, 4) for v in sig_qual],
                "autonomous_quality": round(auto_quality_mean, 4),
                "autonomous_quality_runs": [round(v, 4) for v in auto_qual],
                "performance_gain": round(perf_gain, 4),
            }
    finally:
        if gpu is not None:
            gpu.shutdown()

    comp_rates = [v["cot_comprehension_rate"] for v in result["per_scenario"].values()]
    perf_gains = [v["performance_gain"] for v in result["per_scenario"].values()]

    if comp_rates:
        result["aggregate"] = {
            "mean_cot_comprehension": round(sum(comp_rates) / len(comp_rates), 4),
            "mean_sig_quality": round(
                sum(v["sig_quality"] for v in result["per_scenario"].values())
                / max(len(result["per_scenario"]), 1),
                4,
            ),
            "mean_autonomous_quality": round(
                sum(v["autonomous_quality"] for v in result["per_scenario"].values())
                / max(len(result["per_scenario"]), 1),
                4,
            ),
            "mean_performance_gain": round(
                sum(perf_gains) / max(len(perf_gains), 1), 4
            ),
            "n_scenarios": len(comp_rates),
        }
    else:
        result["aggregate"] = {
            "mean_cot_comprehension": 0.0,
            "mean_sig_quality": 0.0,
            "mean_autonomous_quality": 0.0,
            "mean_performance_gain": 0.0,
            "n_scenarios": 0,
        }

    return result


# ======================================================================
# Main
# ======================================================================
def parse_size_b(label: str) -> float:
    """Extract the leading float from a model name like 'Qwen3.5-4B-...'."""
    m = re.search(r"(\d+(?:\.\d+)?)\s*B", label, re.IGNORECASE)
    if m:
        return float(m.group(1))
    return 0.0


def main():
    parser = argparse.ArgumentParser(
        description="R4 Teacher-Size Scan — Inverted-U Hypothesis Validation"
    )
    parser.add_argument(
        "--student",
        type=str,
        required=True,
        help="Path to student GGUF model (e.g. models/Qwen3.5-0.8B-Q4_K_M.gguf)",
    )
    parser.add_argument(
        "--teachers",
        type=str,
        nargs="+",
        required=True,
        help="One or more paths to teacher GGUF models",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=3,
        help="Number of student runs per scenario per teacher (default: 3)",
    )
    parser.add_argument("--n-ctx", type=int, default=4096)
    parser.add_argument("--n-threads", type=int, default=4)
    parser.add_argument("--n-gpu-layers", type=int, default=0)
    parser.add_argument("--max-new", type=int, default=300)
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument(
        "--output",
        type=str,
        default=DEFAULT_OUTPUT,
        help="Path to output JSON (default: data/r4_teacher_scan_results.json)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )

    if not os.path.exists(args.student):
        print(f"ERROR: student model not found: {args.student}")
        sys.exit(1)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    per_teacher: Dict[str, dict] = {}
    sizes_order: List[float] = []
    for teacher_path in args.teachers:
        res = run_teacher_scan(
            teacher_path=teacher_path,
            student_path=args.student,
            runs=args.runs,
            max_new=args.max_new,
            n_ctx=args.n_ctx,
            n_threads=args.n_threads,
            n_gpu_layers=args.n_gpu_layers,
            debug=args.debug,
        )
        key = os.path.basename(teacher_path)
        per_teacher[key] = res
        if not res.get("skipped"):
            sizes_order.append((parse_size_b(key), key))

    sizes_order.sort(key=lambda t: t[0])

    curve_comprehension: List[dict] = []
    curve_performance_gain: List[dict] = []
    for size_b, key in sizes_order:
        agg = per_teacher[key].get("aggregate", {})
        curve_comprehension.append(
            {"teacher_size_b": size_b, "cot_comprehension": agg.get("mean_cot_comprehension", 0.0)}
        )
        curve_performance_gain.append(
            {
                "teacher_size_b": size_b,
                "performance_gain": agg.get("mean_performance_gain", 0.0),
            }
        )

    xs_comp = [pt["teacher_size_b"] for pt in curve_comprehension]
    ys_comp = [pt["cot_comprehension"] for pt in curve_comprehension]
    xs_gain = [pt["teacher_size_b"] for pt in curve_performance_gain]
    ys_gain = [pt["performance_gain"] for pt in curve_performance_gain]

    inverted_u_comprehension = detect_inverted_u(xs_comp, ys_comp)
    inverted_u_gain = detect_inverted_u(xs_gain, ys_gain)

    summary = {
        "student_model": args.student,
        "n_teachers_total": len(args.teachers),
        "n_teachers_run": len([r for r in per_teacher.values() if not r.get("skipped")]),
        "n_teachers_skipped": len([r for r in per_teacher.values() if r.get("skipped")]),
        "curve_cot_comprehension_vs_size": curve_comprehension,
        "curve_performance_gain_vs_size": curve_performance_gain,
        "inverted_u_test": {
            "comprehension_rate": inverted_u_comprehension,
            "performance_gain": inverted_u_gain,
        },
    }

    overall = {
        "summary": summary,
        "per_teacher": per_teacher,
    }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(overall, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 72)
    print("  R4 Teacher-Size Scan Summary")
    print("=" * 72)
    print(f"  Student: {args.student}")
    print(f"  Teachers requested: {len(args.teachers)} | "
          f"run: {summary['n_teachers_run']} | skipped: {summary['n_teachers_skipped']}")
    print(f"\n  CoT comprehension rate vs teacher size:")
    for pt in curve_comprehension:
        print(f"    {pt['teacher_size_b']:.2f}B -> {pt['cot_comprehension']:.3f}")
    print(f"\n  Performance gain vs teacher size:")
    for pt in curve_performance_gain:
        print(f"    {pt['teacher_size_b']:.2f}B -> {pt['performance_gain']:+.3f}")
    print(f"\n  Inverted-U test (CoT comprehension): {inverted_u_comprehension}")
    print(f"  Inverted-U test (performance gain):    {inverted_u_gain}")
    print(f"\n  Results written to: {args.output}")


if __name__ == "__main__":
    main()
