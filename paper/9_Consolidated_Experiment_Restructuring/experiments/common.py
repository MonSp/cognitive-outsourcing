"""
Common utilities for Paper 9 consolidated experiments.
=============================================
Shared infrastructure: subprocess runner, metric parser, statistical aggregation,
bootstrap confidence intervals, and data I/O.
"""

import subprocess, os, sys, json, time, math, random, re, hashlib
import numpy as np
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

PROJECT_ROOT = Path(r"d:\trunk\SIG\output\cognitive-outsourcing")
PYTHON_EXE = r"D:\ProgramData\miniconda3\envs\sig_bench\python.exe"
BENCH_SCRIPT = str(PROJECT_ROOT / "edge_agent_bench.py")
RESULTS_DIR = Path(__file__).parent.parent / "results"
FIGURES_DIR = Path(__file__).parent.parent / "figures"

RESULTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)

COMMON_ARGS = ["--n-ctx", "16384", "--n-gpu-layers", "99", "--no-debug"]

MODEL_PATHS = {
    "0.5B": str(PROJECT_ROOT / "models" / "Qwen3.5-0.5B-Q4_K_M.gguf"),
    "0.8B": str(PROJECT_ROOT / "models" / "Qwen3.5-0.8B-Q4_K_M.gguf"),
    "2B":   str(PROJECT_ROOT / "models" / "Qwen3.5-2B-Q4_K_M.gguf"),
    "4B":   str(PROJECT_ROOT / "models" / "Qwen3.5-4B-Q4_K_M.gguf"),
}

CROSS_ARCH_MODELS = {
    "gemma":     str(PROJECT_ROOT / "models" / "gemma-4-E2B-it-Q4_K_M.gguf"),
    "nemotron":  str(PROJECT_ROOT / "models" / "nvidia_Nemotron-3-Nano-4B-Q4_K_M.gguf"),
}

MTP_MODEL = str(PROJECT_ROOT / "models" / "mtp" / "Qwen3.5-4B-Q4_K_M.gguf")


def mean_std(values: List[float]) -> Tuple[float, float]:
    if not values:
        return 0.0, 0.0
    m = sum(values) / len(values)
    if len(values) < 2:
        return m, 0.0
    var = sum((v - m) ** 2 for v in values) / (len(values) - 1)
    return m, math.sqrt(var)


def bootstrap_ci(values: List[float], n_resamples: int = 10000,
                 confidence: float = 0.95, stat: str = "mean") -> Tuple[float, float, float]:
    arr = np.array(values, dtype=float)
    n = len(arr)
    if n < 2:
        return float(arr[0]) if n == 1 else 0.0, 0.0, 0.0
    rng = np.random.default_rng(42)
    indices = rng.integers(0, n, size=(n_resamples, n))
    if stat == "mean":
        samples = arr[indices].mean(axis=1)
    elif stat == "median":
        samples = np.median(arr[indices], axis=1)
    else:
        samples = arr[indices].mean(axis=1)
    alpha = (1 - confidence) / 2
    lo = float(np.percentile(samples, alpha * 100))
    hi = float(np.percentile(samples, (1 - alpha) * 100))
    return float(np.mean(samples)), lo, hi


def bootstrap_ratio_ci(numerators: List[float], denominators: List[float],
                       n_resamples: int = 10000, confidence: float = 0.95) -> Tuple[float, float, float]:
    num = np.array(numerators, dtype=float)
    den = np.array(denominators, dtype=float)
    n = len(num)
    rng = np.random.default_rng(42)
    indices = rng.integers(0, n, size=(n_resamples, n))
    ratios = num[indices].mean(axis=1) / den[indices].mean(axis=1)
    alpha = (1 - confidence) / 2
    lo = float(np.percentile(ratios, alpha * 100))
    hi = float(np.percentile(ratios, (1 - alpha) * 100))
    return float(np.mean(ratios)), lo, hi


def shapiro_wilk_p(values: List[float]) -> float:
    from scipy.stats import shapiro
    if len(values) < 3:
        return 1.0
    _, p = shapiro(values)
    return float(p)


def welch_t_test(a: List[float], b: List[float]) -> Tuple[float, float]:
    from scipy.stats import ttest_ind
    t_stat, p_val = ttest_ind(a, b, equal_var=False)
    return float(t_stat), float(p_val)


def cohens_d(a: List[float], b: List[float]) -> float:
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return 0.0
    ma, sa = mean_std(a)
    mb, sb = mean_std(b)
    pooled = math.sqrt(((na - 1) * sa**2 + (nb - 1) * sb**2) / (na + nb - 2))
    if pooled == 0:
        return 0.0
    return (ma - mb) / pooled


def benjamini_hochberg(p_values: List[float], fdr: float = 0.05) -> List[bool]:
    m = len(p_values)
    if m == 0:
        return []
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    rejected = [False] * m
    for rank, (idx, p) in enumerate(indexed, 1):
        threshold = (rank / m) * fdr
        if p <= threshold:
            rejected[idx] = True
        else:
            break
    return rejected


def run_subprocess(task: str, model_path: str, kitchen_steps: int = 35,
                   kitchen_max_new: int = 60, extra_args: Optional[List[str]] = None,
                   timeout: int = 900) -> Dict[str, Any]:
    cmd = [PYTHON_EXE, "-u", BENCH_SCRIPT,
           "--task", task,
           "--model", model_path] + COMMON_ARGS + [
           "--kitchen-steps", str(kitchen_steps),
           "--kitchen-max-new", str(kitchen_max_new)]
    if extra_args:
        cmd.extend(extra_args)
    t0 = time.time()
    try:
        r = subprocess.run(cmd, cwd=str(PROJECT_ROOT), capture_output=True,
                           text=True, timeout=timeout)
        elapsed = time.time() - t0
        return {"ok": r.returncode == 0, "elapsed": elapsed,
                "stdout": r.stdout, "stderr": r.stderr[-500:]}
    except subprocess.TimeoutExpired:
        return {"ok": False, "elapsed": timeout, "stdout": "", "stderr": "TIMEOUT"}
    except Exception as e:
        return {"ok": False, "elapsed": time.time() - t0, "stdout": "", "stderr": str(e)}


def parse_kitchen_metrics(stdout: str) -> Dict[str, Dict[str, float]]:
    results = {}
    line_re = re.compile(
        r'^\s*(SIG|AppLoop-PC|AppLoop-Sliding|SIG-Hybrid|AppLoop)'
        r'\s*\(([\d.]+)x\)\s+'
        r'([\d.]+)s\s+'
        r'([\d.]+)\s+'
        r'(\d+)\s+'
        r'([\d.]+)%\s+'
        r'([\d.]+)s\s+'
        r'([\d.]+)s'
    )
    for line in stdout.split("\n"):
        m = line_re.match(line)
        if m:
            name = m.group(1)
            results[name] = {
                "wall_clock_s": float(m.group(3)),
                "speedup": float(m.group(2)),
                "turns_per_s": float(m.group(4)),
                "completed": int(m.group(5)),
                "probe_f1": float(m.group(6)) / 100.0,
                "gen_s": float(m.group(7)),
                "prefill_s": float(m.group(8)),
            }
    return results


def parse_kitchen_json(stdout: str) -> Optional[Dict[str, Any]]:
    for line in stdout.split("\n"):
        line_s = line.strip()
        if line_s.startswith("{") and "wall_clock" in line_s:
            try:
                return json.loads(line_s)
            except json.JSONDecodeError:
                continue
    return None


def collect_run(experiment_id: str, condition: str, model: str, run_id: int,
                task: str, kitchen_steps: int, kitchen_max_new: int = 60,
                extra_args: Optional[List[str]] = None,
                evaluation_mode: str = "pre-scripted") -> Dict[str, Any]:
    result = run_subprocess(task, MODEL_PATHS.get(model, model),
                            kitchen_steps, kitchen_max_new, extra_args)
    record = {
        "experiment_id": experiment_id,
        "condition": condition,
        "model": model,
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(),
        "benchmark": task,
        "steps": kitchen_steps,
        "evaluation_mode": evaluation_mode,
        "wall_clock_s": result["elapsed"],
        "ok": result["ok"],
        "stderr_tail": result["stderr"][:200] if not result["ok"] else "",
    }
    metrics = parse_kitchen_metrics(result["stdout"])
    if metrics:
        record["parsed_baselines"] = metrics
    json_out = parse_kitchen_json(result["stdout"])
    if json_out:
        record["json_output"] = json_out
    return record


def save_run(experiment_id: str, slug: str, run_id: int, record: Dict[str, Any]):
    cond = record.get("condition", "unknown").replace("+", "p").replace("-", "m")
    fname = RESULTS_DIR / f"{experiment_id.lower()}_{slug}_{cond}_run_{run_id:02d}.json"
    with open(fname, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, ensure_ascii=False, default=str)
    return fname


def aggregate_experiment(experiment_id: str, slug: str,
                         all_records: List[Dict[str, Any]],
                         primary_metric: str = "wall_clock_s") -> Dict[str, Any]:
    by_cond = {}
    for rec in all_records:
        cond = rec["condition"]
        by_cond.setdefault(cond, []).append(rec)

    aggregation = {"experiment_id": experiment_id, "timestamp": datetime.now().isoformat(),
                   "conditions": {}}

    for cond, records in by_cond.items():
        valid = [r for r in records if r.get("ok", False)]
        values = [r.get(primary_metric, 0.0) for r in valid]
        if not values:
            aggregation["conditions"][cond] = {"n_valid": 0, "n_total": len(records)}
            continue
        m, s = mean_std(values)
        _, ci_lo, ci_hi = bootstrap_ci(values)
        sw_p = shapiro_wilk_p(values) if len(values) >= 3 else 1.0
        aggregation["conditions"][cond] = {
            "n_valid": len(valid), "n_total": len(records),
            f"{primary_metric}_mean": round(m, 4),
            f"{primary_metric}_std": round(s, 4),
            f"{primary_metric}_ci95_lo": round(ci_lo, 4),
            f"{primary_metric}_ci95_hi": round(ci_hi, 4),
            "shapiro_wilk_p": round(sw_p, 4),
        }

    agg_path = RESULTS_DIR / f"{experiment_id.lower()}_{slug}_aggregate.json"
    with open(agg_path, "w", encoding="utf-8") as f:
        json.dump(aggregation, f, indent=2, ensure_ascii=False, default=str)
    return aggregation


def randomized_run_order(experiment_id: str, conditions: List[str],
                         n_runs: int) -> List[Tuple[str, int]]:
    rng = random.Random(hashlib.md5((experiment_id + "2026").encode()).hexdigest())
    pairs = [(c, r) for c in conditions for r in range(n_runs)]
    rng.shuffle(pairs)
    return pairs


def orthogonality_ratio(s_sig_plus_x: float, s_sig: float, s_x: float) -> Optional[float]:
    if s_sig <= 0 or s_x <= 0:
        return None
    return s_sig_plus_x / (s_sig * s_x)
