#!/usr/bin/env python3
"""
Multi-Run EdgeAgent-Kitchen Benchmark Orchestrator
===================================================
Runs each benchmark task N times, parses metric lines from stdout,
computes mean ± std across runs, and produces:
  - bench_multi_results.json (raw + aggregated)
  - BENCHMARK_RESULTS.md (updated with error bars)

Usage:
  python run_multi_bench.py --model models/Qwen3.5-0.8B-Q4_K_M.gguf --runs 5
  python run_multi_bench.py --model models/Qwen3.5-4B-Q4_K_M.gguf --runs 3 --only kitchen,r15
"""

import subprocess, os, sys, json, time, re, math, argparse
from datetime import datetime

PROJECT = r"d:\trunk\SIG\output\cognitive-outsourcing"
PYTHON = r"C:\Users\钱光华\AppData\Local\Programs\Python\Python311\python.exe"
TORCH_LIB = r"C:\Users\钱光华\AppData\Roaming\Python\Python311\site-packages\torch\lib"
SCRIPT = os.path.join(PROJECT, "edge_agent_bench.py")

BASE_ARGS_08B = ["--n-ctx", "16384", "--n-gpu-layers", "99", "--no-debug"]
BASE_ARGS_4B = ["--n-ctx", "16384", "--n-gpu-layers", "99", "--no-debug"]

TASKS = {
    "kitchen": ["kitchen", ["--kitchen-steps", "30", "--kitchen-max-new", "60"]],
    "r15": ["r15", ["--r15-steps", "30", "--r15-max-new", "60"]],
    "r16": ["r16", ["--r16-steps", "15", "--r16-households", "3", "--r16-max-new", "50"]],
    "r17": ["r17", ["--r17-steps", "40", "--r17-max-new", "60"]],
    "r18": ["r18", ["--r18-steps", "25", "--r18-max-new", "80"]],
    "r19": ["r19", ["--r19-steps", "30", "--r19-max-new", "60"]],
}

CPU_ONLY_TASKS = set()


def mean_std(values):
    if not values:
        return 0.0, 0.0
    m = sum(values) / len(values)
    if len(values) < 2:
        return m, 0.0
    var = sum((v - m) ** 2 for v in values) / (len(values) - 1)
    return m, math.sqrt(var)


def run_one(task_id, model, base_args, no_gpu=False, timeout=900):
    t0 = time.time()
    task_name, task_extras = TASKS[task_id]
    cmd = [PYTHON, "-u", SCRIPT, "--task", task_name, "--model", model] + base_args + task_extras
    if no_gpu:
        cmd.append("--no-gpu")
    env = os.environ.copy()
    env["PATH"] = TORCH_LIB + ";" + env.get("PATH", "")
    try:
        r = subprocess.run(cmd, env=env, cwd=PROJECT,
                           capture_output=True, text=True, timeout=timeout)
        elapsed = time.time() - t0
        return {"ok": r.returncode == 0, "elapsed": elapsed,
                "stdout": r.stdout, "stderr": r.stderr[-300:]}
    except subprocess.TimeoutExpired:
        return {"ok": False, "elapsed": timeout, "stdout": "", "stderr": "TIMEOUT"}
    except Exception as e:
        return {"ok": False, "elapsed": time.time() - t0, "stdout": "", "stderr": str(e)}


# ── Parsers ─────────────────────────────────────────────────────────

def parse_kitchen(stdout):
    """Parse kitchen benchmark output using suffix-based anchoring."""
    out = {}
    in_result = False
    for line in stdout.split("\n"):
        if "Baseline" in line and "Wall-Clock" in line:
            in_result = True
            continue
        if in_result and "Hybrid split" in line:
            in_result = False
            continue
        if in_result and line.strip() and not line.strip().startswith("---"):
            stripped = line.strip()
            name_end = 0
            for ch in [":", "("]:
                idx = stripped.find(ch)
                if idx > 0:
                    name_end = idx
                    break
            if name_end == 0:
                name_end = stripped.find("  ")
                if name_end < 0:
                    name_end = 18
            name = stripped[:name_end].strip()
            if name in ("Baseline", "Hybrid", ""):
                continue
            suffix_nums = re.findall(r'\(([\d.]+)x\)', stripped)
            suffix = float(suffix_nums[0]) if suffix_nums else 0.0
            times = re.findall(r'([\d.]+)s', stripped)
            wc_s = float(times[0]) if len(times) >= 1 else 0.0
            gen_s = float(times[-2]) if len(times) >= 2 else 0.0
            pf_s = float(times[-1]) if len(times) >= 1 else 0.0
            f1_match = re.search(r'([\d.]+)%', stripped)
            f1 = float(f1_match.group(1)) / 100.0 if f1_match else 0.0
            done_match = re.search(r'(\d+)/(\d+)', stripped)
            if done_match:
                comp = int(done_match.group(1))
                tot = int(done_match.group(2))
            else:
                raw_nums = re.findall(r'(?<![/(\d.])(\d+)(?![s%x/(\d.])', stripped)
                raw_nums_int = [int(x) for x in raw_nums if 0 < int(x) < 1000]
                comp = raw_nums_int[0] if raw_nums_int else 0
                tot = comp
            tps_match = re.search(r'([\d.]+)\s+[\d.]+%', stripped)
            tps = 0.0
            if not tps_match:
                after_wc = stripped[stripped.find("s") + 1:].strip()
                tps_m = re.search(r'^([\d.]+)', after_wc)
                if tps_m:
                    tps = float(tps_m.group(1))
            out[name] = dict(wall_clock_s=wc_s, completed=comp, total_steps=tot,
                             probe_f1=f1, gen_s=gen_s, prefill_s=pf_s, turns_per_s=tps)
    return out


def parse_r15(stdout):
    out = {}
    for line in stdout.split("\n"):
        m = re.search(r"Pure SIG:\s+([\d.]+)s\s+SIG/AppLoop:\s+([\d.]+)x", line)
        if m:
            out["pure_sig_s"] = float(m.group(1))
            out["sig_app_speedup"] = float(m.group(2))
        m = re.search(r"Pure AppLoop:\s+([\d.]+)s", line)
        if m:
            out["pure_app_s"] = float(m.group(1))
        m = re.search(r"Best hybrid threshold:\s+(\d+)\s+\(wall-clock:\s+([\d.]+)s\)", line)
        if m:
            out["best_threshold"] = int(m.group(1))
            out["best_hybrid_s"] = float(m.group(2))
    out.setdefault("pure_sig_s", 0.0)
    out.setdefault("pure_app_s", 0.0)
    out.setdefault("sig_app_speedup", 0.0)
    return out


def parse_r16(stdout):
    out = {}
    for line in stdout.split("\n"):
        m = re.search(r"Avg switch latency:\s+([\d.]+)ms", line)
        if m:
            out["avg_switch_ms"] = float(m.group(1))
        m = re.search(r"Total wall-clock:\s+([\d.]+)s", line)
        if m:
            out["total_wc_s"] = float(m.group(1))
        m = re.search(r"Steps completed:\s+([\d]+)/([\d]+)", line)
        if m:
            out["completed"] = int(m.group(1))
            out["total"] = int(m.group(2))
    return out


def parse_r17(stdout):
    strategies = {}
    for line in stdout.split("\n"):
        parts = line.split()
        if len(parts) >= 4:
            name = parts[0]
            if name in ("None", "Drop-25%", "Drop-50%", "Recent-30"):
                wc = float(parts[1].rstrip("s"))
                f1 = float(parts[2].rstrip("%")) / 100.0
                cache = int(parts[3]) if parts[3].isdigit() else 0
                strategies[name] = dict(wall_clock_s=wc, probe_f1=f1, cache_tokens=cache)
    return strategies


def parse_r18(stdout):
    out = {}
    for line in stdout.split("\n"):
        m = re.search(r"SIG\s+([\d.]+)s\s+([\d.]+)s\s+([\d.]+)s", line)
        if m:
            out["sig_wc_s"] = float(m.group(1))
            out["sig_pf_s"] = float(m.group(2))
            out["sig_gen_s"] = float(m.group(3))
        m = re.search(r"AppLoop\s+([\d.]+)s\s+([\d.]+)s\s+([\d.]+)s", line)
        if m:
            out["app_wc_s"] = float(m.group(1))
            out["app_pf_s"] = float(m.group(2))
            out["app_gen_s"] = float(m.group(3))
        m = re.search(r"SIG/AppLoop speedup:\s+([\d.]+)x", line)
        if m:
            out["speedup"] = float(m.group(1))
        m = re.search(r"Prefill overlap potential:\s+([\d.]+)%", line)
        if m:
            out["overlap_pct"] = float(m.group(1))
    return out


def parse_r19(stdout):
    out = {}
    for line in stdout.split("\n"):
        m = re.search(r"Total tokens.*?(\d+)", line)
        if m:
            out["total_tokens"] = int(m.group(1))
        m = re.search(r"Total re-encode equivalent time:\s+([\d.]+)ms", line)
        if m:
            out["reencode_ms"] = float(m.group(1))
    return out


PARSERS = {"kitchen": parse_kitchen, "r15": parse_r15, "r16": parse_r16,
           "r17": parse_r17, "r18": parse_r18, "r19": parse_r19}


def aggregate_runs(run_results, task_id):
    """Aggregate N runs into mean±std dict."""
    parser = PARSERS[task_id]
    parsed_runs = []
    for rr in run_results:
        if rr["ok"]:
            p = parser(rr["stdout"])
            if p:
                parsed_runs.append(p)

    if not parsed_runs:
        return {"n_runs": len(run_results), "n_valid": 0, "error": "No valid parse"}

    agg = {"n_runs": len(run_results), "n_valid": len(parsed_runs)}

    if task_id == "kitchen":
        agents = set()
        for p in parsed_runs:
            agents.update(p.keys())
        for agent in sorted(agents):
            wcs = [p[agent]["wall_clock_s"] for p in parsed_runs if agent in p]
            f1s = [p[agent]["probe_f1"] for p in parsed_runs if agent in p]
            comps = [p[agent]["completed"] for p in parsed_runs if agent in p]
            gens = [p[agent]["gen_s"] for p in parsed_runs if agent in p]
            pfs = [p[agent]["prefill_s"] for p in parsed_runs if agent in p]
            m_wc, s_wc = mean_std(wcs)
            m_f1, s_f1 = mean_std(f1s)
            m_comp, s_comp = mean_std(comps)
            m_gen, s_gen = mean_std(gens)
            m_pf, s_pf = mean_std(pfs)
            agg[agent] = dict(wc_mean=m_wc, wc_std=s_wc, f1_mean=m_f1, f1_std=s_f1,
                              completed_mean=m_comp, completed_std=s_comp,
                              gen_mean=m_gen, gen_std=s_gen,
                              pf_mean=m_pf, pf_std=s_pf)
        if "SIG" in agg and isinstance(agg["SIG"], dict):
            sig_mean = agg["SIG"]["wc_mean"]
            for agent in sorted(agents):
                if agent in agg and isinstance(agg[agent], dict) and agent != "SIG":
                    agg[agent]["speedup_vs_sig"] = agg[agent]["wc_mean"] / max(sig_mean, 0.001)

    elif task_id == "r15":
        for k in ["pure_sig_s", "pure_app_s", "sig_app_speedup"]:
            vals = [p[k] for p in parsed_runs if k in p]
            m, s = mean_std(vals)
            agg[f"{k}_mean"] = m
            agg[f"{k}_std"] = s

    elif task_id == "r16":
        for k in ["avg_switch_ms", "total_wc_s"]:
            vals = [p[k] for p in parsed_runs if k in p]
            m, s = mean_std(vals)
            agg[f"{k}_mean"] = m
            agg[f"{k}_std"] = s

    elif task_id == "r17":
        strats = set()
        for p in parsed_runs:
            strats.update(p.keys())
        for s in sorted(strats):
            wcs = [p[s]["wall_clock_s"] for p in parsed_runs if s in p]
            f1s = [p[s]["probe_f1"] for p in parsed_runs if s in p]
            caches = [p[s]["cache_tokens"] for p in parsed_runs if s in p]
            m_w, s_w = mean_std(wcs)
            m_f, s_f = mean_std(f1s)
            m_c, s_c = mean_std(caches)
            agg[s] = dict(wc_mean=m_w, wc_std=s_w, f1_mean=m_f, f1_std=s_f,
                          cache_mean=m_c, cache_std=s_c)

    elif task_id == "r18":
        for k in ["speedup"]:
            vals = [p[k] for p in parsed_runs if k in p]
            m, s = mean_std(vals)
            agg[f"{k}_mean"] = m
            agg[f"{k}_std"] = s
        for k in ["sig_wc_s", "app_wc_s", "sig_pf_s", "app_pf_s", "overlap_pct"]:
            vals = [p[k] for p in parsed_runs if k in p]
            m, s = mean_std(vals)
            agg[f"{k}_mean"] = m
            agg[f"{k}_std"] = s

    elif task_id == "r19":
        for k in ["total_tokens", "reencode_ms"]:
            vals = [p[k] for p in parsed_runs if k in p]
            m, s = mean_std(vals)
            agg[f"{k}_mean"] = m
            agg[f"{k}_std"] = s

    return agg


def fmt_ws(mean, std, unit=""):
    if std < 0.005:
        return f"{mean:.1f}{unit}"
    return f"{mean:.1f}±{std:.1f}{unit}"


def fmt_pct(mean, std):
    return f"{mean*100:.1f}±{std*100:.1f}%"


def generate_report(all_agg, model_name):
    lines = []
    k_name = "kitchen"
    r15_name = "r15"
    r16_name = "r16"
    r17_name = "r17"
    r18_name = "r18"
    r19_name = "r19"

    lines.append(f"## {model_name} Multi-Run Results\n")

    # Kitchen
    if k_name in all_agg:
        a = all_agg[k_name]
        lines.append(f"### Kitchen ({a['n_valid']}/{a['n_runs']} valid runs)\n")
        lines.append("| Baseline | Wall-Clock | Completed | Probe F1 | Gen Time | Prefill Time | vs SIG |")
        lines.append("|----------|-----------|-----------|----------|----------|-------------|--------|")
        for agent in ["SIG", "AppLoop", "AppLoop-PC", "AppLoop-Sliding", "SIG-Hybrid"]:
            if agent in a and isinstance(a[agent], dict):
                d = a[agent]
                su = d.get("speedup_vs_sig", 1.0)
                lines.append(f"| {agent} | {fmt_ws(d['wc_mean'],d['wc_std'],'s')} | "
                             f"{d['completed_mean']:.0f}/{d.get('total_steps','?')} | "
                             f"{fmt_pct(d['f1_mean'],d['f1_std'])} | "
                             f"{fmt_ws(d['gen_mean'],0,'s')} | {fmt_ws(d['pf_mean'],0,'s')} | "
                             f"{su:.1f}× |")
        lines.append("")

    # R15
    if r15_name in all_agg:
        a = all_agg[r15_name]
        lines.append(f"### R15: Hybrid Scheduling ({a['n_valid']}/{a['n_runs']} valid runs)\n")
        lines.append(f"| Metric | Value |")
        lines.append(f"|--------|-------|")
        lines.append(f"| Pure SIG | {fmt_ws(a['pure_sig_s_mean'],a['pure_sig_s_std'],'s')} |")
        lines.append(f"| Pure AppLoop | {fmt_ws(a['pure_app_s_mean'],a['pure_app_s_std'],'s')} |")
        lines.append(f"| SIG/AppLoop Speedup | {fmt_ws(a['sig_app_speedup_mean'],a['sig_app_speedup_std'],'x')} |")
        lines.append("")

    # R16
    if r16_name in all_agg:
        a = all_agg[r16_name]
        lines.append(f"### R16: Multi-Sequence ({a['n_valid']}/{a['n_runs']} valid runs)\n")
        lines.append(f"| Metric | Value |")
        lines.append(f"|--------|-------|")
        lines.append(f"| Total Wall-Clock | {fmt_ws(a['total_wc_s_mean'],a['total_wc_s_std'],'s')} |")
        lines.append(f"| Avg Switch Latency | {fmt_ws(a['avg_switch_ms_mean'],a['avg_switch_ms_std'],'ms')} |")
        lines.append("")

    # R17
    if r17_name in all_agg:
        a = all_agg[r17_name]
        lines.append(f"### R17: Context Compression ({a['n_valid']}/{a['n_runs']} valid runs)\n")
        lines.append("| Strategy | Wall-Clock | Probe F1 | Cache Tokens |")
        lines.append("|----------|-----------|----------|-------------|")
        for s in ["None", "Drop-25%", "Drop-50%", "Recent-30"]:
            if s in a and isinstance(a[s], dict):
                d = a[s]
                lines.append(f"| {s} | {fmt_ws(d['wc_mean'],d['wc_std'],'s')} | "
                             f"{fmt_pct(d['f1_mean'],d['f1_std'])} | "
                             f"{fmt_ws(d['cache_mean'],d['cache_std'])} |")
        lines.append("")

    # R18
    if r18_name in all_agg:
        a = all_agg[r18_name]
        lines.append(f"### R18: Pipeline Separation ({a['n_valid']}/{a['n_runs']} valid runs)\n")
        lines.append(f"| Metric | Value |")
        lines.append(f"|--------|-------|")
        lines.append(f"| SIG Wall-Clock | {fmt_ws(a['sig_wc_s_mean'],a['sig_wc_s_std'],'s')} |")
        lines.append(f"| AppLoop Wall-Clock | {fmt_ws(a['app_wc_s_mean'],a['app_wc_s_std'],'s')} |")
        lines.append(f"| SIG/AppLoop Speedup | {fmt_ws(a['speedup_mean'],a['speedup_std'],'x')} |")
        lines.append(f"| SIG Prefill | {fmt_ws(a['sig_pf_s_mean'],a['sig_pf_s_std'],'s')} |")
        lines.append(f"| AppLoop Prefill | {fmt_ws(a['app_pf_s_mean'],a['app_pf_s_std'],'s')} |")
        lines.append(f"| Overlap Potential | {fmt_ws(a['overlap_pct_mean'],a['overlap_pct_std'],'%')} |")
        lines.append("")

    # R19
    if r19_name in all_agg:
        a = all_agg[r19_name]
        lines.append(f"### R19: Fragment Routing ({a['n_valid']}/{a['n_runs']} valid runs)\n")
        lines.append(f"| Metric | Value |")
        lines.append(f"|--------|-------|")
        lines.append(f"| Total Tokens | {fmt_ws(a['total_tokens_mean'],a['total_tokens_std'])} |")
        lines.append(f"| Re-encode Equivalent | {fmt_ws(a['reencode_ms_mean'],a['reencode_ms_std'],'ms')} |")
        lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Multi-Run EdgeAgent-Kitchen Benchmark")
    parser.add_argument("--model", required=True, help="Path to GGUF model")
    parser.add_argument("--runs", type=int, default=5, help="Runs per task")
    parser.add_argument("--only", type=str, default="", help="Comma-separated task IDs")
    parser.add_argument("--skip", type=str, default="", help="Comma-separated task IDs to skip")
    parser.add_argument("--model-name", type=str, default="", help="Display name for report")
    parser.add_argument("--no-gpu", action="store_true", default=False,
                        help="Force CPU inference for all tasks")
    args = parser.parse_args()

    only = set(x.strip() for x in args.only.split(",") if x.strip())
    skip = set(x.strip() for x in args.skip.split(",") if x.strip())
    if only:
        task_ids = [t for t in TASKS if t in only]
    else:
        task_ids = [t for t in TASKS if t not in skip]

    if not task_ids:
        print("No tasks selected.")
        return

    model_name = args.model_name or os.path.basename(args.model).replace(".gguf", "")

    print("=" * 70)
    print(f"  Multi-Run EdgeAgent-Kitchen Benchmark")
    print(f"  Model:  {model_name}")
    print(f"  Tasks:  {', '.join(task_ids)}")
    print(f"  Runs:   {args.runs} per task")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    base_args = BASE_ARGS_4B if "4B" in model_name else BASE_ARGS_08B
    all_raw = {}
    all_agg = {}
    total_ok, total_fail = 0, 0

    for task_id in task_ids:
        task_name = TASKS[task_id][0]
        print(f"\n  [{task_id}] Running {args.runs} times...")
        run_results = []
        for run_i in range(args.runs):
            print(f"    run {run_i + 1}/{args.runs}...", end=" ", flush=True)
            rr = run_one(task_id, args.model, base_args, no_gpu=args.no_gpu)
            run_results.append(rr)
            status = "OK" if rr["ok"] else "FAIL"
            print(f"{status} ({rr['elapsed']:.0f}s)")
            if rr["ok"]:
                total_ok += 1
            else:
                total_fail += 1
                if rr["stderr"]:
                    print(f"      stderr: {rr['stderr'][:200]}")

        all_raw[task_id] = run_results
        agg = aggregate_runs(run_results, task_id)
        all_agg[task_id] = agg
        print(f"    Aggregated: {agg.get('n_valid',0)}/{agg['n_runs']} valid parses")

    json_path = os.path.join(PROJECT, f"bench_multi_{model_name}.json")
    now_str = datetime.now().strftime('%Y-%m-%d %H:%M')
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"model": model_name, "runs_per_task": args.runs,
                   "started": datetime.now().isoformat(),
                   "total_ok": total_ok, "total_fail": total_fail,
                   "aggregated": _make_serializable(all_agg),
                   "raw_counts": {k: len(v) for k, v in all_raw.items()}},
                  f, indent=2, default=str, ensure_ascii=False)
    print(f"\n  Data saved to: {json_path}")

    report = generate_report(all_agg, model_name)
    md_path = os.path.join(PROJECT, f"bench_multi_{model_name}.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"# {model_name} Multi-Run Results\n\n"
                f"> **Date**: {now_str}\n"
                f"> **Runs**: {args.runs} per task, {total_ok} OK / {total_fail} FAIL\n\n"
                + report)
    print(f"  Partial report saved to: {md_path}")

    print(f"\n{'='*70}")
    print(f"  Done. {total_ok} OK, {total_fail} FAIL across {len(task_ids)} tasks × {args.runs} runs.")
    print(f"{'='*70}")


def _make_serializable(obj):
    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return len(obj)
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    return obj


if __name__ == "__main__":
    main()
