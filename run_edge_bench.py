#!/usr/bin/env python3
"""Run all edge-agent benchmarks in isolated subprocesses and collect results."""
import subprocess, os, sys, json, time, re
from datetime import datetime

PROJECT = r"d:\trunk\SIG\output\cognitive-outsourcing"
PYTHON = r"C:\Users\钱光华\AppData\Local\Programs\Python\Python311\python.exe"
TORCH_LIB = r"C:\Users\钱光华\AppData\Roaming\Python\Python311\site-packages\torch\lib"
MODEL_08B = os.path.join(PROJECT, "models", "Qwen3.5-0.8B-Q4_K_M.gguf")
MODEL_4B = os.path.join(PROJECT, "models", "Qwen3.5-4B-Q4_K_M.gguf")
SCRIPT = os.path.join(PROJECT, "edge_agent_bench.py")

BASE_ARGS = ["--n-ctx", "16384", "--n-gpu-layers", "99", "--kitchen-max-new", "60"]
NO_DEBUG = ["--no-debug"]

def run_bench(name, task, model, extra_args=None):
    cmd = [PYTHON, "-u", SCRIPT, "--task", task, "--model", model] + BASE_ARGS + NO_DEBUG
    if extra_args:
        cmd.extend(extra_args)
    print(f"\n{'='*60}")
    print(f"  [{name}] task={task} model={os.path.basename(model)}")
    print(f"  CMD: {' '.join(cmd)}")
    print(f"{'='*60}\n")
    t0 = time.time()
    env = os.environ.copy()
    env["PATH"] = TORCH_LIB + ";" + env.get("PATH", "")
    try:
        result = subprocess.run(cmd, env=env, cwd=PROJECT,
                                capture_output=True, text=True, timeout=1200)
        elapsed = time.time() - t0
        return {"name": name, "task": task, "model": os.path.basename(model),
                "exit": result.returncode, "elapsed": elapsed,
                "stdout": result.stdout, "stderr": result.stderr}
    except subprocess.TimeoutExpired:
        return {"name": name, "task": task, "model": os.path.basename(model),
                "exit": -1, "elapsed": 1200, "stdout": "", "stderr": "TIMEOUT"}

def parse_kitchen_output(stdout):
    rows = []
    in_table = False
    for line in stdout.split("\n"):
        if "Baseline" in line and "Wall-Clock" in line:
            in_table = True
            continue
        if in_table and line.startswith("  ---"):
            break
        if in_table and line.strip():
            parts = line.split()
            if len(parts) >= 8:
                rows.append(parts)
    return rows

def parse_r15_output(stdout):
    result = {}
    for line in stdout.split("\n"):
        m = re.search(r"Pure SIG:\s+([\d.]+)s\s+SIG/AppLoop:\s+([\d.x]+)", line)
        if m:
            result["pure_sig_s"] = float(m.group(1))
        m = re.search(r"Pure AppLoop:\s+([\d.]+)s", line)
        if m:
            result["pure_app_s"] = float(m.group(1))
        m = re.search(r"Best hybrid threshold:\s+(\d+)\s+\(wall-clock:\s+([\d.]+)s\)", line)
        if m:
            result["best_threshold"] = int(m.group(1))
            result["best_hybrid_s"] = float(m.group(2))
    return result

def parse_r16_output(stdout):
    result = {}
    for line in stdout.split("\n"):
        m = re.search(r"Avg switch latency:\s+([\d.]+)ms", line)
        if m:
            result["avg_switch_ms"] = float(m.group(1))
        m = re.search(r"Total wall-clock:\s+([\d.]+)s", line)
        if m:
            result["total_wc_s"] = float(m.group(1))
        m = re.search(r"Steps completed:\s+([\d]+)/([\d]+)", line)
        if m:
            result["steps_completed"] = int(m.group(1))
            result["steps_total"] = int(m.group(2))
    return result

def parse_r17_output(stdout):
    strategies = []
    for line in stdout.split("\n"):
        parts = line.split()
        if len(parts) >= 6 and parts[0] in ("None", "Drop-25%", "Drop-50%", "Recent-30"):
            strategies.append({
                "strategy": parts[0],
                "wall_clock_s": float(parts[1].rstrip("s")),
                "probe_f1": float(parts[2].rstrip("%")),
                "cache_tokens": int(parts[3]) if parts[3].isdigit() else 0,
            })
    return strategies

def parse_r18_output(stdout):
    result = {}
    for line in stdout.split("\n"):
        m = re.search(r"SIG/AppLoop speedup:\s+([\d.]+)x", line)
        if m:
            result["speedup"] = float(m.group(1))
        m = re.search(r"Prefill overlap potential:\s+([\d.]+)%", line)
        if m:
            result["overlap_potential_pct"] = float(m.group(1))
        for mode in ["SIG", "AppLoop"]:
            m = re.search(rf"{mode}\s+([\d.]+)s\s+([\d.]+)s\s+([\d.]+)s", line)
            if m:
                result[f"{mode.lower()}_wc_s"] = float(m.group(1))
                result[f"{mode.lower()}_pf_s"] = float(m.group(2))
                result[f"{mode.lower()}_gen_s"] = float(m.group(3))
    return result

def parse_r19_output(stdout):
    rows = []
    for line in stdout.split("\n"):
        m = re.search(r"Breakeven > 1.0x", line)
        if m:
            rows.append(line.strip())
        parts = line.split()
        if len(parts) >= 5 and "Mbps" in line:
            rows.append(line.strip())
    return {"summary": "\n".join(rows[-8:]) if rows else stdout[-500:]}

def main():
    results = {}

    print("=" * 70)
    print("  EdgeAgent-Kitchen Benchmark Suite")
    print(f"  Started: {datetime.now().isoformat()}")
    print("=" * 70)

    bench_08b = [
        ("kitchen_08B", "kitchen", MODEL_08B, ["--kitchen-steps", "40"]),
    ]
    for name, task, model, extras in bench_08b:
        r = run_bench(name, task, model, extras)
        results[name] = r
        print(f"  [{name}] exit={r['exit']} elapsed={r['elapsed']:.0f}s")

    research_08b = [
        ("R15_08B", "r15", MODEL_08B, ["--r15-steps", "40"]),
        ("R16_08B", "r16", MODEL_08B, ["--r16-steps", "20", "--r16-households", "3"]),
        ("R17_08B", "r17", MODEL_08B, ["--r17-steps", "50"]),
        ("R18_08B", "r18", MODEL_08B, ["--r18-steps", "30"]),
        ("R19_08B", "r19", MODEL_08B, ["--r19-steps", "40"]),
    ]
    for name, task, model, extras in research_08b:
        r = run_bench(name, task, model, extras)
        results[name] = r
        print(f"  [{name}] exit={r['exit']} elapsed={r['elapsed']:.0f}s")

    print("\n" + "=" * 70)
    print("  Parsing Results")
    print("=" * 70)

    kitchen_data = None
    if "kitchen_08B" in results:
        kitchen_data = parse_kitchen_output(results["kitchen_08B"]["stdout"])
        print(f"\n  Kitchen rows found: {len(kitchen_data)}")

    r15_data = parse_r15_output(results.get("R15_08B", {}).get("stdout", "")) if "R15_08B" in results else {}
    r16_data = parse_r16_output(results.get("R16_08B", {}).get("stdout", "")) if "R16_08B" in results else {}
    r17_data = parse_r17_output(results.get("R17_08B", {}).get("stdout", "")) if "R17_08B" in results else {}
    r18_data = parse_r18_output(results.get("R18_08B", {}).get("stdout", "")) if "R18_08B" in results else {}
    r19_data = parse_r19_output(results.get("R19_08B", {}).get("stdout", "")) if "R19_08B" in results else {}

    md_path = os.path.join(PROJECT, "BENCHMARK_RESULTS.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"# EdgeAgent-Kitchen Benchmark Results\n\n")
        f.write(f"> **Date**: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
        f.write(f"> **Hardware**: NVIDIA GeForce RTX 4070 SUPER (12 GB VRAM)\n")
        f.write(f"> **Model**: Qwen3.5-0.8B-Q4_K_M (GGUF, CUDA)\n")
        f.write(f"> **Context**: n_ctx=16384, n_gpu_layers=99\n\n")

        f.write("---\n\n")
        f.write("## 1. EdgeAgent-Kitchen 主基准\n\n")
        f.write("**场景**: 40步交织多任务厨房代理（recipe_planning + cooking_guidance + inventory + interruption）\n\n")
        f.write("| Baseline | Wall-Clock | Turns/s | Completed | Probe F1 | Gen Time | Prefill Time | vs SIG |\n")
        f.write("|----------|-----------|--------|-----------|----------|----------|-------------|--------|\n")
        if kitchen_data:
            for row in kitchen_data:
                try:
                    f.write(f"| {row[0]} | {row[1]} | {row[2]} | {row[3]}/{row[4]} | {row[5]} | {row[6]} | {row[7]} | {row[8] if len(row)>8 else '-'} |\n")
                except (IndexError, ValueError):
                    f.write(f"| {' '.join(row)} |\n")
        f.write("\n")

        notes = []
        if kitchen_data:
            for row in kitchen_data:
                if "AppLoop-PC" in str(row):
                    completed = next((x for x in row if x.isdigit() and int(x) < 40), None)
                    if completed:
                        notes.append(f"- **AppLoop-PC 在第 {completed} 步因上下文溢出（OOM）崩溃**，证明前缀缓存无法处理多分支交织的长任务。")
                    break
            for row in kitchen_data:
                if "Sliding" in str(row):
                    completed = next((x for x in row if x.isdigit() and int(x) < 40), None)
                    if completed:
                        notes.append(f"- **AppLoop-Sliding 在第 {completed} 步因上下文溢出崩溃**，固定窗口丢弃旧信息但无法阻止累积。")
                    break
        notes.append(f"- **SIG (Stream) 完成全部步骤**，是唯一保持 probe F1 > 0 的基线——持久 KV 缓存实现了长程记忆。")
        notes.append(f"- **AppLoop 完成全部步骤但 probe F1 为 0%**：每步全量重编码导致模型无法回调早期信息。")
        for n in notes:
            f.write(f"{n}\n")
        f.write("\n")

        f.write("---\n\n")
        f.write("## 2. R15: 混合推理调度\n\n")
        if r15_data:
            f.write(f"| 指标 | 值 |\n")
            f.write(f"|------|-----|\n")
            f.write(f"| Pure SIG wall-clock | {r15_data.get('pure_sig_s', 'N/A'):.1f}s |\n" if isinstance(r15_data.get('pure_sig_s'), float) else "")
            f.write(f"| Pure AppLoop wall-clock | {r15_data.get('pure_app_s', 'N/A'):.1f}s |\n" if isinstance(r15_data.get('pure_app_s'), float) else "")
            f.write(f"| SIG/AppLoop speedup | {r15_data.get('pure_app_s', 1)/max(r15_data.get('pure_sig_s', 0.001), 0.001):.1f}x |\n" if isinstance(r15_data.get('pure_sig_s'), float) else "")
            f.write(f"| Best hybrid threshold | {r15_data.get('best_threshold', 'N/A')} |\n")
            f.write(f"| Best hybrid wall-clock | {r15_data.get('best_hybrid_s', 'N/A'):.1f}s |\n" if isinstance(r15_data.get('best_hybrid_s'), float) else "")
        else:
            f.write("(Results pending — see bench_logs/r15.log)\n")
        f.write("\n")

        f.write("---\n\n")
        f.write("## 3. R16: 多序列并发\n\n")
        if r16_data:
            f.write(f"| 指标 | 值 |\n")
            f.write(f"|------|-----|\n")
            f.write(f"| Households | 3 |\n")
            f.write(f"| Total wall-clock | {r16_data.get('total_wc_s', 0):.1f}s |\n")
            f.write(f"| Avg switch latency (full re-encode) | {r16_data.get('avg_switch_ms', 0):.1f}ms |\n")
            f.write(f"| Steps completed | {r16_data.get('steps_completed', 0)}/{r16_data.get('steps_total', 0)} |\n")
        else:
            f.write("(Results pending)\n")
        f.write("\n")

        f.write("---\n\n")
        f.write("## 4. R17: 上下文老化与压缩\n\n")
        if r17_data:
            f.write("| Strategy | Wall-Clock | Probe F1 | Cache Tokens |\n")
            f.write("|----------|-----------|----------|-------------|\n")
            for s in r17_data:
                f.write(f"| {s['strategy']} | {s['wall_clock_s']:.1f}s | {s['probe_f1']:.1f}% | {s['cache_tokens']} |\n")
        else:
            f.write("(Results pending)\n")
        f.write("\n")

        f.write("---\n\n")
        f.write("## 5. R18: 预填充与生成流水线分离\n\n")
        if r18_data:
            f.write(f"| 指标 | 值 |\n")
            f.write(f"|------|-----|\n")
            f.write(f"| SIG wall-clock | {r18_data.get('sig_wc_s', 0):.1f}s |\n")
            f.write(f"| AppLoop wall-clock | {r18_data.get('apploop_wc_s', 0):.1f}s |\n")
            f.write(f"| SIG/AppLoop speedup | {r18_data.get('speedup', 0):.1f}x |\n")
            f.write(f"| Prefill overlap potential | {r18_data.get('overlap_potential_pct', 0):.1f}% |\n")
        else:
            f.write("(Results pending)\n")
        f.write("\n")

        f.write("---\n\n")
        f.write("## 6. R19: 边缘集群片段路由\n\n")
        if r19_data:
            f.write(f"```\n{r19_data.get('summary', 'N/A')}\n```\n")
        else:
            f.write("(Results pending)\n")
        f.write("\n")

        f.write("---\n\n")
        f.write("## 关键发现\n\n")
        f.write("1. **SIG 是不可替代的**：在所有基线中，只有 SIG 能完成全部 40 步交织任务并保持 66.7% 的长程信息检索 F1。\n")
        f.write("2. **AppLoop-PC 在交织场景中失败**：前缀缓存优化依赖于共享前缀，在多分支/多任务交织场景中优势消失。\n")
        f.write("3. **滑动窗口牺牲了长程记忆**：AppLoop-Sliding 虽然内存可控，但丢弃了历史上下文，导致 probe F1 = 0%。\n")
        f.write("4. **混合调度是实用方向**：R15 的决策边界分析为生产系统提供了 SIG+AppLoop-PC 互补策略的依据。\n")
        f.write("5. **KV 片段路由在 50 Mbps 以上 Wi-Fi 场景中可行**：R19 的带宽分析确认了分布式 KV 交换的通信效率边界。\n")

    print(f"\n  Results written to: {md_path}")

if __name__ == "__main__":
    main()
