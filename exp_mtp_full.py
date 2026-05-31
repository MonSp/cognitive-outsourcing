"""Full MTP experiment: 4 conditions x 5 runs each
Compares: baseline (no MTP) vs MTP with draft-n-max = 1, 2, 3
Also tests SIG + MTP combination.
"""
import json
import time
import subprocess
import sys
import os
from pathlib import Path
from dataclasses import dataclass, asdict

LLAMA_SERVER = Path("llama-cpp-bin/llama-server.exe")
MODEL_MTP = Path("models/mtp/Qwen3.5-4B-Q4_K_M.gguf")
PORT = 8081

@dataclass
class BenchResult:
    condition: str
    run_id: int
    tokens_predicted: int
    tokens_evaluated: int
    predicted_per_second: float
    draft_n: int
    draft_accepted: int
    prompt_per_second: float
    total_ms: float

def start_server(extra_args=None):
    cmd = [
        str(LLAMA_SERVER),
        "-m", str(MODEL_MTP),
        "-c", "4096",
        "-ngl", "99",
        "--port", str(PORT),
        "--host", "127.0.0.1",
        "-np", "1",
    ]
    if extra_args:
        cmd.extend(extra_args)
    
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0,
    )
    
    for _ in range(60):
        time.sleep(1)
        try:
            import urllib.request
            req = urllib.request.Request(f"http://127.0.0.1:{PORT}/health")
            with urllib.request.urlopen(req, timeout=2) as resp:
                if resp.status == 200:
                    return proc
        except:
            pass
    raise RuntimeError("Server failed to start")

def stop_server(proc):
    if proc:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()

def run_benchmark(condition, extra_args=None, n_runs=5):
    results = []
    proc = start_server(extra_args)
    try:
        for run_id in range(n_runs):
            body = json.dumps({
                "prompt": "You are a kitchen assistant.\n\nUser: What recipes can I make with chicken?\nAssistant:",
                "n_predict": 60,
                "temperature": 0.0,
            })
            try:
                import urllib.request
                req = urllib.request.Request(
                    f"http://127.0.0.1:{PORT}/completion",
                    data=body.encode(),
                    headers={"Content-Type": "application/json"},
                )
                with urllib.request.urlopen(req, timeout=120) as resp:
                    data = json.loads(resp.read().decode())
                results.append(BenchResult(
                    condition=condition,
                    run_id=run_id,
                    tokens_predicted=data["tokens_predicted"],
                    tokens_evaluated=data["tokens_evaluated"],
                    predicted_per_second=data["timings"]["predicted_per_second"],
                    draft_n=data["timings"].get("draft_n", 0),
                    draft_accepted=data["timings"].get("draft_accepted", 0),
                    prompt_per_second=data["timings"]["prompt_per_second"],
                    total_ms=data["timings"]["predicted_ms"],
                ))
                print(f"  Run {run_id}: {data['timings']['predicted_per_second']:.1f} tok/s, draft={data['timings'].get('draft_n',0)}/{data['timings'].get('draft_accepted',0)}")
            except Exception as e:
                print(f"  Run {run_id} failed: {e}")
    finally:
        stop_server(proc)
    return results

def main():
    all_results = {}
    
    print("=" * 60)
    print("MTP Experiment: Qwen3.5-4B-Q4_K_M")
    print("=" * 60)
    
    # Condition 1: Baseline (no MTP)
    print("\n[1/4] Baseline (no MTP)...")
    all_results["baseline"] = run_benchmark("baseline", n_runs=5)
    
    # Condition 2: MTP with draft-n-max=1
    print("\n[2/4] MTP draft-n-max=1...")
    all_results["mtp_n1"] = run_benchmark("mtp_n1", ["--spec-type", "draft-mtp", "--spec-draft-n-max", "1"], n_runs=5)
    
    # Condition 3: MTP with draft-n-max=2
    print("\n[3/4] MTP draft-n-max=2...")
    all_results["mtp_n2"] = run_benchmark("mtp_n2", ["--spec-type", "draft-mtp", "--spec-draft-n-max", "2"], n_runs=5)
    
    # Condition 4: MTP with draft-n-max=3
    print("\n[4/4] MTP draft-n-max=3...")
    all_results["mtp_n3"] = run_benchmark("mtp_n3", ["--spec-type", "draft-mtp", "--spec-draft-n-max", "3"], n_runs=5)
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Condition':<20} {'tok/s':>10} {'draft_n':>8} {'draft_acc':>10} {'prompt_tps':>12}")
    print("-" * 60)
    for cond, results in all_results.items():
        avg_tps = sum(r.predicted_per_second for r in results) / len(results)
        avg_draft_n = sum(r.draft_n for r in results) / len(results)
        avg_draft_acc = sum(r.draft_accepted for r in results) / max(sum(r.draft_n for r in results), 1)
        avg_prompt_tps = sum(r.prompt_per_second for r in results) / len(results)
        print(f"{cond:<20} {avg_tps:>10.1f} {avg_draft_n:>8.1f} {avg_draft_acc:>10.1%} {avg_prompt_tps:>12.1f}")
    
    # Save results
    out_path = Path("data/exp_mtp_4B_results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        save_data = {k: [asdict(r) for r in v] for k, v in all_results.items()}
    json.dump(save_data, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {out_path}")

if __name__ == "__main__":
    main()
