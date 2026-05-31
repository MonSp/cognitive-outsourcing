"""Improved MTP Experiment v2 — fixes all issues from v1.

Fixes:
  1. Correct field name: draft_n_accepted (not draft_accepted)
  2. Real SIG simulation via llama-server prompt caching
  3. Proper speedup calculation (exclude cache-hit anomalies)
  4. JSON save bug fixed
  5. Comparison with paper's n-gram SpecDec via llama-cpp-python
  6. Per-token draft stats and acceptance rate breakdown
  7. Both 4B and 0.8B models
"""
import json
import time
import subprocess
import sys
import os
import urllib.request
import urllib.error
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Optional, Tuple

LLAMA_SERVER = Path("llama-cpp-bin/llama-server.exe")
MODEL_DIR = Path("models")
PORT = 8081


@dataclass
class BenchResult:
    condition: str
    run_id: int
    tokens_predicted: int
    tokens_evaluated: int
    predicted_per_second: float
    predicted_ms: float
    draft_n: int
    draft_n_accepted: int
    prompt_n: int
    prompt_ms: float
    prompt_per_second: float
    tokens_cached: int
    cache_hit: bool
    error: str = ""

    @property
    def acceptance_rate(self) -> float:
        return self.draft_n_accepted / self.draft_n if self.draft_n > 0 else 0.0

    @property
    def effective_speedup(self) -> float:
        if self.draft_n == 0:
            return 1.0
        return (self.tokens_predicted + self.draft_n_accepted) / max(self.tokens_predicted, 1)


@dataclass
class SIGTurnResult:
    turn_id: int
    prompt_tokens: int
    prompt_cached_tokens: int
    prompt_eval_tokens: int
    prompt_ms: float
    gen_tokens: int
    gen_ms: float
    gen_tok_per_s: float
    draft_n: int
    draft_n_accepted: int
    total_context_tokens: int


def start_server(model_path: str, extra_args: Optional[List[str]] = None,
                 port: int = PORT, timeout: float = 120) -> subprocess.Popen:
    cmd = [
        str(LLAMA_SERVER),
        "-m", model_path,
        "-c", "16384",
        "-ngl", "99",
        "--port", str(port),
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

    for _ in range(int(timeout)):
        time.sleep(1)
        try:
            req = urllib.request.Request(f"http://127.0.0.1:{port}/health")
            with urllib.request.urlopen(req, timeout=2) as resp:
                if resp.status == 200:
                    return proc
        except:
            pass
    proc.kill()
    raise RuntimeError("Server failed to start within timeout")


def stop_server(proc: Optional[subprocess.Popen]):
    if proc:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()


def run_completion(prompt: str, n_predict: int = 60, temperature: float = 0.0,
                   port: int = PORT) -> Dict:
    body = json.dumps({
        "prompt": prompt,
        "n_predict": n_predict,
        "temperature": temperature,
    })
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/completion",
        data=body.encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=300) as resp:
        return json.loads(resp.read().decode())


def parse_result(data: Dict, condition: str, run_id: int) -> BenchResult:
    timings = data.get("timings", {})
    prompt_n = timings.get("prompt_n", 0)
    prompt_ms = timings.get("prompt_ms", 0)
    predicted_ms = timings.get("predicted_ms", 0)
    tokens_evaluated = data.get("tokens_evaluated", 0)
    tokens_cached = data.get("tokens_cached", 0)

    cache_hit = tokens_evaluated < prompt_n if prompt_n > 0 else False

    return BenchResult(
        condition=condition,
        run_id=run_id,
        tokens_predicted=data.get("tokens_predicted", 0),
        tokens_evaluated=tokens_evaluated,
        predicted_per_second=timings.get("predicted_per_second", 0),
        predicted_ms=predicted_ms,
        draft_n=timings.get("draft_n", 0),
        draft_n_accepted=timings.get("draft_n_accepted", 0),
        prompt_n=prompt_n,
        prompt_ms=prompt_ms,
        prompt_per_second=timings.get("prompt_per_second", 0),
        tokens_cached=tokens_cached,
        cache_hit=cache_hit,
    )


# ── Experiment 1: MTP Speed Benchmark ─────────────────────────────

def run_mtp_benchmark(model_path: str, model_name: str, n_runs: int = 5):
    conditions = [
        ("baseline", []),
        ("mtp_n1", ["--spec-type", "draft-mtp", "--spec-draft-n-max", "1"]),
        ("mtp_n2", ["--spec-type", "draft-mtp", "--spec-draft-n-max", "2"]),
        ("mtp_n3", ["--spec-type", "draft-mtp", "--spec-draft-n-max", "3"]),
    ]

    prompt = "You are a kitchen assistant.\n\nUser: What recipes can I make with chicken?\nAssistant:"
    all_results: Dict[str, List[BenchResult]] = {}

    print(f"\n{'='*70}")
    print(f"MTP Speed Benchmark: {model_name}")
    print(f"{'='*70}")

    for cond_name, extra_args in conditions:
        print(f"\n  [{cond_name}] Starting server...")
        proc = start_server(model_path, extra_args)
        try:
            results = []
            for run_id in range(n_runs):
                data = run_completion(prompt, n_predict=60)
                r = parse_result(data, cond_name, run_id)
                results.append(r)
                print(f"    Run {run_id}: {r.predicted_per_second:.1f} tok/s, "
                      f"draft={r.draft_n}/{r.draft_n_accepted} "
                      f"({r.acceptance_rate:.1%}), "
                      f"prompt={r.prompt_n}tok/{r.prompt_ms:.0f}ms, "
                      f"cache_hit={r.cache_hit}")
            all_results[cond_name] = results
        finally:
            stop_server(proc)
            time.sleep(3)

    print(f"\n{'='*70}")
    print(f"SUMMARY: {model_name}")
    print(f"{'='*70}")
    print(f"{'Condition':<12} {'tok/s':>8} {'draft_n':>8} {'accepted':>9} {'AR':>7} {'prompt_ms':>10} {'speedup':>8}")
    print("-" * 70)

    baseline_tps = None
    for cond_name, results in all_results.items():
        valid = [r for r in results if not r.cache_hit and r.predicted_per_second > 0]
        if not valid:
            valid = results
        avg_tps = sum(r.predicted_per_second for r in valid) / len(valid)
        avg_draft_n = sum(r.draft_n for r in valid) / len(valid)
        avg_accepted = sum(r.draft_n_accepted for r in valid) / len(valid)
        avg_ar = avg_accepted / avg_draft_n if avg_draft_n > 0 else 0
        avg_prompt_ms = sum(r.prompt_ms for r in valid) / len(valid)

        if baseline_tps is None:
            baseline_tps = avg_tps
            speedup = 1.0
        else:
            speedup = avg_tps / baseline_tps if baseline_tps > 0 else 0

        print(f"{cond_name:<12} {avg_tps:>8.1f} {avg_draft_n:>8.1f} {avg_accepted:>9.1f} "
              f"{avg_ar:>6.1%} {avg_prompt_ms:>9.0f}ms {speedup:>7.2f}x")

    return all_results


# ── Experiment 2: SIG + MTP Multi-turn ────────────────────────────

def run_sig_mtp_experiment(model_path: str, model_name: str, n_runs: int = 3):
    """Test SIG (prompt cache reuse) + MTP across multi-turn conversation.

    This simulates the SIG pipeline:
      Turn 1: Full prefill of system prompt + user query → generate
      Turn 2: Prefix cached (system prompt) → only new tokens evaluated → generate
      Turn 3: More prefix cached → even fewer new tokens → generate

    llama-server's prompt cache automatically handles this.
    """
    system_prompt = "You are a helpful cooking assistant. You provide detailed recipes with ingredients and steps."

    turns = [
        ("User: What recipes can I make with chicken?\nAssistant:", "chicken recipes"),
        ("User: How about something with pasta instead?\nAssistant:", "pasta recipes"),
        ("User: Can you suggest a vegetarian option?\nAssistant:", "vegetarian option"),
    ]

    all_results: Dict[str, List[List[SIGTurnResult]]] = {}

    for mtp_mode, extra_args in [
        ("no_mtp", []),
        ("mtp_n2", ["--spec-type", "draft-mtp", "--spec-draft-n-max", "2"]),
    ]:
        print(f"\n{'='*70}")
        print(f"SIG + MTP Multi-turn: {model_name} [{mtp_mode}]")
        print(f"{'='*70}")

        run_results = []
        for run_id in range(n_runs):
            proc = start_server(model_path, extra_args)
            try:
                turn_results = []
                full_prompt = system_prompt + "\n\n"

                for turn_id, (user_msg, desc) in enumerate(turns):
                    full_prompt += user_msg
                    data = run_completion(full_prompt, n_predict=60)
                    timings = data.get("timings", {})

                    result = SIGTurnResult(
                        turn_id=turn_id,
                        prompt_tokens=timings.get("prompt_n", 0),
                        prompt_cached_tokens=timings.get("prompt_n", 0) - data.get("tokens_evaluated", 0),
                        prompt_eval_tokens=data.get("tokens_evaluated", 0),
                        prompt_ms=timings.get("prompt_ms", 0),
                        gen_tokens=data.get("tokens_predicted", 0),
                        gen_ms=timings.get("predicted_ms", 0),
                        gen_tok_per_s=timings.get("predicted_per_second", 0),
                        draft_n=timings.get("draft_n", 0),
                        draft_n_accepted=timings.get("draft_n_accepted", 0),
                        total_context_tokens=data.get("tokens_cached", 0),
                    )
                    turn_results.append(result)

                    gen_text = data.get("content", "")[:80].replace("\n", " ")
                    print(f"    Run {run_id} Turn {turn_id}: "
                          f"prompt={result.prompt_eval_tokens}/{result.prompt_tokens}tok "
                          f"({result.prompt_ms:.0f}ms), "
                          f"gen={result.gen_tok_per_s:.1f}tok/s, "
                          f"draft={result.draft_n}/{result.draft_n_accepted}, "
                          f"text={gen_text}...")

                    full_prompt += data.get("content", "") + "\n\n"

                run_results.append(turn_results)
            finally:
                stop_server(proc)
                time.sleep(3)

        all_results[mtp_mode] = run_results

    print(f"\n{'='*70}")
    print(f"SIG + MTP SUMMARY: {model_name}")
    print(f"{'='*70}")
    print(f"{'Mode':<10} {'Turn':>5} {'P(eval)':>8} {'P(total)':>9} {'P(ms)':>7} "
          f"{'G(tok/s)':>9} {'draft_n':>8} {'draft_acc':>9} {'AR':>6}")
    print("-" * 80)

    for mtp_mode, run_results in all_results.items():
        for turn_id in range(len(turns)):
            turn_data = [run[turn_id] for run in run_results if turn_id < len(run)]
            if not turn_data:
                continue
            avg_eval = sum(t.prompt_eval_tokens for t in turn_data) / len(turn_data)
            avg_total = sum(t.prompt_tokens for t in turn_data) / len(turn_data)
            avg_pms = sum(t.prompt_ms for t in turn_data) / len(turn_data)
            avg_gtps = sum(t.gen_tok_per_s for t in turn_data) / len(turn_data)
            avg_dn = sum(t.draft_n for t in turn_data) / len(turn_data)
            avg_da = sum(t.draft_n_accepted for t in turn_data) / len(turn_data)
            avg_ar = avg_da / avg_dn if avg_dn > 0 else 0
            print(f"{mtp_mode:<10} {turn_id:>5} {avg_eval:>8.1f} {avg_total:>9.1f} {avg_pms:>6.0f}ms "
                  f"{avg_gtps:>9.1f} {avg_dn:>8.1f} {avg_da:>9.1f} {avg_ar:>5.1%}")

    return all_results


# ── Experiment 3: n-gram SpecDec vs Native MTP ────────────────────

def run_ngram_vs_mtp(model_path: str, model_name: str):
    """Compare paper's n-gram SpecDec (llama-cpp-python) with native MTP (llama-server)."""
    prompt = "You are a kitchen assistant.\n\nUser: What recipes can I make with chicken?\nAssistant:"

    print(f"\n{'='*70}")
    print(f"n-gram SpecDec vs Native MTP: {model_name}")
    print(f"{'='*70}")

    # Part A: n-gram SpecDec via llama-cpp-python (use standard GGUF)
    std_model = str(MODEL_DIR / "Qwen3.5-4B-Q4_K_M.gguf")
    if not Path(std_model).exists():
        std_model = model_path
    print("\n  [n-gram SpecDec] Loading model via llama-cpp-python...")
    try:
        from core.llamacpp_specdec import ManualSpecDecCompiler
        compiler = ManualSpecDecCompiler(
            model_path=std_model, n_ctx=4096, n_gpu_layers=99,
            ngram_size=3, num_pred=3,
        )
        tokens = compiler.tokenize(prompt)
        compiler.reset_cache()
        compiler.eval(tokens)

        t0 = time.time()
        text, gen_ids, stats = compiler.generate_with_specdec(
            stop_str="\nUser:", max_new=60, num_pred=3)
        ngram_time = time.time() - t0
        ngram_tps = len(gen_ids) / ngram_time if ngram_time > 0 else 0

        print(f"    n-gram: {len(gen_ids)} tokens in {ngram_time:.2f}s ({ngram_tps:.1f} tok/s)")
        print(f"    Draft proposed: {stats['draft_proposed']}, accepted: {stats['draft_accepted']}")
        if stats['draft_proposed'] > 0:
            print(f"    Acceptance rate: {100*stats['draft_accepted']/stats['draft_proposed']:.1f}%")
        del compiler
    except Exception as e:
        print(f"    n-gram SpecDec failed: {e}")
        ngram_tps = 0
        ngram_ar = 0

    # Part B: Baseline via llama-cpp-python (no SpecDec, use standard GGUF)
    print("\n  [Baseline] Loading model via llama-cpp-python...")
    try:
        from llama_cpp import Llama
        llm = Llama(model_path=std_model, n_ctx=4096, n_gpu_layers=99, verbose=False)
        tokens = list(llm.tokenize(prompt.encode()))
        llm.eval(tokens)
        gen_ids = []
        t0 = time.time()
        for _ in range(60):
            tok = llm.sample(temp=0.0)
            gen_ids.append(tok)
            llm.eval([tok])
            text = llm.detokenize(gen_ids).decode("utf-8", errors="replace")
            if "\nUser:" in text or "Assistant:" in text[len(text)//2:]:
                break
        baseline_time = time.time() - t0
        baseline_tps = len(gen_ids) / baseline_time if baseline_time > 0 else 0
        print(f"    Baseline: {len(gen_ids)} tokens in {baseline_time:.2f}s ({baseline_tps:.1f} tok/s)")
        del llm
    except Exception as e:
        print(f"    Baseline failed: {e}")
        baseline_tps = 0

    # Part C: Native MTP via llama-server
    print("\n  [Native MTP] Starting llama-server...")
    proc = start_server(model_path, ["--spec-type", "draft-mtp", "--spec-draft-n-max", "2"])
    try:
        mtp_results = []
        for run_id in range(3):
            data = run_completion(prompt, n_predict=60)
            r = parse_result(data, "mtp_n2", run_id)
            mtp_results.append(r)
            print(f"    Run {run_id}: {r.predicted_per_second:.1f} tok/s, "
                  f"draft={r.draft_n}/{r.draft_n_accepted} ({r.acceptance_rate:.1%})")
    finally:
        stop_server(proc)

    avg_mtp_tps = sum(r.predicted_per_second for r in mtp_results) / len(mtp_results)
    avg_mtp_ar = sum(r.acceptance_rate for r in mtp_results) / len(mtp_results)

    print(f"\n{'='*70}")
    print(f"COMPARISON: {model_name}")
    print(f"{'='*70}")
    print(f"  Baseline (llama-cpp-python):  {baseline_tps:.1f} tok/s")
    print(f"  n-gram SpecDec (sequential):   {ngram_tps:.1f} tok/s  ({ngram_tps/baseline_tps:.2f}x)" if baseline_tps > 0 else "")
    print(f"  Native MTP (parallel verify):  {avg_mtp_tps:.1f} tok/s  ({avg_mtp_tps/baseline_tps:.2f}x)" if baseline_tps > 0 else "")
    print(f"  MTP acceptance rate:           {avg_mtp_ar:.1%}")

    return {
        "baseline_tps": baseline_tps,
        "ngram_tps": ngram_tps,
        "mtp_tps": avg_mtp_tps,
        "mtp_acceptance_rate": avg_mtp_ar,
    }


# ── Main ──────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="MTP Experiment v2")
    parser.add_argument("--exp", choices=["mtp", "sig", "compare", "all"], default="all")
    parser.add_argument("--model", choices=["4B", "0.8B", "both"], default="4B")
    parser.add_argument("--n-runs", type=int, default=5)
    args = parser.parse_args()

    models = []
    if args.model in ("4B", "both"):
        models.append(("4B", str(MODEL_DIR / "mtp" / "Qwen3.5-4B-Q4_K_M.gguf")))
    if args.model in ("0.8B", "both"):
        models.append(("0.8B", str(MODEL_DIR / "mtp" / "Qwen3.5-0.8B-Q4_K_M.gguf")))

    all_data = {}

    for model_name, model_path in models:
        if not Path(model_path).exists():
            print(f"Model not found: {model_path}, skipping")
            continue

        if args.exp in ("mtp", "all"):
            results = run_mtp_benchmark(model_path, model_name, args.n_runs)
            all_data[f"{model_name}_mtp"] = {k: [asdict(r) for r in v] for k, v in results.items()}

        if args.exp in ("sig", "all"):
            results = run_sig_mtp_experiment(model_path, model_name, n_runs=3)
            all_data[f"{model_name}_sig"] = {
                k: [[asdict(t) for t in run] for run in v]
                for k, v in results.items()
            }

        if args.exp in ("compare", "all"):
            results = run_ngram_vs_mtp(model_path, model_name)
            all_data[f"{model_name}_compare"] = results

    out_path = Path("data/exp_mtp_v2_results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_data, f, indent=2, ensure_ascii=False, default=str)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
