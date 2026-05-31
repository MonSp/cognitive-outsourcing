"""Experiment: Native MTP Speculative Decoding with SIG on Qwen3.5

This script tests whether llama.cpp's native MTP support (--spec-type draft-mtp)
can bypass the hybrid-attention barrier documented in Paper 5.

Background:
- Paper 5 found that Qwen3.5's hybrid attention (full_attention_interval=4)
  causes three independent failures blocking all parallel verification paths
  in llama-cpp-python's generate()+drafter path.
- However, llama.cpp has since merged PR #22673 which adds native MTP support
  via --spec-type draft-mtp, which may handle hybrid attention correctly
  at the C++ level.

Two experimental paths:
  Path A: llama-server with MTP + OpenAI-compatible API (recommended)
  Path B: Upgraded llama-cpp-python with MTP support (if available)

Prerequisites:
  1. Download MTP-enabled GGUF models:
     - huggingface-cli download unsloth/Qwen3.5-4B-GGUF
       --include "Qwen3.5-4B-UD-Q4_K_XL.gguf" --local-dir ./models
     Or check for MTP-specific GGUF:
     - huggingface-cli download unsloth/Qwen3.5-4B-MTP-GGUF ...
  2. Compile llama.cpp with MTP support (B9180+):
     git clone https://github.com/ggml-org/llama.cpp.git
     cd llama.cpp
     cmake -B build -DGGML_CUDA=ON
     cmake --build build --config Release --target llama-server llama-cli
  3. Verify MTP support:
     ./build/bin/llama-server --help | grep -A1 spec-type
     (should show "draft-mtp" option)
"""

import json
import time
import subprocess
import os
import sys
import signal
import threading
import urllib.request
import urllib.error
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict


# ── Configuration ──────────────────────────────────────────────────

MODELS_DIR = Path("models")
LLAMA_CPP_BIN_DIR = Path("llama-cpp-bin")

MODEL_CONFIGS = {
    "4B": {
        "standard_gguf": MODELS_DIR / "Qwen3.5-4B-Q4_K_M.gguf",
        "mtp_gguf": MODELS_DIR / "mtp" / "Qwen3.5-4B-Q4_K_M.gguf",
    },
    "0.8B": {
        "standard_gguf": MODELS_DIR / "Qwen3.5-0.8B-Q4_K_M.gguf",
        "mtp_gguf": MODELS_DIR / "mtp" / "Qwen3.5-0.8B-Q4_K_M.gguf",
    },
}

LLAMA_SERVER_BIN = LLAMA_CPP_BIN_DIR / "llama-server.exe"
LLAMA_CLI_BIN = LLAMA_CPP_BIN_DIR / "llama-cli.exe"

N_CTX = 16384
N_GPU_LAYERS = 99
N_THREADS = 4
TEMPERATURE = 0.0
MAX_NEW_TOKENS = 60
SPEC_DRAFT_N_MAX = 2
PORT = 8081


# ── Data Classes ───────────────────────────────────────────────────

@dataclass
class MTPBenchmarkResult:
    model_size: str
    mode: str
    wall_clock_s: float
    gen_tokens: int
    tok_per_s: float
    acceptance_rate: float
    draft_proposed: int
    draft_accepted: int
    spec_type: str
    spec_draft_n_max: int
    error: str = ""


# ── Path A: llama-server with MTP ──────────────────────────────────

class LlamaServerMTP:
    """Manages a llama-server instance with MTP speculative decoding.

    Uses llama.cpp's native --spec-type draft-mtp support, which handles
    hybrid attention KV cache management at the C++ level, potentially
    bypassing the kv_cache_seq_rm failures in the Python API.
    """

    def __init__(
        self,
        model_path: str,
        spec_type: str = "draft-mtp",
        spec_draft_n_max: int = 2,
        n_ctx: int = N_CTX,
        n_gpu_layers: int = N_GPU_LAYERS,
        port: int = PORT,
        extra_args: Optional[List[str]] = None,
    ):
        self.model_path = model_path
        self.spec_type = spec_type
        self.spec_draft_n_max = spec_draft_n_max
        self.n_ctx = n_ctx
        self.n_gpu_layers = n_gpu_layers
        self.port = port
        self.extra_args = extra_args or []
        self.process = None

    def _build_cmd(self) -> List[str]:
        cmd = [
            str(LLAMA_SERVER_BIN),
            "-m", str(self.model_path),
            "--host", "127.0.0.1",
            "--port", str(self.port),
            "-c", str(self.n_ctx),
            "-ngl", str(self.n_gpu_layers),
            "--spec-type", self.spec_type,
            "--spec-draft-n-max", str(self.spec_draft_n_max),
            "--temp", str(TEMPERATURE),
            "-np", "1",
            "--parallel", "1",
        ]
        cmd.extend(self.extra_args)
        return cmd

    def start(self, timeout: float = 120.0) -> bool:
        if not LLAMA_SERVER_BIN.exists():
            raise FileNotFoundError(
                f"llama-server not found at {LLAMA_SERVER_BIN}. "
                f"Please compile llama.cpp with MTP support:\n"
                f"  cd {LLAMA_CPP_DIR}\n"
                f"  cmake -B build -DGGML_CUDA=ON\n"
                f"  cmake --build build --config Release --target llama-server"
            )

        cmd = self._build_cmd()
        print(f"Starting llama-server: {' '.join(cmd)}")

        self.process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0,
        )

        t0 = time.time()
        while time.time() - t0 < timeout:
            try:
                req = urllib.request.Request(f"http://127.0.0.1:{self.port}/health")
                with urllib.request.urlopen(req, timeout=2) as resp:
                    if resp.status == 200:
                        print(f"Server ready in {time.time()-t0:.1f}s")
                        return True
            except (urllib.error.URLError, ConnectionError, OSError):
                pass
            time.sleep(2)

        print("Server failed to start within timeout")
        self.stop()
        return False

    def stop(self):
        if self.process:
            if sys.platform == "win32":
                self.process.send_signal(signal.CTRL_BREAK_EVENT)
            else:
                self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
            self.process = None

    def generate(
        self,
        prompt: str,
        max_tokens: int = MAX_NEW_TOKENS,
        temperature: float = TEMPERATURE,
    ) -> Dict:
        payload = json.dumps({
            "prompt": prompt,
            "n_predict": max_tokens,
            "temperature": temperature,
            "stop": ["\nUser:", "Assistant:", "assistant:"],
        }).encode()

        req = urllib.request.Request(
            f"http://127.0.0.1:{self.port}/completion",
            data=payload,
            headers={"Content-Type": "application/json"},
        )

        t0 = time.time()
        with urllib.request.urlopen(req, timeout=300) as resp:
            result = json.loads(resp.read().decode())
        wall_clock = time.time() - t0

        return {
            "text": result.get("content", ""),
            "tokens_predicted": result.get("tokens_predicted", 0),
            "tokens_evaluated": result.get("tokens_evaluated", 0),
            "wall_clock_s": wall_clock,
            "tok_per_s": result.get("tokens_predicted", 0) / max(wall_clock, 0.001),
            "timings": result.get("timings", {}),
            "draft": result.get("draft", {}),
        }

    def generate_chat(
        self,
        messages: List[Dict],
        max_tokens: int = MAX_NEW_TOKENS,
        temperature: float = TEMPERATURE,
    ) -> Dict:
        payload = json.dumps({
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }).encode()

        req = urllib.request.Request(
            f"http://127.0.0.1:{self.port}/v1/chat/completions",
            data=payload,
            headers={"Content-Type": "application/json"},
        )

        t0 = time.time()
        with urllib.request.urlopen(req, timeout=300) as resp:
            result = json.loads(resp.read().decode())
        wall_clock = time.time() - t0

        choice = result.get("choices", [{}])[0]
        usage = result.get("usage", {})

        return {
            "text": choice.get("message", {}).get("content", ""),
            "tokens_predicted": usage.get("completion_tokens", 0),
            "tokens_evaluated": usage.get("prompt_tokens", 0),
            "wall_clock_s": wall_clock,
            "tok_per_s": usage.get("completion_tokens", 0) / max(wall_clock, 0.001),
        }


# ── Path B: llama-cli with MTP (one-shot) ─────────────────────────

def run_llama_cli_mtp(
    model_path: str,
    prompt: str,
    spec_type: str = "draft-mtp",
    spec_draft_n_max: int = 2,
    n_ctx: int = N_CTX,
    n_gpu_layers: int = N_GPU_LAYERS,
    max_tokens: int = MAX_NEW_TOKENS,
) -> Dict:
    """Run llama-cli with MTP speculative decoding (one-shot mode)."""

    if not LLAMA_CLI_BIN.exists():
        raise FileNotFoundError(f"llama-cli not found at {LLAMA_CLI_BIN}")

    cmd = [
        str(LLAMA_CLI_BIN),
        "-m", str(model_path),
        "-c", str(n_ctx),
        "-ngl", str(n_gpu_layers),
        "-n", str(max_tokens),
        "--spec-type", spec_type,
        "--spec-draft-n-max", str(spec_draft_n_max),
        "--temp", str(TEMPERATURE),
        "-p", prompt,
    ]

    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    wall_clock = time.time() - t0

    output = result.stdout + result.stderr

    draft_info = {}
    for line in output.split("\n"):
        if "draft" in line.lower() and "accept" in line.lower():
            draft_info["raw"] = line.strip()
        if "spec" in line.lower():
            draft_info["spec_raw"] = line.strip()

    return {
        "wall_clock_s": wall_clock,
        "stdout": output[-2000:],
        "draft_info": draft_info,
        "returncode": result.returncode,
    }


# ── SIG + MTP Integration ──────────────────────────────────────────

class SIGMTPExperiment:
    """SIG + native MTP experiment using llama-server.

    This tests the key hypothesis: can llama.cpp's native MTP support
    bypass the hybrid-attention barrier that blocked parallel verification
    in the Python API?

    Experiment conditions:
    1. AppLoop (baseline): Full context re-encoding each turn
    2. SIG: KV-cache injection across turns
    3. AppLoop + MTP: Full re-encoding + native MTP speculative decoding
    4. SIG + MTP: KV-cache injection + native MTP speculative decoding
    """

    def __init__(self, model_size: str = "4B", spec_draft_n_max: int = 2):
        self.model_size = model_size
        self.spec_draft_n_max = spec_draft_n_max
        self.results: List[MTPBenchmarkResult] = []

    def _find_model(self, need_mtp: bool = False) -> Path:
        config = MODEL_CONFIGS[self.model_size]
        if need_mtp:
            path = config["mtp_gguf"]
            if path.exists():
                return path
            print(f"WARNING: MTP GGUF not found at {path}")
            print("Attempting to use standard GGUF (MTP heads may be embedded)...")
        path = config["standard_gguf"]
        if not path.exists():
            raise FileNotFoundError(f"Model not found: {path}")
        return path

    def run_baseline_no_mtp(
        self,
        model_path: Optional[str] = None,
        prompt: str = "You are a kitchen assistant.\n\nUser: What recipes can I make with chicken?\nAssistant:",
        n_runs: int = 5,
    ) -> List[MTPBenchmarkResult]:
        """Run baseline without MTP (standard generation)."""
        if model_path is None:
            model_path = str(self._find_model(need_mtp=False))

        results = []
        server = LlamaServerMTP(
            model_path=model_path,
            spec_type="none",
            n_ctx=N_CTX,
            n_gpu_layers=N_GPU_LAYERS,
        )

        try:
            if not server.start():
                return [MTPBenchmarkResult(
                    model_size=self.model_size, mode="baseline_no_mtp",
                    wall_clock_s=0, gen_tokens=0, tok_per_s=0,
                    acceptance_rate=0, draft_proposed=0, draft_accepted=0,
                    spec_type="none", spec_draft_n_max=0,
                    error="Server failed to start",
                )]

            for run_id in range(n_runs):
                res = server.generate(prompt, max_tokens=MAX_NEW_TOKENS)
                results.append(MTPBenchmarkResult(
                    model_size=self.model_size,
                    mode="baseline_no_mtp",
                    wall_clock_s=res["wall_clock_s"],
                    gen_tokens=res["tokens_predicted"],
                    tok_per_s=res["tok_per_s"],
                    acceptance_rate=0,
                    draft_proposed=0,
                    draft_accepted=0,
                    spec_type="none",
                    spec_draft_n_max=0,
                ))
                print(f"  Run {run_id}: {res['tokens_predicted']} tokens, "
                      f"{res['wall_clock_s']:.2f}s, {res['tok_per_s']:.1f} tok/s")
        finally:
            server.stop()

        return results

    def run_mtp(
        self,
        model_path: Optional[str] = None,
        prompt: str = "You are a kitchen assistant.\n\nUser: What recipes can I make with chicken?\nAssistant:",
        n_runs: int = 5,
        spec_type: str = "draft-mtp",
        spec_draft_n_max: Optional[int] = None,
    ) -> List[MTPBenchmarkResult]:
        """Run with native MTP speculative decoding."""
        if model_path is None:
            model_path = str(self._find_model(need_mtp=True))
        if spec_draft_n_max is None:
            spec_draft_n_max = self.spec_draft_n_max

        results = []
        server = LlamaServerMTP(
            model_path=model_path,
            spec_type=spec_type,
            spec_draft_n_max=spec_draft_n_max,
            n_ctx=N_CTX,
            n_gpu_layers=N_GPU_LAYERS,
        )

        try:
            if not server.start():
                return [MTPBenchmarkResult(
                    model_size=self.model_size, mode=f"mtp_{spec_type}",
                    wall_clock_s=0, gen_tokens=0, tok_per_s=0,
                    acceptance_rate=0, draft_proposed=0, draft_accepted=0,
                    spec_type=spec_type, spec_draft_n_max=spec_draft_n_max,
                    error="Server failed to start",
                )]

            for run_id in range(n_runs):
                res = server.generate(prompt, max_tokens=MAX_NEW_TOKENS)
                draft = res.get("draft", {})
                ar = draft.get("acceptance_rate", 0)
                if ar == 0 and draft.get("accepted", 0) > 0:
                    ar = draft["accepted"] / max(draft.get("proposed", 1), 1)

                results.append(MTPBenchmarkResult(
                    model_size=self.model_size,
                    mode=f"mtp_{spec_type}",
                    wall_clock_s=res["wall_clock_s"],
                    gen_tokens=res["tokens_predicted"],
                    tok_per_s=res["tok_per_s"],
                    acceptance_rate=ar,
                    draft_proposed=draft.get("proposed", 0),
                    draft_accepted=draft.get("accepted", 0),
                    spec_type=spec_type,
                    spec_draft_n_max=spec_draft_n_max,
                ))
                print(f"  Run {run_id}: {res['tokens_predicted']} tokens, "
                      f"{res['wall_clock_s']:.2f}s, {res['tok_per_s']:.1f} tok/s, "
                      f"AR={ar:.1%}")
        finally:
            server.stop()

        return results

    def run_full_experiment(
        self,
        prompt: str = "You are a kitchen assistant.\n\nUser: What recipes can I make with chicken?\nAssistant:",
        n_runs: int = 5,
    ) -> Dict:
        """Run the full 4-condition experiment matching Paper 5's EXP-1 design."""
        all_results = {}

        print("=" * 60)
        print("EXP-MTP: Native MTP Speculative Decoding on Qwen3.5")
        print("=" * 60)

        # Condition 1: Baseline (no MTP)
        print("\n[1/4] Baseline (no MTP)...")
        all_results["baseline_no_mtp"] = self.run_baseline_no_mtp(
            prompt=prompt, n_runs=n_runs)

        # Condition 2: MTP with draft-n-max=1
        print("\n[2/4] MTP (draft-n-max=1)...")
        all_results["mtp_n1"] = self.run_mtp(
            prompt=prompt, n_runs=n_runs, spec_draft_n_max=1)

        # Condition 3: MTP with draft-n-max=2
        print("\n[3/4] MTP (draft-n-max=2)...")
        all_results["mtp_n2"] = self.run_mtp(
            prompt=prompt, n_runs=n_runs, spec_draft_n_max=2)

        # Condition 4: MTP with draft-n-max=3
        print("\n[4/4] MTP (draft-n-max=3)...")
        all_results["mtp_n3"] = self.run_mtp(
            prompt=prompt, n_runs=n_runs, spec_draft_n_max=3)

        # Summary
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        for cond, results in all_results.items():
            if not results or results[0].error:
                print(f"  {cond}: ERROR - {results[0].error if results else 'no results'}")
                continue
            avg_time = sum(r.wall_clock_s for r in results) / len(results)
            avg_tok_s = sum(r.tok_per_s for r in results) / len(results)
            avg_ar = sum(r.acceptance_rate for r in results) / len(results)
            print(f"  {cond}: {avg_time:.2f}s, {avg_tok_s:.1f} tok/s, AR={avg_ar:.1%}")

        return all_results


# ── Setup Helpers ──────────────────────────────────────────────────

def check_prerequisites() -> Dict[str, bool]:
    """Check if all prerequisites are met for the MTP experiment."""
    checks = {
        "llama_server": LLAMA_SERVER_BIN.exists(),
        "llama_cli": LLAMA_CLI_BIN.exists(),
        "model_4b": MODEL_CONFIGS["4B"]["standard_gguf"].exists(),
        "model_08b": MODEL_CONFIGS["0.8B"]["standard_gguf"].exists(),
        "model_4b_mtp": MODEL_CONFIGS["4B"]["mtp_gguf"].exists(),
        "model_08b_mtp": MODEL_CONFIGS["0.8B"]["mtp_gguf"].exists(),
    }
    return checks


def print_setup_instructions():
    """Print step-by-step setup instructions."""
    print("""
╔══════════════════════════════════════════════════════════════════╗
║  MTP Experiment Setup Instructions                              ║
╚══════════════════════════════════════════════════════════════════╝

Step 1: Compile llama.cpp with MTP support (requires B9180+)
─────────────────────────────────────────────────────────────
  git clone https://github.com/ggml-org/llama.cpp.git ../llama.cpp
  cd ../llama.cpp
  cmake -B build -DGGML_CUDA=ON
  cmake --build build --config Release --target llama-server llama-cli

  Verify MTP support:
    ./build/bin/llama-server --help | findstr "spec-type"
    (should show "draft-mtp" option)

Step 2: Download MTP-enabled GGUF models
──────────────────────────────────────────
  Option A: Download from unsloth (if available for Qwen3.5-4B/0.8B)
    pip install huggingface-hub
    huggingface-cli download unsloth/Qwen3.5-4B-GGUF ^
      --include "Qwen3.5-4B-UD-Q4_K_XL.gguf" ^
      --local-dir ./models

  Option B: Convert from HuggingFace with MTP heads
    python convert_hf_to_gguf.py ^
      --outfile models/Qwen3.5-4B-MTP-Q4_K_M.gguf ^
      --outtype q4_k_m ^
      Qwen/Qwen3.5-4B

  NOTE: The standard Qwen3.5-4B-Q4_K_M.gguf may already contain MTP
  head weights. Check with:
    python -c "from llama_cpp import Llama; llm = Llama(model_path='models/Qwen3.5-4B-Q4_K_M.gguf', vocab_only=True); print([k for k in llm._model.metadata().keys() if 'mtp' in k.lower()])"

Step 3: Run the experiment
───────────────────────────
  python exp_mtp_native.py --check          # Check prerequisites
  python exp_mtp_native.py --quick          # Quick test (1 run)
  python exp_mtp_native.py --full           # Full experiment (5 runs)
  python exp_mtp_native.py --cli-test       # Test via llama-cli
""")


def download_mtp_models():
    """Download MTP-enabled GGUF models from HuggingFace."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("Please install huggingface-hub: pip install huggingface-hub")
        return

    models_to_try = [
        ("unsloth/Qwen3.5-4B-GGUF", "Qwen3.5-4B-UD-Q4_K_XL.gguf"),
        ("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-UD-Q4_K_XL.gguf"),
    ]

    for repo_id, filename in models_to_try:
        target = MODELS_DIR / filename
        if target.exists():
            print(f"  Already exists: {target}")
            continue
        print(f"  Downloading {filename} from {repo_id}...")
        try:
            path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                local_dir=str(MODELS_DIR),
            )
            print(f"  Downloaded to: {path}")
        except Exception as e:
            print(f"  Failed: {e}")
            print(f"  Try manually: huggingface-cli download {repo_id} --include \"{filename}\" --local-dir ./models")


# ── CLI Interface ──────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Native MTP SpecDec Experiment")
    parser.add_argument("--check", action="store_true", help="Check prerequisites")
    parser.add_argument("--setup", action="store_true", help="Print setup instructions")
    parser.add_argument("--download", action="store_true", help="Download MTP models")
    parser.add_argument("--quick", action="store_true", help="Quick test (1 run)")
    parser.add_argument("--full", action="store_true", help="Full experiment (5 runs)")
    parser.add_argument("--cli-test", action="store_true", help="Test via llama-cli")
    parser.add_argument("--model", choices=["4B", "0.8B"], default="4B")
    parser.add_argument("--draft-n-max", type=int, default=2)
    parser.add_argument("--port", type=int, default=PORT)
    args = parser.parse_args()

    if args.setup or (not any([args.check, args.quick, args.full, args.cli_test, args.download])):
        print_setup_instructions()
        return

    if args.check:
        checks = check_prerequisites()
        print("Prerequisites Check:")
        for name, ok in checks.items():
            status = "✅" if ok else "❌"
            print(f"  {status} {name}")
        if not checks["llama_server"]:
            print("\n⚠️  llama-server not found. Please compile llama.cpp with MTP support.")
        if not checks["model_4b_mtp"]:
            print("\n⚠️  MTP GGUF model not found. Run --download or convert manually.")
        return

    if args.download:
        download_mtp_models()
        return

    global PORT
    PORT = args.port

    exp = SIGMTPExperiment(model_size=args.model, spec_draft_n_max=args.draft_n_max)

    if args.cli_test:
        model_path = str(exp._find_model(need_mtp=True))
        prompt = "You are a kitchen assistant.\n\nUser: What recipes can I make with chicken?\nAssistant:"
        print(f"Testing llama-cli with MTP on {args.model}...")
        result = run_llama_cli_mtp(
            model_path=model_path,
            prompt=prompt,
            spec_type="draft-mtp",
            spec_draft_n_max=args.draft_n_max,
        )
        print(f"Result: {result['wall_clock_s']:.2f}s")
        print(f"Draft info: {result['draft_info']}")
        if result["returncode"] != 0:
            print(f"Error (rc={result['returncode']}): {result['stdout'][-500:]}")
        return

    n_runs = 1 if args.quick else 5
    results = exp.run_full_experiment(n_runs=n_runs)

    output_path = Path("data") / f"exp_mtp_{args.model}_n{n_runs}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            {cond: [asdict(r) for r in res] for cond, res in results.items()},
            f, indent=2, ensure_ascii=False,
        )
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
