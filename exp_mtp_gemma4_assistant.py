"""Experiment: Gemma-4-E2B-it + Gemma-4-E2B-it-assistant (official MTP draft)

This script validates orthogonality between the SIG (Stagnant-Information-
Grouping) cache-reuse strategy and the official Gemma-4 MTP (Multi-Token
Prediction) draft accelerator. The pairing is:

    target model : Gemma-4-E2B-it  (4.6B params, Q4_K_M)
    MTP draft    : Gemma-4-E2B-it-assistant  (78M, Q4_K_M, official MTP head)

The llama-server is invoked with --model-draft and --spec-type draft-mtp,
which is the upstream llama.cpp path for native MTP (Gemma-4 ships a
designated MTP head that llama-server wires into the speculative sampler
automatically). We sweep four draft-n-max values (none, 1, 2, 3) over
four prompt token counts (30, 500, 1000, 2000) at n=10 runs per cell.

The script follows the same patterns as exp_mtp_full_v2.py:
  - llama-server subprocess lifecycle (start -> healthcheck -> test -> stop)
  - Streaming-based tok/s measurement
  - Placeholder JSON emission if the binary is missing
  - AR normalization across server versions

SIG simulation:
  We approximate the SIG "client-side token passthrough" mode without
  touching the C++ KV cache (no `kv_cache_seq_rm` API is exposed via HTTP
  on llama-server). Each chat turn is sent as a sequence of pre-tokenized
  token arrays via the `prompt_tokens` field, which lets us measure
  *only the new tokens* being decoded while keeping the prefix as
  pre-computed ids. The JSON output is annotated with
  `sig_simulation_mode: "client_side_token_passthrough"` so downstream
  consumers can tell the SIG vs vanilla gap was measured at the prompt-
  encoding layer, not at the actual KV-cache slot level.

SpecDec compatibility probe:
  Three prompt sizes (30, 2000, 8000) are pushed through the MTP pipeline
  to verify that long prompts do not trigger OOM/OOG and that MTP still
  proposes draft tokens. We treat any successful generation as a pass and
  record first-token latency and AR per length bucket.
"""

import argparse
import json
import math
import os
import shutil
import signal
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ── Configuration ──────────────────────────────────────────────────

MODELS_DIR = Path("models")
LLAMA_CPP_BIN_DIR = Path("llama-cpp-bin")
LLAMA_SERVER_BIN = LLAMA_CPP_BIN_DIR / "llama-server.exe"
LLAMA_SERVER_BIN_UNIX = LLAMA_CPP_BIN_DIR / "llama-server"

DEFAULT_TARGET = str(MODELS_DIR / "gemma-4-E2B-it-Q4_K_M.gguf")
DEFAULT_DRAFT = str(MODELS_DIR / "gemma-4-E2B-it-assistant.Q4_K_M.gguf")

N_CTX = 16384
N_GPU_LAYERS = 99
N_THREADS = 4
TEMPERATURE = 0.0
MAX_NEW_TOKENS = 64
DEFAULT_PORT = 8083
HEALTHCHECK_TIMEOUT_S = 180.0
HEALTHCHECK_INTERVAL_S = 2.0
DEFAULT_CONTEXT_LENGTHS = (30, 500, 1000, 2000)
DEFAULT_N_RUNS = 10

# SpecDec compatibility probe lengths: short / medium / very long.
# We deliberately do NOT pre-allocate a slot at 8000 to stress the
# KV-cache allocator inside llama-server.
SPECDEC_PROBE_LENGTHS = (30, 2000, 8000)

SIG_SIMULATION_MODE = "client_side_token_passthrough"

# Multi-turn chat script used for end-to-end SIG vs SIG+MTP comparison.
# Each tuple is (user_utterance, max_new_tokens_for_turn).
SIG_CHAT_TURNS: List[Tuple[str, int]] = [
    ("Hi, I need a 3-step plan to write a SIG benchmark report.", 80),
    ("Actually make it 4 steps and emphasize reproducibility.", 80),
    ("Now give a 2-sentence summary I can paste into the abstract.", 60),
    ("Thanks. Finally, list 3 risks I should mitigate.", 60),
]

# Mode specifications: (mode_label, spec_type, draft_n_max, use_draft_arg)
# - no_mtp: no --model-draft, --spec-type none
# - mtp_n1/2/3: --model-draft, --spec-type draft-mtp, --draft-n-max N
FULL_MODE_SPECS: List[Tuple[str, str, int, bool]] = [
    ("no_mtp", "none", 0, False),
    ("mtp_n1", "draft-mtp", 1, True),
    ("mtp_n2", "draft-mtp", 2, True),
    ("mtp_n3", "draft-mtp", 3, True),
]


# ── Data Classes ───────────────────────────────────────────────────

@dataclass
class SingleRunResult:
    """One run for a given (mode, context_length) cell."""
    mode: str
    context_length: int
    run_id: int
    wall_clock_s: float
    gen_tokens: int
    prompt_tokens: int
    tok_per_s: float
    acceptance_rate: float
    draft_proposed: int
    draft_accepted: int
    spec_type: str
    spec_draft_n_max: int
    streaming_measured: bool
    error: str = ""
    skipped: bool = False
    skip_reason: str = ""


@dataclass
class ModeContextSummary:
    """Aggregate statistics over n runs for a given (mode, context_length)."""
    mode: str
    context_length: int
    n_runs: int
    mean_tok_per_s: float
    std_tok_per_s: float
    mean_wall_clock_s: float
    std_wall_clock_s: float
    mean_gen_tokens: float
    mean_acceptance_rate: float
    mean_draft_proposed: float
    mean_draft_accepted: float
    skipped: bool
    skip_reason: str = ""


@dataclass
class ChatTurnResult:
    """One turn of a multi-turn SIG/SIG+MTP comparison."""
    turn_id: int
    prompt_tokens: int
    gen_tokens: int
    wall_clock_s: float
    tok_per_s: float
    cumulative_wall_clock_s: float
    acceptance_rate: float
    draft_proposed: int
    draft_accepted: int


@dataclass
class ChatEndToEndResult:
    """End-to-end result for a multi-turn chat (SIG-only or SIG+MTP)."""
    mode: str
    n_turns: int
    total_wall_clock_s: float
    mean_tok_per_s: float
    sum_gen_tokens: int
    turns: List[ChatTurnResult] = field(default_factory=list)
    skipped: bool = False
    skip_reason: str = ""


@dataclass
class SpecDecProbeResult:
    """One specdec-compatibility probe (mode x prompt_length)."""
    mode: str
    prompt_length: int
    n_runs: int
    pass_count: int
    fail_count: int
    mean_tok_per_s: float
    mean_acceptance_rate: float
    oom_error: bool
    oog_error: bool
    notes: str = ""


# ── llama-server detection ──────────────────────────────────────────

def detect_llama_server() -> Tuple[bool, Optional[Path]]:
    """Return (available, resolved_binary_path)."""
    candidates = [LLAMA_SERVER_BIN, LLAMA_SERVER_BIN_UNIX]
    for cand in candidates:
        if cand.exists() and cand.is_file():
            return True, cand
    path_lookup = shutil.which("llama-server")
    if path_lookup:
        return True, Path(path_lookup)
    return False, None


def detect_model(model_path: str) -> bool:
    return Path(model_path).exists()


# ── AR normalization (same rules as exp_mtp_full_v2.py) ─────────────

def _normalize_ar(value: float) -> float:
    """Normalize acceptance_rate to a fraction in [0, 1].

    llama-server has historically reported this field with inconsistent
    scaling: some versions return a fraction (0.24), others a percentage
    (24.0). We treat any value > 1.5 as a percentage and divide by 100.
    """
    if value is None:
        return 0.0
    try:
        v = float(value)
    except (TypeError, ValueError):
        return 0.0
    if v < 0:
        return 0.0
    if v > 1.5:
        v = v / 100.0
    return min(v, 1.0)


def safe_std(xs: List[float]) -> float:
    if len(xs) < 2:
        return 0.0
    try:
        return statistics.stdev(xs)
    except statistics.StatisticsError:
        return 0.0


# ── LlamaServerMTP class ───────────────────────────────────────────

class LlamaServerMTP:
    """Manages a llama-server subprocess with MTP speculative decoding.

    Supports paired main + draft model startup via --model-draft.

    Lifecycle: __init__ -> start() -> generate/stream -> stop()
    Each instance binds a single port; create a new instance per (mode, ctx).
    """

    def __init__(
        self,
        binary: Path,
        model_path: str,
        draft_model_path: Optional[str] = None,
        spec_type: str = "none",
        spec_draft_n_max: int = 0,
        n_ctx: int = N_CTX,
        n_gpu_layers: int = N_GPU_LAYERS,
        port: int = DEFAULT_PORT,
        extra_args: Optional[List[str]] = None,
        n_threads: int = N_THREADS,
    ):
        self.binary = binary
        self.model_path = model_path
        self.draft_model_path = draft_model_path
        self.spec_type = spec_type
        self.spec_draft_n_max = spec_draft_n_max
        self.n_ctx = n_ctx
        self.n_gpu_layers = n_gpu_layers
        self.port = port
        self.extra_args = list(extra_args or [])
        self.n_threads = n_threads
        self.process: Optional[subprocess.Popen] = None
        self._log_lines: List[str] = []

    def _build_cmd(self) -> List[str]:
        cmd = [
            str(self.binary),
            "-m", str(self.model_path),
            "--host", "127.0.0.1",
            "--port", str(self.port),
            "-c", str(self.n_ctx),
            "-ngl", str(self.n_gpu_layers),
            "-t", str(self.n_threads),
            "--temp", str(TEMPERATURE),
            "-np", "1",
            "--parallel", "1",
        ]
        # Paired main+draft model startup for MTP path.
        if self.draft_model_path:
            cmd += ["--model-draft", str(self.draft_model_path)]
        # Spec-decoding args.
        if self.spec_type and self.spec_type != "none":
            cmd += ["--spec-type", self.spec_type]
            if self.spec_draft_n_max > 0:
                # llama-server exposes this as --spec-draft-n-max
                # (alias: --spec-draft-max). We use the canonical name.
                cmd += ["--spec-draft-n-max", str(self.spec_draft_n_max)]
        cmd.extend(self.extra_args)
        return cmd

    def start(self, timeout: float = HEALTHCHECK_TIMEOUT_S) -> bool:
        cmd = self._build_cmd()
        print(f"  [server] starting: {' '.join(cmd)}")
        try:
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                creationflags=(
                    subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0
                ),
            )
        except FileNotFoundError as e:
            print(f"  [server] FAILED to spawn: {e}")
            self.process = None
            return False

        t0 = time.time()
        while time.time() - t0 < timeout:
            if self.process.poll() is not None:
                try:
                    out, _ = self.process.communicate(timeout=2)
                except Exception:
                    out = b""
                self._log_lines.append(out.decode(errors="replace")[-2000:])
                print(f"  [server] exited prematurely rc={self.process.returncode}")
                self.process = None
                return False
            try:
                req = urllib.request.Request(f"http://127.0.0.1:{self.port}/health")
                with urllib.request.urlopen(req, timeout=2) as resp:
                    if resp.status == 200:
                        print(f"  [server] ready in {time.time()-t0:.1f}s on :{self.port}")
                        return True
            except (urllib.error.URLError, ConnectionError, OSError):
                pass
            time.sleep(HEALTHCHECK_INTERVAL_S)

        print(f"  [server] timeout after {timeout:.0f}s")
        self.stop()
        return False

    def stop(self) -> None:
        if self.process is None:
            return
        proc = self.process
        self.process = None
        try:
            if sys.platform == "win32":
                # Prefer CTRL_BREAK_EVENT; fall back to taskkill /F /T /PID.
                try:
                    proc.send_signal(signal.CTRL_BREAK_EVENT)
                except Exception:
                    try:
                        subprocess.run(
                            ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                            check=False,
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL,
                        )
                    except Exception:
                        proc.terminate()
            else:
                proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                try:
                    proc.wait(timeout=5)
                except Exception:
                    pass
        except Exception as e:
            print(f"  [server] stop error: {e}")

    # ── HTTP API helpers ──────────────────────────────────────────

    def _post(self, path: str, payload: dict, stream: bool = False, timeout: float = 300.0):
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"http://127.0.0.1:{self.port}{path}",
            data=data,
            headers={"Content-Type": "application/json"},
        )
        if stream:
            return urllib.request.urlopen(req, timeout=timeout)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def generate(
        self,
        prompt: str,
        max_tokens: int = MAX_NEW_TOKENS,
        temperature: float = TEMPERATURE,
    ) -> Dict:
        """Non-streaming completion. Returns parsed result + wall-clock."""
        payload = {
            "prompt": prompt,
            "n_predict": max_tokens,
            "temperature": temperature,
            "stream": False,
        }
        t0 = time.time()
        result = self._post("/completion", payload)
        wall_clock = time.time() - t0
        timings = result.get("timings", {}) or {}
        draft = result.get("draft", {}) or {}
        tokens_predicted = int(result.get("tokens_predicted", 0) or 0)
        return {
            "text": result.get("content", "") or "",
            "tokens_predicted": tokens_predicted,
            "tokens_evaluated": int(result.get("tokens_evaluated", 0) or 0),
            "wall_clock_s": wall_clock,
            "tok_per_s": tokens_predicted / max(wall_clock, 1e-3),
            "timings": timings,
            "draft": draft,
            "raw": result,
        }

    def generate_streaming(
        self,
        prompt: str,
        max_tokens: int = MAX_NEW_TOKENS,
        temperature: float = TEMPERATURE,
    ) -> Dict:
        """Streaming completion; measures tok/s by token-arrival cadence."""
        payload = {
            "prompt": prompt,
            "n_predict": max_tokens,
            "temperature": temperature,
            "stream": True,
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"http://127.0.0.1:{self.port}/completion",
            data=data,
            headers={"Content-Type": "application/json"},
        )

        token_count = 0
        last_draft_proposed = 0
        last_draft_accepted = 0
        last_acceptance_rate = 0.0
        last_prompt_tokens = 0
        text_buf: List[str] = []
        t_open: Optional[float] = None
        t_close: Optional[float] = None
        t_first_token: Optional[float] = None

        try:
            with urllib.request.urlopen(req, timeout=600) as resp:
                t_open = time.time()
                for raw_line in resp:
                    if not raw_line:
                        continue
                    line = raw_line.decode("utf-8", errors="replace").rstrip("\r\n")
                    if not line.startswith("data:"):
                        continue
                    payload_str = line[5:].strip()
                    if not payload_str or payload_str == "[DONE]":
                        continue
                    try:
                        obj = json.loads(payload_str)
                    except json.JSONDecodeError:
                        continue
                    delta = obj.get("content", "")
                    if delta:
                        text_buf.append(delta)
                        token_count += 1
                        if t_first_token is None:
                            t_first_token = time.time()
                    if "draft" in obj and isinstance(obj["draft"], dict):
                        d = obj["draft"]
                        if "proposed" in d:
                            last_draft_proposed = int(d.get("proposed", 0) or 0)
                        if "accepted" in d:
                            last_draft_accepted = int(d.get("accepted", 0) or 0)
                        if "acceptance_rate" in d:
                            try:
                                last_acceptance_rate = _normalize_ar(
                                    float(d["acceptance_rate"])
                                )
                            except (TypeError, ValueError):
                                pass
                    if "timings" in obj and isinstance(obj["timings"], dict):
                        tm = obj["timings"]
                        if "draft_n_total" in tm:
                            last_draft_proposed = int(tm.get("draft_n_total", 0) or 0)
                        if "draft_n_accepted" in tm:
                            last_draft_accepted = int(tm.get("draft_n_accepted", 0) or 0)
                        if "draft_accept_rate" in tm:
                            try:
                                last_acceptance_rate = _normalize_ar(
                                    float(tm["draft_accept_rate"])
                                )
                            except (TypeError, ValueError):
                                pass
                    if "tokens_evaluated" in obj:
                        try:
                            last_prompt_tokens = int(obj["tokens_evaluated"])
                        except (TypeError, ValueError):
                            pass
                    if obj.get("stop", False):
                        if "draft" in obj and isinstance(obj["draft"], dict):
                            d = obj["draft"]
                            if "proposed" in d:
                                last_draft_proposed = int(d.get("proposed", 0) or 0)
                            if "accepted" in d:
                                last_draft_accepted = int(d.get("accepted", 0) or 0)
                            if "acceptance_rate" in d:
                                try:
                                    last_acceptance_rate = _normalize_ar(
                                        float(d["acceptance_rate"])
                                    )
                                except (TypeError, ValueError):
                                    pass
                        if "tokens_evaluated" in obj:
                            try:
                                last_prompt_tokens = int(obj["tokens_evaluated"])
                            except (TypeError, ValueError):
                                pass
                        break
                t_close = time.time()
        except (urllib.error.URLError, ConnectionError, OSError) as e:
            return {
                "text": "".join(text_buf),
                "tokens_predicted": token_count,
                "tokens_evaluated": last_prompt_tokens,
                "wall_clock_s": 0.0,
                "tok_per_s": 0.0,
                "acceptance_rate": last_acceptance_rate,
                "draft_proposed": last_draft_proposed,
                "draft_accepted": last_draft_accepted,
                "streaming_measured": True,
                "error": f"stream error: {e}",
            }

        wall_clock = (t_close - t_open) if (t_open and t_close) else 0.0
        first_to_last = (t_close - t_first_token) if (t_first_token and t_close) else wall_clock
        effective = first_to_last if first_to_last > 0 else wall_clock
        tok_per_s = token_count / max(effective, 1e-3)

        return {
            "text": "".join(text_buf),
            "tokens_predicted": token_count,
            "tokens_evaluated": last_prompt_tokens,
            "wall_clock_s": wall_clock,
            "tok_per_s": tok_per_s,
            "acceptance_rate": last_acceptance_rate,
            "draft_proposed": last_draft_proposed,
            "draft_accepted": last_draft_accepted,
            "streaming_measured": True,
        }

    def generate_with_prefix_tokens(
        self,
        prefix_token_ids: List[int],
        max_tokens: int = MAX_NEW_TOKENS,
        temperature: float = TEMPERATURE,
    ) -> Dict:
        """Generate using a pre-tokenized prompt.

        Sends a `prompt_tokens` array (list of int) to llama-server. This is
        the SIG client-side token passthrough path: we let the client encode
        the static prefix once, then re-send only the *delta* (new turn
        text + tool result) as additional token ids.
        """
        payload = {
            "prompt_tokens": list(prefix_token_ids),
            "n_predict": max_tokens,
            "temperature": temperature,
            "stream": False,
        }
        t0 = time.time()
        result = self._post("/completion", payload)
        wall_clock = time.time() - t0
        timings = result.get("timings", {}) or {}
        draft = result.get("draft", {}) or {}
        tokens_predicted = int(result.get("tokens_predicted", 0) or 0)
        return {
            "text": result.get("content", "") or "",
            "tokens_predicted": tokens_predicted,
            "tokens_evaluated": int(result.get("tokens_evaluated", 0) or 0),
            "wall_clock_s": wall_clock,
            "tok_per_s": tokens_predicted / max(wall_clock, 1e-3),
            "timings": timings,
            "draft": draft,
            "raw": result,
        }

    def generate_streaming_with_prefix_tokens(
        self,
        prefix_token_ids: List[int],
        max_tokens: int = MAX_NEW_TOKENS,
        temperature: float = TEMPERATURE,
    ) -> Dict:
        """Streaming variant of generate_with_prefix_tokens."""
        payload = {
            "prompt_tokens": list(prefix_token_ids),
            "n_predict": max_tokens,
            "temperature": temperature,
            "stream": True,
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"http://127.0.0.1:{self.port}/completion",
            data=data,
            headers={"Content-Type": "application/json"},
        )

        token_count = 0
        last_draft_proposed = 0
        last_draft_accepted = 0
        last_acceptance_rate = 0.0
        last_prompt_tokens = 0
        text_buf: List[str] = []
        t_open: Optional[float] = None
        t_close: Optional[float] = None
        t_first_token: Optional[float] = None

        try:
            with urllib.request.urlopen(req, timeout=600) as resp:
                t_open = time.time()
                for raw_line in resp:
                    if not raw_line:
                        continue
                    line = raw_line.decode("utf-8", errors="replace").rstrip("\r\n")
                    if not line.startswith("data:"):
                        continue
                    payload_str = line[5:].strip()
                    if not payload_str or payload_str == "[DONE]":
                        continue
                    try:
                        obj = json.loads(payload_str)
                    except json.JSONDecodeError:
                        continue
                    delta = obj.get("content", "")
                    if delta:
                        text_buf.append(delta)
                        token_count += 1
                        if t_first_token is None:
                            t_first_token = time.time()
                    if "draft" in obj and isinstance(obj["draft"], dict):
                        d = obj["draft"]
                        if "proposed" in d:
                            last_draft_proposed = int(d.get("proposed", 0) or 0)
                        if "accepted" in d:
                            last_draft_accepted = int(d.get("accepted", 0) or 0)
                        if "acceptance_rate" in d:
                            try:
                                last_acceptance_rate = _normalize_ar(
                                    float(d["acceptance_rate"])
                                )
                            except (TypeError, ValueError):
                                pass
                    if "timings" in obj and isinstance(obj["timings"], dict):
                        tm = obj["timings"]
                        if "draft_n_total" in tm:
                            last_draft_proposed = int(tm.get("draft_n_total", 0) or 0)
                        if "draft_n_accepted" in tm:
                            last_draft_accepted = int(tm.get("draft_n_accepted", 0) or 0)
                        if "draft_accept_rate" in tm:
                            try:
                                last_acceptance_rate = _normalize_ar(
                                    float(tm["draft_accept_rate"])
                                )
                            except (TypeError, ValueError):
                                pass
                    if "tokens_evaluated" in obj:
                        try:
                            last_prompt_tokens = int(obj["tokens_evaluated"])
                        except (TypeError, ValueError):
                            pass
                    if obj.get("stop", False):
                        if "draft" in obj and isinstance(obj["draft"], dict):
                            d = obj["draft"]
                            if "proposed" in d:
                                last_draft_proposed = int(d.get("proposed", 0) or 0)
                            if "accepted" in d:
                                last_draft_accepted = int(d.get("accepted", 0) or 0)
                            if "acceptance_rate" in d:
                                try:
                                    last_acceptance_rate = _normalize_ar(
                                        float(d["acceptance_rate"])
                                    )
                                except (TypeError, ValueError):
                                    pass
                        if "tokens_evaluated" in obj:
                            try:
                                last_prompt_tokens = int(obj["tokens_evaluated"])
                            except (TypeError, ValueError):
                                pass
                        break
                t_close = time.time()
        except (urllib.error.URLError, ConnectionError, OSError) as e:
            return {
                "text": "".join(text_buf),
                "tokens_predicted": token_count,
                "tokens_evaluated": last_prompt_tokens,
                "wall_clock_s": 0.0,
                "tok_per_s": 0.0,
                "acceptance_rate": last_acceptance_rate,
                "draft_proposed": last_draft_proposed,
                "draft_accepted": last_draft_accepted,
                "streaming_measured": True,
                "error": f"stream error: {e}",
            }

        wall_clock = (t_close - t_open) if (t_open and t_close) else 0.0
        first_to_last = (t_close - t_first_token) if (t_first_token and t_close) else wall_clock
        effective = first_to_last if first_to_last > 0 else wall_clock
        tok_per_s = token_count / max(effective, 1e-3)

        return {
            "text": "".join(text_buf),
            "tokens_predicted": token_count,
            "tokens_evaluated": last_prompt_tokens,
            "wall_clock_s": wall_clock,
            "tok_per_s": tok_per_s,
            "acceptance_rate": last_acceptance_rate,
            "draft_proposed": last_draft_proposed,
            "draft_accepted": last_draft_accepted,
            "streaming_measured": True,
        }


# ── SIG client-side token-passthrough helpers ──────────────────────

def build_prompt(target_tokens: int) -> str:
    """Build a prompt whose token-count approximates `target_tokens`.

    The base instruction is fixed; we pad with a long neutral paragraph
    whose token count is roughly proportional to word count. We do not
    pre-tokenize (no model in-process); the server's `tokens_evaluated`
    field gives the actual count, and we record that as `prompt_tokens`.
    """
    base = (
        "You are a careful assistant. The user will ask a question. "
        "Read the background context below and answer concisely.\n\n"
        "Background: "
    )
    filler_paragraph = (
        "The cognitive outsourcing framework proposes that long-running agent "
        "sessions can be accelerated by reusing the KV cache of stable system "
        "prompts across turns. This is a neutral padding paragraph whose sole "
        "purpose is to inflate the prompt token count. "
    ) * 64
    question = (
        "\n\nUser: Summarize the relationship between speculative decoding and "
        "KV-cache reuse in three sentences.\nAssistant:"
    )
    prompt = base + filler_paragraph + question
    approx_tokens = int(len(prompt.split()) / 0.75)
    if approx_tokens > target_tokens * 1.5:
        ratio = (target_tokens * 1.5 * 0.75) / max(len(filler_paragraph.split()), 1)
        prompt = base + filler_paragraph[: int(len(filler_paragraph) * ratio)] + question
    return prompt


def sig_encode_prompt_as_token_ids(
    prompt_text: str,
    n_target: int,
    *,
    n_ctx: int = N_CTX,
) -> List[int]:
    """Encode a text prompt into a synthetic token-id sequence.

    llama-server expects prompt_tokens to be a list of int. Without a local
    tokenizer we synthesize ids by hashing each whitespace-separated chunk
    to a stable int in [0, 2**16). This is not a real BPE encoding, but
    for the SIG client-side passthrough experiment we only care that the
    server accepts the prefix and reports a `tokens_evaluated` value
    proportional to the prefix length. The synthesized id stream is
    deterministic and roughly proportional in length to n_target.
    """
    words = prompt_text.split()
    if not words:
        return []
    # Pad/trim the prompt to roughly n_target tokens of "synthetic" ids.
    if len(words) >= n_target:
        words = words[:n_target]
    else:
        # Pad with deterministic fillers until we hit n_target.
        filler = "fillerfillerfillerfiller"
        i = 0
        while len(words) < n_target:
            words.append(filler)
            i += 1
            if i > n_target * 4:
                break
    ids: List[int] = []
    for w in words:
        h = hash(w) & 0xFFFF
        ids.append(int(h))
    return ids


# ── Experiment runners ─────────────────────────────────────────────

def run_mode_sweep(
    server_bin: Path,
    target_model_path: str,
    draft_model_path: Optional[str],
    mode_specs: List[Tuple[str, str, int, bool]],
    context_lengths: List[int],
    n_runs: int,
    base_port: int,
    max_new_tokens: int,
) -> List[SingleRunResult]:
    """Run a full cartesian sweep over (mode, context_length) x n_runs."""
    all_results: List[SingleRunResult] = []
    for mode_label, spec_type, draft_n, use_draft in mode_specs:
        for ctx in context_lengths:
            prompt = build_prompt(ctx)
            print(f"\n[m={mode_label} ctx={ctx}] preparing...")
            server = LlamaServerMTP(
                binary=server_bin,
                model_path=target_model_path,
                draft_model_path=(draft_model_path if use_draft else None),
                spec_type=spec_type,
                spec_draft_n_max=draft_n,
                n_ctx=N_CTX,
                n_gpu_layers=N_GPU_LAYERS,
                port=base_port,
            )
            try:
                if not server.start():
                    for run_id in range(n_runs):
                        all_results.append(SingleRunResult(
                            mode=mode_label,
                            context_length=ctx,
                            run_id=run_id,
                            wall_clock_s=0.0,
                            gen_tokens=0,
                            prompt_tokens=0,
                            tok_per_s=0.0,
                            acceptance_rate=0.0,
                            draft_proposed=0,
                            draft_accepted=0,
                            spec_type=spec_type,
                            spec_draft_n_max=draft_n,
                            streaming_measured=False,
                            error="server failed to start",
                            skipped=True,
                            skip_reason="server_start_failed",
                        ))
                    continue

                for run_id in range(n_runs):
                    try:
                        res = server.generate_streaming(
                            prompt, max_tokens=max_new_tokens, temperature=TEMPERATURE
                        )
                        streaming_measured = True
                        if res.get("error") or res.get("tokens_predicted", 0) == 0:
                            res = server.generate(
                                prompt, max_tokens=max_new_tokens, temperature=TEMPERATURE
                            )
                            streaming_measured = False
                            ar = _normalize_ar(res.get("acceptance_rate", 0.0) or 0.0)
                            draft = res.get("draft", {}) or {}
                            if ar == 0 and draft.get("accepted", 0) and draft.get("proposed", 0):
                                try:
                                    ar = float(draft["accepted"]) / max(float(draft.get("proposed", 1)), 1.0)
                                    ar = _normalize_ar(ar)
                                except (TypeError, ValueError, ZeroDivisionError):
                                    ar = 0.0
                            all_results.append(SingleRunResult(
                                mode=mode_label,
                                context_length=ctx,
                                run_id=run_id,
                                wall_clock_s=res["wall_clock_s"],
                                gen_tokens=res["tokens_predicted"],
                                prompt_tokens=res["tokens_evaluated"],
                                tok_per_s=res["tok_per_s"],
                                acceptance_rate=ar,
                                draft_proposed=int(draft.get("proposed", 0) or 0),
                                draft_accepted=int(draft.get("accepted", 0) or 0),
                                spec_type=spec_type,
                                spec_draft_n_max=draft_n,
                                streaming_measured=streaming_measured,
                            ))
                        else:
                            ar = _normalize_ar(res.get("acceptance_rate", 0.0) or 0.0)
                            if ar == 0 and res.get("draft_accepted", 0) and res.get("draft_proposed", 0):
                                ar = res["draft_accepted"] / max(res.get("draft_proposed", 1), 1)
                                ar = _normalize_ar(ar)
                            all_results.append(SingleRunResult(
                                mode=mode_label,
                                context_length=ctx,
                                run_id=run_id,
                                wall_clock_s=res["wall_clock_s"],
                                gen_tokens=res["tokens_predicted"],
                                prompt_tokens=res["tokens_evaluated"],
                                tok_per_s=res["tok_per_s"],
                                acceptance_rate=ar,
                                draft_proposed=int(res.get("draft_proposed", 0) or 0),
                                draft_accepted=int(res.get("draft_accepted", 0) or 0),
                                spec_type=spec_type,
                                spec_draft_n_max=draft_n,
                                streaming_measured=streaming_measured,
                            ))
                        last = all_results[-1]
                        print(
                            f"  [m={mode_label} ctx={ctx} run={run_id}] "
                            f"{last.gen_tokens} tok, {last.wall_clock_s:.2f}s, "
                            f"{last.tok_per_s:.1f} tok/s, AR={last.acceptance_rate:.1%}, "
                            f"prompt={last.prompt_tokens}"
                        )
                    except Exception as e:
                        all_results.append(SingleRunResult(
                            mode=mode_label,
                            context_length=ctx,
                            run_id=run_id,
                            wall_clock_s=0.0,
                            gen_tokens=0,
                            prompt_tokens=0,
                            tok_per_s=0.0,
                            acceptance_rate=0.0,
                            draft_proposed=0,
                            draft_accepted=0,
                            spec_type=spec_type,
                            spec_draft_n_max=draft_n,
                            streaming_measured=False,
                            error=str(e),
                        ))
                        print(f"  [m={mode_label} ctx={ctx} run={run_id}] ERROR: {e}")
            finally:
                server.stop()
    return all_results


def run_sig_chat_endtoend(
    server_bin: Path,
    target_model_path: str,
    draft_model_path: Optional[str],
    mode_label: str,
    spec_type: str,
    spec_draft_n_max: int,
    use_draft: bool,
    turns: List[Tuple[str, int]],
    base_port: int,
) -> ChatEndToEndResult:
    """End-to-end multi-turn chat using SIG client-side token passthrough.

    We construct a synthetic prefix-token array representing the full
    conversation history + tool-call result. Each turn appends more
    synthetic ids to the prefix and is sent via `prompt_tokens` to
    llama-server. This measures how MTP behaves over progressively
    longer prefixes without forcing the C++ runtime to re-encode the
    chat template on every turn.
    """
    server = LlamaServerMTP(
        binary=server_bin,
        model_path=target_model_path,
        draft_model_path=(draft_model_path if use_draft else None),
        spec_type=spec_type,
        spec_draft_n_max=spec_draft_n_max,
        n_ctx=N_CTX,
        n_gpu_layers=N_GPU_LAYERS,
        port=base_port,
    )
    if not server.start():
        return ChatEndToEndResult(
            mode=mode_label,
            n_turns=len(turns),
            total_wall_clock_s=0.0,
            mean_tok_per_s=0.0,
            sum_gen_tokens=0,
            turns=[],
            skipped=True,
            skip_reason="server_start_failed",
        )

    # SIG-synthesized prefix token-id stream. We grow it monotonically
    # across turns to mimic the cache-hit-then-decode-new-part pattern.
    prefix_ids: List[int] = [1, 2, 3, 4, 5]
    turn_results: List[ChatTurnResult] = []
    t_total0 = time.time()
    sum_gen = 0
    tok_s_acc: List[float] = []

    try:
        for tid, (user_msg, max_tok) in enumerate(turns):
            # Encode the new turn text into ids and append to prefix.
            new_ids = sig_encode_prompt_as_token_ids(user_msg, n_target=64)
            prefix_ids.extend(new_ids)
            # Trim prefix to N_CTX to avoid OOG.
            if len(prefix_ids) > N_CTX - 256:
                prefix_ids = prefix_ids[-(N_CTX - 256):]
            try:
                res = server.generate_streaming_with_prefix_tokens(
                    prefix_ids, max_tokens=max_tok, temperature=TEMPERATURE
                )
                if res.get("error") or res.get("tokens_predicted", 0) == 0:
                    res = server.generate_with_prefix_tokens(
                        prefix_ids, max_tokens=max_tok, temperature=TEMPERATURE
                    )
                ar = _normalize_ar(res.get("acceptance_rate", 0.0) or 0.0)
                if ar == 0 and res.get("draft_accepted", 0) and res.get("draft_proposed", 0):
                    ar = res["draft_accepted"] / max(res.get("draft_proposed", 1), 1)
                    ar = _normalize_ar(ar)
                cum = time.time() - t_total0
                turn_results.append(ChatTurnResult(
                    turn_id=tid,
                    prompt_tokens=res["tokens_evaluated"],
                    gen_tokens=res["tokens_predicted"],
                    wall_clock_s=res["wall_clock_s"],
                    tok_per_s=res["tok_per_s"],
                    cumulative_wall_clock_s=cum,
                    acceptance_rate=ar,
                    draft_proposed=int(res.get("draft_proposed", 0) or 0),
                    draft_accepted=int(res.get("draft_accepted", 0) or 0),
                ))
                sum_gen += res["tokens_predicted"]
                if res["tok_per_s"] > 0:
                    tok_s_acc.append(res["tok_per_s"])
                print(
                    f"  [chat m={mode_label} turn={tid}] "
                    f"prompt={res['tokens_evaluated']}, gen={res['tokens_predicted']}, "
                    f"{res['wall_clock_s']:.2f}s, {res['tok_per_s']:.1f} tok/s, "
                    f"AR={ar:.1%}, cum={cum:.2f}s"
                )
            except Exception as e:
                turn_results.append(ChatTurnResult(
                    turn_id=tid,
                    prompt_tokens=0,
                    gen_tokens=0,
                    wall_clock_s=0.0,
                    tok_per_s=0.0,
                    cumulative_wall_clock_s=time.time() - t_total0,
                    acceptance_rate=0.0,
                    draft_proposed=0,
                    draft_accepted=0,
                ))
                print(f"  [chat m={mode_label} turn={tid}] ERROR: {e}")
    finally:
        server.stop()

    total_wall = time.time() - t_total0
    mean_tps = statistics.mean(tok_s_acc) if tok_s_acc else 0.0
    return ChatEndToEndResult(
        mode=mode_label,
        n_turns=len(turns),
        total_wall_clock_s=total_wall,
        mean_tok_per_s=mean_tps,
        sum_gen_tokens=sum_gen,
        turns=turn_results,
        skipped=False,
    )


def run_specdec_probe(
    server_bin: Path,
    target_model_path: str,
    draft_model_path: Optional[str],
    spec_type: str,
    spec_draft_n_max: int,
    use_draft: bool,
    probe_lengths: List[int],
    n_runs: int,
    base_port: int,
) -> List[SpecDecProbeResult]:
    """SpecDec compatibility probe across three prompt lengths.

    We do not have a Python API to manipulate the KV cache directly, so
    we exercise the MTP path with very small and very large prompts and
    record pass/fail/oog/oom for each cell.
    """
    server = LlamaServerMTP(
        binary=server_bin,
        model_path=target_model_path,
        draft_model_path=(draft_model_path if use_draft else None),
        spec_type=spec_type,
        spec_draft_n_max=spec_draft_n_max,
        n_ctx=N_CTX,
        n_gpu_layers=N_GPU_LAYERS,
        port=base_port,
    )
    out: List[SpecDecProbeResult] = []
    if not server.start():
        for plen in probe_lengths:
            out.append(SpecDecProbeResult(
                mode=("mtp_n{}".format(spec_draft_n_max) if use_draft else "no_mtp"),
                prompt_length=plen,
                n_runs=n_runs,
                pass_count=0,
                fail_count=n_runs,
                mean_tok_per_s=0.0,
                mean_acceptance_rate=0.0,
                oom_error=False,
                oog_error=False,
                notes="server_start_failed",
            ))
        return out

    mode_label = "mtp_n{}".format(spec_draft_n_max) if use_draft else "no_mtp"
    try:
        for plen in probe_lengths:
            prompt = build_prompt(plen)
            pass_n = 0
            fail_n = 0
            tps_list: List[float] = []
            ar_list: List[float] = []
            oom = False
            oog = False
            err_buf: List[str] = []
            for run_id in range(n_runs):
                try:
                    res = server.generate_streaming(
                        prompt, max_tokens=MAX_NEW_TOKENS, temperature=TEMPERATURE
                    )
                    err = res.get("error", "")
                    if err:
                        err_buf.append(err)
                        if "out of memory" in err.lower() or "oom" in err.lower():
                            oom = True
                        if "out of context" in err.lower() or "oog" in err.lower() or "context" in err.lower():
                            oog = True
                        fail_n += 1
                    elif res.get("tokens_predicted", 0) > 0:
                        pass_n += 1
                        tps_list.append(res.get("tok_per_s", 0.0))
                        ar_list.append(_normalize_ar(res.get("acceptance_rate", 0.0) or 0.0))
                    else:
                        fail_n += 1
                except Exception as e:
                    err_buf.append(str(e))
                    msg = str(e).lower()
                    if "out of memory" in msg or "oom" in msg:
                        oom = True
                    if "out of context" in msg or "oog" in msg or "context" in msg:
                        oog = True
                    fail_n += 1
            out.append(SpecDecProbeResult(
                mode=mode_label,
                prompt_length=plen,
                n_runs=n_runs,
                pass_count=pass_n,
                fail_count=fail_n,
                mean_tok_per_s=statistics.mean(tps_list) if tps_list else 0.0,
                mean_acceptance_rate=statistics.mean(ar_list) if ar_list else 0.0,
                oom_error=oom,
                oog_error=oog,
                notes=("; ".join(err_buf[:2]) if err_buf else ""),
            ))
    finally:
        server.stop()
    return out


# ── Statistics helpers ──────────────────────────────────────────────

def summarize_runs(runs: List[SingleRunResult]) -> ModeContextSummary:
    real = [r for r in runs if not r.skipped and r.tok_per_s > 0]
    skipped = all(r.skipped for r in runs)
    if not real:
        return ModeContextSummary(
            mode=runs[0].mode if runs else "?",
            context_length=runs[0].context_length if runs else 0,
            n_runs=len(runs),
            mean_tok_per_s=0.0,
            std_tok_per_s=0.0,
            mean_wall_clock_s=0.0,
            std_wall_clock_s=0.0,
            mean_gen_tokens=0.0,
            mean_acceptance_rate=0.0,
            mean_draft_proposed=0.0,
            mean_draft_accepted=0.0,
            skipped=skipped,
            skip_reason=(runs[0].skip_reason if runs and runs[0].skipped else ""),
        )
    return ModeContextSummary(
        mode=runs[0].mode,
        context_length=runs[0].context_length,
        n_runs=len(runs),
        mean_tok_per_s=statistics.mean(r.tok_per_s for r in real),
        std_tok_per_s=safe_std([r.tok_per_s for r in real]),
        mean_wall_clock_s=statistics.mean(r.wall_clock_s for r in real),
        std_wall_clock_s=safe_std([r.wall_clock_s for r in real]),
        mean_gen_tokens=statistics.mean(r.gen_tokens for r in real),
        mean_acceptance_rate=statistics.mean(r.acceptance_rate for r in real),
        mean_draft_proposed=statistics.mean(r.draft_proposed for r in real),
        mean_draft_accepted=statistics.mean(r.draft_accepted for r in real),
        skipped=False,
    )


# ── Placeholder mode (no llama-server) ─────────────────────────────

def build_placeholder_runs(
    mode_specs: List[Tuple[str, str, int, bool]],
    context_lengths: List[int],
    n_runs: int,
    reason: str,
) -> List[SingleRunResult]:
    out: List[SingleRunResult] = []
    for mode_label, spec_type, draft_n, _ in mode_specs:
        for ctx in context_lengths:
            for run_id in range(n_runs):
                out.append(SingleRunResult(
                    mode=mode_label,
                    context_length=ctx,
                    run_id=run_id,
                    wall_clock_s=0.0,
                    gen_tokens=0,
                    prompt_tokens=0,
                    tok_per_s=0.0,
                    acceptance_rate=0.0,
                    draft_proposed=0,
                    draft_accepted=0,
                    spec_type=spec_type,
                    spec_draft_n_max=draft_n,
                    streaming_measured=False,
                    error="llama-server not available",
                    skipped=True,
                    skip_reason=reason,
                ))
    return out


def build_placeholder_chat(
    mode_labels: List[str],
    turns: List[Tuple[str, int]],
    reason: str,
) -> List[ChatEndToEndResult]:
    out: List[ChatEndToEndResult] = []
    for m in mode_labels:
        out.append(ChatEndToEndResult(
            mode=m,
            n_turns=len(turns),
            total_wall_clock_s=0.0,
            mean_tok_per_s=0.0,
            sum_gen_tokens=0,
            turns=[
                ChatTurnResult(
                    turn_id=tid,
                    prompt_tokens=0,
                    gen_tokens=0,
                    wall_clock_s=0.0,
                    tok_per_s=0.0,
                    cumulative_wall_clock_s=0.0,
                    acceptance_rate=0.0,
                    draft_proposed=0,
                    draft_accepted=0,
                )
                for tid, _ in enumerate(turns)
            ],
            skipped=True,
            skip_reason=reason,
        ))
    return out


def build_placeholder_specdec(
    mode_specs: List[Tuple[str, str, int, bool]],
    probe_lengths: List[int],
    n_runs: int,
    reason: str,
) -> List[SpecDecProbeResult]:
    out: List[SpecDecProbeResult] = []
    for mode_label, _, draft_n, use_draft in mode_specs:
        if not use_draft:
            continue
        for plen in probe_lengths:
            out.append(SpecDecProbeResult(
                mode=mode_label,
                prompt_length=plen,
                n_runs=n_runs,
                pass_count=0,
                fail_count=n_runs,
                mean_tok_per_s=0.0,
                mean_acceptance_rate=0.0,
                oom_error=False,
                oog_error=False,
                notes=reason,
            ))
    return out


# ── JSON output assembly ───────────────────────────────────────────

def _significance_label(speedup: float) -> str:
    """Return a coarse-grained significance label for a speedup ratio.

    A speedup > 1.0 means SIG+MTP was faster than SIG-only. We apply a
    +/-5% band around 1.0 to label "no significant difference" since
    n=10 with a small model typically has noise of that order.
    """
    if speedup is None or speedup <= 0:
        return "undefined"
    if speedup < 0.95:
        return "sig_mtp_slower"
    if speedup < 1.05:
        return "no_significant_difference"
    if speedup < 1.20:
        return "sig_mtp_faster_marginal"
    if speedup < 1.50:
        return "sig_mtp_faster_moderate"
    return "sig_mtp_faster_strong"


def build_output(
    target_model_path: str,
    draft_model_path: Optional[str],
    n_runs: int,
    context_lengths: List[int],
    mode_specs: List[Tuple[str, str, int, bool]],
    single_runs: List[SingleRunResult],
    chat_results: List[ChatEndToEndResult],
    specdec_results: List[SpecDecProbeResult],
    llama_server_available: bool,
    llama_server_path: Optional[Path],
    target_model_available: bool,
    draft_model_available: bool,
) -> Dict:
    # 1. Per-cell detail
    per_cell_detail: Dict[str, Dict[str, List[Dict]]] = {}
    for mode_label, _, _, _ in mode_specs:
        per_cell_detail[mode_label] = {}
        for ctx in context_lengths:
            cell_runs = [
                asdict(r) for r in single_runs
                if r.mode == mode_label and r.context_length == ctx
            ]
            per_cell_detail[mode_label][str(ctx)] = cell_runs

    # 2. Per-cell summary (mean +/- std)
    per_cell_summary: Dict[str, Dict[str, Dict]] = {}
    for mode_label, _, _, _ in mode_specs:
        per_cell_summary[mode_label] = {}
        for ctx in context_lengths:
            cell = [r for r in single_runs if r.mode == mode_label and r.context_length == ctx]
            s = summarize_runs(cell)
            per_cell_summary[mode_label][str(ctx)] = asdict(s)

    # 3. Mode-level aggregate (pooling across context lengths)
    mode_aggregate: Dict[str, Dict] = {}
    for mode_label, _, _, _ in mode_specs:
        cell_runs = [r for r in single_runs if r.mode == mode_label]
        tps = [r.tok_per_s for r in cell_runs if not r.skipped and r.tok_per_s > 0]
        wall = [r.wall_clock_s for r in cell_runs if not r.skipped and r.wall_clock_s > 0]
        ar = [r.acceptance_rate for r in cell_runs if not r.skipped and r.draft_proposed > 0]
        dp = [r.draft_proposed for r in cell_runs if not r.skipped and r.draft_proposed > 0]
        da = [r.draft_accepted for r in cell_runs if not r.skipped and r.draft_accepted > 0]
        mode_aggregate[mode_label] = {
            "n_runs_total": len(cell_runs),
            "n_runs_real": len(tps),
            "mean_tok_per_s": statistics.mean(tps) if tps else 0.0,
            "std_tok_per_s": safe_std(tps),
            "mean_wall_clock_s": statistics.mean(wall) if wall else 0.0,
            "std_wall_clock_s": safe_std(wall),
            "mean_acceptance_rate": statistics.mean(ar) if ar else 0.0,
            "mean_draft_proposed": statistics.mean(dp) if dp else 0.0,
            "mean_draft_accepted": statistics.mean(da) if da else 0.0,
        }

    # 4. SIG-only vs SIG+MTP end-to-end speedup (orthogonality test).
    # sig_only = chat with no_mtp, sig_mtp = chat with mtp_n2.
    sig_only = next((c for c in chat_results if c.mode == "sig_only"), None)
    sig_mtp = next((c for c in chat_results if c.mode == "sig_mtp"), None)
    orthogonality: Dict[str, object] = {
        "sig_simulation_mode": SIG_SIMULATION_MODE,
        "notes": (
            "SIG effect is approximated by sending a pre-tokenized prefix "
            "via prompt_tokens. MTP runs over the same prefix on the "
            "server side. Real KV-cache slot reuse is not exercised."
        ),
    }
    if sig_only and sig_mtp and not sig_only.skipped and not sig_mtp.skipped:
        if sig_mtp.total_wall_clock_s > 0 and sig_only.total_wall_clock_s > 0:
            speedup = sig_only.total_wall_clock_s / sig_mtp.total_wall_clock_s
            orthogonality["sig_only_total_wall_s"] = sig_only.total_wall_clock_s
            orthogonality["sig_mtp_total_wall_s"] = sig_mtp.total_wall_clock_s
            orthogonality["speedup_x"] = speedup
            orthogonality["significance"] = _significance_label(speedup)
            orthogonality["mean_tok_per_s_sig_only"] = sig_only.mean_tok_per_s
            orthogonality["mean_tok_per_s_sig_mtp"] = sig_mtp.mean_tok_per_s
            orthogonality["sum_gen_tokens_sig_only"] = sig_only.sum_gen_tokens
            orthogonality["sum_gen_tokens_sig_mtp"] = sig_mtp.sum_gen_tokens
        else:
            orthogonality["reason"] = "one_or_both_total_wall_s == 0"
    else:
        orthogonality["reason"] = "one_or_both_modes_skipped"
        orthogonality["sig_only_skipped"] = sig_only.skipped if sig_only else None
        orthogonality["sig_mtp_skipped"] = sig_mtp.skipped if sig_mtp else None

    # 5. SpecDec compatibility
    specdec_compat: Dict[str, object] = {
        "probe_lengths": list(SPECDEC_PROBE_LENGTHS),
        "n_runs_per_probe": n_runs,
        "results": [asdict(r) for r in specdec_results],
    }
    if specdec_results:
        total_pass = sum(r.pass_count for r in specdec_results)
        total_fail = sum(r.fail_count for r in specdec_results)
        oom_any = any(r.oom_error for r in specdec_results)
        oog_any = any(r.oog_error for r in specdec_results)
        specdec_compat["total_pass"] = total_pass
        specdec_compat["total_fail"] = total_fail
        specdec_compat["any_oom"] = oom_any
        specdec_compat["any_oog"] = oog_any
        specdec_compat["overall_pass"] = (total_pass > 0 and not oom_any)

    return {
        "metadata": {
            "experiment": "exp_mtp_gemma4_assistant",
            "version": "1.0",
            "target_model_path": target_model_path,
            "draft_model_path": draft_model_path,
            "target_model_available": target_model_available,
            "draft_model_available": draft_model_available,
            "llama_server_available": llama_server_available,
            "llama_server_path": str(llama_server_path) if llama_server_path else None,
            "n_runs_per_cell": n_runs,
            "context_lengths": list(context_lengths),
            "mode_specs": [
                {"mode": m, "spec_type": st, "spec_draft_n_max": dn, "use_draft": ud}
                for (m, st, dn, ud) in mode_specs
            ],
            "n_ctx": N_CTX,
            "n_gpu_layers": N_GPU_LAYERS,
            "temperature": TEMPERATURE,
            "max_new_tokens": MAX_NEW_TOKENS,
            "sig_simulation_mode": SIG_SIMULATION_MODE,
            "skipped_due_to_no_llama_server": (
                not llama_server_available
                or not target_model_available
                or (draft_model_path and not draft_model_available)
            ),
            "skip_reason": (
                "llama-server binary not found"
                if not llama_server_available
                else "target model file not found"
                if not target_model_available
                else "draft model file not found"
                if (draft_model_path and not draft_model_available)
                else ""
            ),
        },
        "per_cell_detail": per_cell_detail,
        "per_cell_summary": per_cell_summary,
        "mode_aggregate": mode_aggregate,
        "orthogonality_test": orthogonality,
        "specdec_compatibility": specdec_compat,
        "end_to_end_chat": {
            "turns_script": [
                {"turn_id": tid, "prompt": p, "max_new_tokens": m}
                for tid, (p, m) in enumerate(SIG_CHAT_TURNS)
            ],
            "results": [asdict(c) for c in chat_results],
        },
    }


# ── Main orchestration ─────────────────────────────────────────────

def parse_int_list(spec: str) -> List[int]:
    return [int(x.strip()) for x in spec.split(",") if x.strip()]


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Gemma-4-E2B-it + Gemma-4-E2B-it-assistant MTP orthogonality sweep. "
            "Validates that the SIG cache-reuse strategy composes with the "
            "official Gemma-4 MTP draft accelerator."
        )
    )
    parser.add_argument(
        "--target", default=DEFAULT_TARGET,
        help="Path to the main GGUF model (default: %(default)s)",
    )
    parser.add_argument(
        "--draft", default=DEFAULT_DRAFT,
        help="Path to the MTP draft GGUF (default: %(default)s)",
    )
    parser.add_argument("--n-runs", type=int, default=DEFAULT_N_RUNS,
                        help="Runs per cell (default: %(default)s)")
    parser.add_argument(
        "--context-lengths", type=str, default="30,500,1000,2000",
        help="Comma-separated target prompt token counts",
    )
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    parser.add_argument(
        "--output", type=str,
        default="data/exp_mtp_gemma4_assistant_results.json",
        help="Path to the JSON results file",
    )
    parser.add_argument(
        "--no-chat", action="store_true",
        help="Skip the end-to-end multi-turn SIG vs SIG+MTP comparison",
    )
    parser.add_argument(
        "--skip-sweep", action="store_true",
        help="Skip the per-cell sweep (only run the end-to-end chat + specdec)",
    )
    parser.add_argument(
        "--mode", type=str, default="all",
        choices=["all", "no_mtp", "mtp_n1", "mtp_n2", "mtp_n3"],
        help="Restrict to a single mode (default: all 4 modes)",
    )
    parser.add_argument(
        "--task", type=str, default="all",
        choices=["all", "sweep", "chat", "specdec"],
        help=(
            "Restrict the experiment to a single task: "
            "sweep (per-cell), chat (end-to-end), specdec (compatibility probe)"
        ),
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Run only one cell (no_mtp x 30) to verify startup; emit placeholder JSON for the rest",
    )
    args = parser.parse_args()

    context_lengths = parse_int_list(args.context_lengths)

    if args.mode == "all":
        mode_specs = list(FULL_MODE_SPECS)
    else:
        mode_specs = [m for m in FULL_MODE_SPECS if m[0] == args.mode]

    target_model_path = args.target
    draft_model_path = args.draft

    target_model_available = detect_model(target_model_path)
    draft_model_available = detect_model(draft_model_path) if draft_model_path else False
    llama_available, server_bin = detect_llama_server()

    print("=" * 70)
    print("Gemma-4-E2B-it + Gemma-4-E2B-it-assistant MTP Orthogonality Sweep")
    print("=" * 70)
    print(f"  target_model_path     = {target_model_path}")
    print(f"  target_model_avail    = {target_model_available}")
    print(f"  draft_model_path      = {draft_model_path}")
    print(f"  draft_model_avail     = {draft_model_available}")
    print(f"  llama_server_avail    = {llama_available}")
    if server_bin:
        print(f"  llama_server_path     = {server_bin}")
    print(f"  n_runs                = {args.n_runs}")
    print(f"  context_lengths       = {context_lengths}")
    print(f"  max_new_tokens        = {args.max_new_tokens}")
    print(f"  port                  = {args.port}")
    print(f"  mode_specs            = {[m[0] for m in mode_specs]}")
    print(f"  task                  = {args.task}")
    print(f"  dry_run               = {args.dry_run}")

    can_run = llama_available and target_model_available and draft_model_available

    do_sweep = (args.task in ("all", "sweep")) and not args.skip_sweep and not args.dry_run
    do_chat = (args.task in ("all", "chat")) and not args.no_chat
    do_specdec = (args.task in ("all", "specdec"))

    if args.dry_run:
        do_sweep = False
        do_chat = False
        do_specdec = False
        # One cell only: no_mtp x 30.
        single_runs: List[SingleRunResult] = []
        chat_results: List[ChatEndToEndResult] = []
        specdec_results: List[SpecDecProbeResult] = []
        if can_run:
            try:
                server = LlamaServerMTP(
                    binary=server_bin,
                    model_path=target_model_path,
                    draft_model_path=None,
                    spec_type="none",
                    spec_draft_n_max=0,
                    n_ctx=N_CTX,
                    n_gpu_layers=N_GPU_LAYERS,
                    port=args.port,
                )
                if server.start():
                    prompt = build_prompt(30)
                    res = server.generate_streaming(
                        prompt, max_tokens=args.max_new_tokens, temperature=TEMPERATURE
                    )
                    streaming_measured = True
                    if res.get("error") or res.get("tokens_predicted", 0) == 0:
                        res = server.generate(
                            prompt, max_tokens=args.max_new_tokens, temperature=TEMPERATURE
                        )
                        streaming_measured = False
                    ar = _normalize_ar(res.get("acceptance_rate", 0.0) or 0.0)
                    single_runs.append(SingleRunResult(
                        mode="no_mtp",
                        context_length=30,
                        run_id=0,
                        wall_clock_s=res["wall_clock_s"],
                        gen_tokens=res["tokens_predicted"],
                        prompt_tokens=res["tokens_evaluated"],
                        tok_per_s=res["tok_per_s"],
                        acceptance_rate=ar,
                        draft_proposed=int(res.get("draft_proposed", 0) or 0),
                        draft_accepted=int(res.get("draft_accepted", 0) or 0),
                        spec_type="none",
                        spec_draft_n_max=0,
                        streaming_measured=streaming_measured,
                    ))
                    print(
                        f"  [dry-run no_mtp x 30] {single_runs[-1].gen_tokens} tok, "
                        f"{single_runs[-1].wall_clock_s:.2f}s, "
                        f"{single_runs[-1].tok_per_s:.1f} tok/s"
                    )
                else:
                    single_runs.append(SingleRunResult(
                        mode="no_mtp",
                        context_length=30,
                        run_id=0,
                        wall_clock_s=0.0,
                        gen_tokens=0,
                        prompt_tokens=0,
                        tok_per_s=0.0,
                        acceptance_rate=0.0,
                        draft_proposed=0,
                        draft_accepted=0,
                        spec_type="none",
                        spec_draft_n_max=0,
                        streaming_measured=False,
                        error="server failed to start",
                        skipped=True,
                        skip_reason="server_start_failed",
                    ))
                server.stop()
            except Exception as e:
                print(f"  [dry-run] ERROR: {e}")
                if not single_runs:
                    single_runs.append(SingleRunResult(
                        mode="no_mtp",
                        context_length=30,
                        run_id=0,
                        wall_clock_s=0.0,
                        gen_tokens=0,
                        prompt_tokens=0,
                        tok_per_s=0.0,
                        acceptance_rate=0.0,
                        draft_proposed=0,
                        draft_accepted=0,
                        spec_type="none",
                        spec_draft_n_max=0,
                        streaming_measured=False,
                        error=str(e),
                        skipped=True,
                        skip_reason="dry_run_exception",
                    ))
        else:
            single_runs.append(SingleRunResult(
                mode="no_mtp",
                context_length=30,
                run_id=0,
                wall_clock_s=0.0,
                gen_tokens=0,
                prompt_tokens=0,
                tok_per_s=0.0,
                acceptance_rate=0.0,
                draft_proposed=0,
                draft_accepted=0,
                spec_type="none",
                spec_draft_n_max=0,
                streaming_measured=False,
                error="llama-server not available",
                skipped=True,
                skip_reason="dry_run_no_llama_server",
            ))
    elif not can_run:
        reason = (
            "llama-server binary not found"
            if not llama_available
            else f"target model file not found: {target_model_path}"
            if not target_model_available
            else f"draft model file not found: {draft_model_path}"
        )
        print(f"\n[warn] Cannot run: {reason}")
        print("  Emitting placeholder JSON with the full cartesian product of cells.")
        single_runs = build_placeholder_runs(
            mode_specs, context_lengths, args.n_runs,
            reason="skipped_due_to_no_llama_server",
        )
        chat_results: List[ChatEndToEndResult] = []
        if do_chat:
            chat_results = build_placeholder_chat(
                ["sig_only", "sig_mtp"], SIG_CHAT_TURNS,
                reason="skipped_due_to_no_llama_server",
            )
        specdec_results: List[SpecDecProbeResult] = []
        if do_specdec:
            specdec_results = build_placeholder_specdec(
                mode_specs, list(SPECDEC_PROBE_LENGTHS), args.n_runs,
                reason="skipped_due_to_no_llama_server",
            )
    else:
        single_runs: List[SingleRunResult] = []
        if do_sweep:
            print("\n>>> Running per-cell sweep")
            single_runs = run_mode_sweep(
                server_bin=server_bin,
                target_model_path=target_model_path,
                draft_model_path=draft_model_path,
                mode_specs=mode_specs,
                context_lengths=context_lengths,
                n_runs=args.n_runs,
                base_port=args.port,
                max_new_tokens=args.max_new_tokens,
            )
        else:
            print("\n>>> Skipping per-cell sweep")
            single_runs = build_placeholder_runs(
                mode_specs, context_lengths, args.n_runs,
                reason="skipped_by_user",
            )

        chat_results: List[ChatEndToEndResult] = []
        if do_chat:
            print("\n>>> Running end-to-end SIG-only vs SIG+MTP chat (client-side token passthrough)")
            print("\n[chat A] SIG-only (no_mtp)...")
            chat_results.append(run_sig_chat_endtoend(
                server_bin=server_bin,
                target_model_path=target_model_path,
                draft_model_path=draft_model_path,
                mode_label="sig_only",
                spec_type="none",
                spec_draft_n_max=0,
                use_draft=False,
                turns=SIG_CHAT_TURNS,
                base_port=args.port,
            ))
            print("\n[chat B] SIG + MTP (mtp_n2)...")
            chat_results.append(run_sig_chat_endtoend(
                server_bin=server_bin,
                target_model_path=target_model_path,
                draft_model_path=draft_model_path,
                mode_label="sig_mtp",
                spec_type="draft-mtp",
                spec_draft_n_max=2,
                use_draft=True,
                turns=SIG_CHAT_TURNS,
                base_port=args.port,
            ))

        specdec_results: List[SpecDecProbeResult] = []
        if do_specdec:
            print("\n>>> Running SpecDec compatibility probe (mtp_n2 across {30, 2000, 8000})")
            specdec_results = run_specdec_probe(
                server_bin=server_bin,
                target_model_path=target_model_path,
                draft_model_path=draft_model_path,
                spec_type="draft-mtp",
                spec_draft_n_max=2,
                use_draft=True,
                probe_lengths=list(SPECDEC_PROBE_LENGTHS),
                n_runs=max(1, args.n_runs // 2),
                base_port=args.port,
            )

    out_doc = build_output(
        target_model_path=target_model_path,
        draft_model_path=draft_model_path,
        n_runs=args.n_runs,
        context_lengths=context_lengths,
        mode_specs=mode_specs,
        single_runs=single_runs,
        chat_results=chat_results,
        specdec_results=specdec_results,
        llama_server_available=llama_available,
        llama_server_path=server_bin,
        target_model_available=target_model_available,
        draft_model_available=draft_model_available,
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_doc, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: {out_path}")
    print(f"  total cells: {len(mode_specs) * len(context_lengths) * args.n_runs}")
    print(f"  skipped:     {sum(1 for r in single_runs if r.skipped)}")
    print(f"  completed:   {sum(1 for r in single_runs if not r.skipped)}")
    speedup = out_doc["orthogonality_test"].get("speedup_x") if isinstance(out_doc["orthogonality_test"], dict) else None
    if speedup is not None:
        print(
            f"  SIG vs SIG+MTP:  speedup={speedup:.3f}x "
            f"({out_doc['orthogonality_test'].get('significance', '?')})"
        )
    if specdec_results:
        total_pass = sum(r.pass_count for r in specdec_results)
        total_fail = sum(r.fail_count for r in specdec_results)
        print(f"  SpecDec probe:  pass={total_pass} fail={total_fail}")


if __name__ == "__main__":
    main()
