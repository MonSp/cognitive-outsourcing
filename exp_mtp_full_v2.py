"""Experiment: P5 Native MTP Full Sweep v2 (n=10, multi-context, SIG vs SIG+MTP)

This is the reinforced P5 experiment that extends the original n=3 MTP study to:
  - n=10 runs per condition (statistically meaningful mean +/- std)
  - Four MTP modes: no_mtp, mtp_n1, mtp_n2, mtp_n3
  - Four context lengths: 30, 500, 1000, 2000 tokens
  - End-to-end wall-clock comparison: SIG-only vs SIG+MTP
  - llama-server subprocess lifecycle management (start -> healthcheck -> test -> stop)

The script deliberately bypasses the llama-cpp-python Python API and drives
llama.cpp's native MTP path via the OpenAI-compatible HTTP surface exposed by
llama-server. This is the only path known to correctly handle the hybrid
attention barrier documented in P5.

If llama-server is not available on the current host, the script still emits
a complete JSON document containing the full cartesian product of conditions
(modes x context lengths x runs) with `skipped_due_to_no_llama_server` flags,
so downstream analysis can be re-run when the binary becomes available.
"""

import argparse
import json
import math
import os
import random
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

DEFAULT_MODEL = str(MODELS_DIR / "Qwen3.5-4B-Q4_K_M.gguf")
MTP_MODEL = str(MODELS_DIR / "mtp" / "Qwen3.5-4B-Q4_K_M.gguf")

N_CTX = 16384
N_GPU_LAYERS = 99
N_THREADS = 4
TEMPERATURE = 0.0
MAX_NEW_TOKENS = 64
DEFAULT_PORT = 8082
HEALTHCHECK_TIMEOUT_S = 180.0
HEALTHCHECK_INTERVAL_S = 2.0
DEFAULT_CONTEXT_LENGTHS = (30, 500, 1000, 2000)
DEFAULT_N_RUNS = 10

# Multi-turn chat script used for end-to-end SIG vs SIG+MTP comparison.
# Each tuple is (user_utterance, max_new_tokens_for_turn).
SIG_CHAT_TURNS: List[Tuple[str, int]] = [
    ("Hi, I need a 3-step plan to write a SIG benchmark report.", 80),
    ("Actually make it 4 steps and emphasize reproducibility.", 80),
    ("Now give a 2-sentence summary I can paste into the abstract.", 60),
    ("Thanks. Finally, list 3 risks I should mitigate.", 60),
]

# ── EdgeAgent-Kitchen scenario (unified protocol with Paper5 Section 5.1-5.6) ──

KITCHEN_SYSTEM_PROMPT = """You are an intelligent kitchen assistant running on an edge device.
You help users with recipe planning, real-time cooking guidance, inventory management,
and handling interruptions. Always consider dietary profile and kitchen state.
Be concise and specific."""

# Simplified Kitchen scenario for llama-server benchmarking.
# Each tuple: (user_query, tool_name, tool_response_text)
# We pre-compute tool responses to avoid needing the full KitchenToolRegistry.
KITCHEN_SCENARIO_TURNS: List[Tuple[str, str, str]] = [
    # (user_query, tool_name, tool_response)
    ("Set family profile: 4 people, nut allergies, prefer Italian.", "set_user_profile",
     "Profile: Family, allergies=nuts, diet=omnivore, servings=4, cuisine=italian"),
    ("Stock pantry: pasta 600g.", "add_to_pantry", "Added 600g pasta to pantry (now 600g)"),
    ("Stock pantry: rice 800g.", "add_to_pantry", "Added 800g rice to pantry (now 800g)"),
    ("Stock pantry: olive_oil 500g.", "add_to_pantry", "Added 500g olive_oil to pantry (now 500g)"),
    ("Stock fridge: chicken_breast 500g.", "add_to_fridge", "Added 500g chicken_breast to fridge (now 500g)"),
    ("Stock fridge: eggs 300g.", "add_to_fridge", "Added 300g eggs to fridge (now 300g)"),
    ("Plan Monday dinner: italian, under 45min.", "find_recipes",
     "Found: spaghetti_bolognese (35min, 650cal), caprese_salad (10min, 280cal)"),
    ("Get spaghetti bolognese recipe.", "get_recipe",
     "Spaghetti Bolognese: pasta, beef, tomato, onion, garlic, olive_oil, oregano, salt, pepper, cheese_parmesan. Steps: Chop onion and garlic; Brown beef; Add tomato and oregano simmer 20min; Boil pasta 8min; Serve with parmesan. Time: 35min"),
    ("Check ingredients for spaghetti_bolognese.", "check_ingredients",
     "Missing: beef, cheese_parmesan. Have: pasta, onion, garlic, olive_oil, oregano, salt, pepper, tomato"),
    ("Preheat oven to 180C.", "set_oven", "Oven set to 180C, preheating."),
    ("Start cooking spaghetti_bolognese.", "start_cooking", "Started spaghetti_bolognese. Step 1: Chop onion and garlic"),
    ("Next step?", "next_step", "Step 2: Brown beef in olive oil"),
    ("Next step?", "next_step", "Step 3: Add tomato and oregano, simmer 20min"),
    ("Set 15min timer for sauce.", "set_timer", "Timer set: 15 minutes for sauce"),
    ("Check pantry.", "check_pantry", "Pantry: pasta 600g, rice 800g, olive_oil 500g, salt 300g, pepper 100g, oregano 50g, garlic 200g, onion 300g, tomato 500g, soy_sauce 300g, ginger 100g"),
    ("Add beef to shopping list.", "add_shopping_item", "Added beef x2 to shopping list"),
    ("Switch to Asian cuisine. Find recipes.", "find_recipes",
     "Found: chicken_stir_fry (25min, 480cal, asian), salmon_teriyaki (20min, 520cal, asian)"),
    ("Get chicken stir fry recipe.", "get_recipe",
     "Chicken Stir Fry: chicken_breast, broccoli, bell_pepper, carrot, soy_sauce, ginger, garlic, rice, olive_oil. Steps: Slice chicken and vegetables; Heat oil in wok; Stir-fry chicken 5min; Add vegetables 3min; Add soy sauce and ginger; Serve over rice. Time: 25min"),
    ("Check ingredients for chicken_stir_fry.", "check_ingredients",
     "Missing: broccoli, bell_pepper, carrot. Have: chicken_breast, soy_sauce, ginger, garlic, rice, olive_oil"),
    ("Substitute for cheese_parmesan?", "get_substitution", "Substitution for cheese_parmesan: almonds"),
    ("Check fridge.", "check_fridge", "Fridge: chicken_breast 500g, eggs 300g, butter 200g, cheese_parmesan 200g, milk 500g"),
    ("Nutrition for chicken_stir_fry?", "get_nutrition", "Nutrition for Chicken Stir Fry: 480 calories, allergens: soy"),
    ("Plan Tuesday dinner: italian, under 30min.", "find_recipes",
     "Found: caprese_salad (10min, 280cal, italian), omelette (8min, 320cal, french)"),
    ("Start cooking chicken_stir_fry.", "start_cooking", "Started chicken_stir_fry. Step 1: Slice chicken and vegetables"),
    ("Next step?", "next_step", "Step 2: Heat oil in wok"),
    ("Next step?", "next_step", "Step 3: Stir-fry chicken 5min"),
    ("Oven ready?", "get_oven_status", "Oven: 180C, ready"),
    ("Compare prices: olive_oil, garlic, basil.", "compare_prices",
     "Price comparison: olive_oil $5.00, garlic $0.20, basil $2.00"),
    ("Show shopping list.", "get_shopping_list", "Shopping list: beef x2 ($12.00), broccoli x1 ($1.50), bell_pepper x1 ($1.50), carrot x1 ($1.00). Total: $16.00"),
    ("Next step?", "next_step", "Step 4: Add vegetables 3min"),
    ("Mother-in-law is vegetarian. What can we make?", "find_recipes",
     "Found: vegetable_curry (40min, 420cal, indian, vegetarian), caprese_salad (10min, 280cal, italian, vegetarian)"),
    ("Next step?", "next_step", "Step 5: Add soy sauce and ginger"),
    ("Next step?", "next_step", "Step 6: Serve over rice"),
    ("Get mushroom risotto recipe.", "get_recipe",
     "Mushroom Risotto: rice, mushroom, onion, garlic, butter, cheese_parmesan, wine_white, olive_oil, salt, pepper. Steps: Saute onion and garlic in butter; Add rice toast 2min; Add wine stir until absorbed; Add broth gradually stirring 18min; Add mushrooms cook 5min; Finish with parmesan. Time: 35min"),
    ("Nutrition for mushroom_risotto?", "get_nutrition", "Nutrition for Mushroom Risotto: 550 calories, allergens: dairy"),
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
    ci_low_wall_clock_s: float = 0.0
    ci_high_wall_clock_s: float = 0.0
    ci_low_tok_per_s: float = 0.0
    ci_high_tok_per_s: float = 0.0
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
class KitchenTurnResult:
    """One turn of the EdgeAgent-Kitchen benchmark."""
    turn_id: int
    condition: str  # "apploop", "sig", "apploop_mtp", "sig_mtp"
    user_query: str
    tool_name: str
    prompt_tokens: int
    gen_tokens: int
    wall_clock_s: float
    tok_per_s: float
    prompt_eval_s: float  # time spent on prompt evaluation
    acceptance_rate: float
    draft_proposed: int
    draft_accepted: int


@dataclass
class KitchenSessionResult:
    """Full session result for one condition of the Kitchen benchmark."""
    condition: str
    n_turns: int
    total_wall_clock_s: float
    mean_tok_per_s: float
    sum_gen_tokens: int
    turns: List[KitchenTurnResult] = field(default_factory=list)
    skipped: bool = False
    skip_reason: str = ""


# ── llama-server detection ──────────────────────────────────────────

def detect_llama_server() -> Tuple[bool, Optional[Path]]:
    """Return (available, resolved_binary_path).

    Resolution order:
      1. Explicit path at LLAMA_CPP_BIN_DIR
      2. PATH lookup for "llama-server" / "llama-server.exe"
    """
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


# ── LlamaServerMTP class ───────────────────────────────────────────

class LlamaServerMTP:
    """Manages a llama-server subprocess with MTP speculative decoding.

    Lifecycle: __init__ -> start() -> generate/stream -> stop()
    Each instance binds a single port; create a new instance per (mode, ctx).
    """

    def __init__(
        self,
        binary: Path,
        model_path: str,
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
        # Only add MTP args when the mode requires them. llama-server with
        # --spec-type none or empty is the most compatible baseline.
        if self.spec_type and self.spec_type != "none":
            cmd += ["--spec-type", self.spec_type]
            if self.spec_draft_n_max > 0:
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
                # Process exited early; capture any output
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
                try:
                    proc.send_signal(signal.CTRL_BREAK_EVENT)
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
        """Streaming completion; measures tok/s by token-arrival cadence.

        Uses /completion with stream=True. Each SSE chunk contains a "content"
        delta and (in llama-server >= b9000) incremental timings and draft
        statistics. We reconstruct:
          - tokens streamed (count of content deltas)
          - wall-clock from first byte to last byte
          - streaming tok/s = tokens / wall_clock
          - acceptance_rate / draft counts from final aggregate fields
        """
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
                    # Stop tokens can appear as content=""
                    delta = obj.get("content", "")
                    if delta:
                        text_buf.append(delta)
                        token_count += 1
                        if t_first_token is None:
                            t_first_token = time.time()
                    # Capture incremental aggregate fields if present
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
                        # End-of-stream signal
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
        # tok/s is most meaningful from first-token to last-token (decoding
        # throughput, excluding the prompt evaluation that happens first).
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

    def chat(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = MAX_NEW_TOKENS,
        temperature: float = TEMPERATURE,
    ) -> Dict:
        """OpenAI-compatible chat completion (used for SIG/SIG+MTP end-to-end)."""
        payload = {
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
        }
        t0 = time.time()
        result = self._post("/v1/chat/completions", payload)
        wall_clock = time.time() - t0
        choices = result.get("choices", []) or [{}]
        text = (choices[0] or {}).get("message", {}).get("content", "") or ""
        usage = result.get("usage", {}) or {}
        timings = result.get("timings", {}) or {}
        return {
            "text": text,
            "tokens_predicted": int(usage.get("completion_tokens", 0) or 0),
            "tokens_evaluated": int(usage.get("prompt_tokens", 0) or 0),
            "wall_clock_s": wall_clock,
            "tok_per_s": (
                int(usage.get("completion_tokens", 0) or 0) / max(wall_clock, 1e-3)
            ),
            "acceptance_rate": _normalize_ar(float(timings.get("draft_accept_rate", 0.0) or 0.0)),
            "draft_proposed": int(timings.get("draft_n_total", 0) or 0),
            "draft_accepted": int(timings.get("draft_n_accepted", 0) or 0),
        }

    def chat_streaming(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = MAX_NEW_TOKENS,
        temperature: float = TEMPERATURE,
    ) -> Dict:
        """Streaming chat completion. Returns per-turn dict similar to
        generate_streaming but with chat-message payload.
        """
        payload = {
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True,
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"http://127.0.0.1:{self.port}/v1/chat/completions",
            data=data,
            headers={"Content-Type": "application/json"},
        )
        token_count = 0
        text_buf: List[str] = []
        prompt_tokens = 0
        completion_tokens = 0
        last_draft_proposed = 0
        last_draft_accepted = 0
        last_acceptance_rate = 0.0
        t_open = t_close = t_first = None
        try:
            with urllib.request.urlopen(req, timeout=600) as resp:
                t_open = time.time()
                for raw_line in resp:
                    if not raw_line:
                        continue
                    line = raw_line.decode("utf-8", errors="replace").rstrip("\r\n")
                    if not line.startswith("data:"):
                        continue
                    body = line[5:].strip()
                    if not body or body == "[DONE]":
                        continue
                    try:
                        obj = json.loads(body)
                    except json.JSONDecodeError:
                        continue
                    delta = ""
                    try:
                        delta = obj["choices"][0]["delta"].get("content", "") or ""
                    except (KeyError, IndexError, TypeError):
                        delta = ""
                    if delta:
                        text_buf.append(delta)
                        token_count += 1
                        if t_first is None:
                            t_first = time.time()
                    if "usage" in obj and obj["usage"]:
                        prompt_tokens = int(obj["usage"].get("prompt_tokens", prompt_tokens) or 0)
                        completion_tokens = int(
                            obj["usage"].get("completion_tokens", completion_tokens) or 0
                        )
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
                t_close = time.time()
        except (urllib.error.URLError, ConnectionError, OSError) as e:
            return {
                "text": "".join(text_buf),
                "tokens_predicted": token_count,
                "tokens_evaluated": prompt_tokens,
                "wall_clock_s": 0.0,
                "tok_per_s": 0.0,
                "acceptance_rate": last_acceptance_rate,
                "draft_proposed": last_draft_proposed,
                "draft_accepted": last_draft_accepted,
                "error": f"chat stream error: {e}",
            }

        wall_clock = (t_close - t_open) if (t_open and t_close) else 0.0
        first_to_last = (t_close - t_first) if (t_first and t_close) else wall_clock
        effective = first_to_last if first_to_last > 0 else wall_clock
        tok_per_s = token_count / max(effective, 1e-3)
        return {
            "text": "".join(text_buf),
            "tokens_predicted": token_count or completion_tokens,
            "tokens_evaluated": prompt_tokens,
            "wall_clock_s": wall_clock,
            "tok_per_s": tok_per_s,
            "acceptance_rate": last_acceptance_rate,
            "draft_proposed": last_draft_proposed,
            "draft_accepted": last_draft_accepted,
        }


# ── Prompt generation for context-length targets ───────────────────

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
    # Crude heuristic: ~1 token per 0.75 word. Trim filler if we overshoot.
    approx_tokens = int(len(prompt.split()) / 0.75)
    if approx_tokens > target_tokens * 1.5:
        ratio = (target_tokens * 1.5 * 0.75) / max(len(filler_paragraph.split()), 1)
        prompt = base + filler_paragraph[: int(len(filler_paragraph) * ratio)] + question
    return prompt


# ── Experiment runners ─────────────────────────────────────────────

def run_mode_sweep(
    server_bin: Path,
    model_path: str,
    mode_specs: List[Tuple[str, str, int]],
    context_lengths: List[int],
    n_runs: int,
    base_port: int,
    max_new_tokens: int,
) -> List[SingleRunResult]:
    """Run a full cartesian sweep over (mode, context_length) x n_runs.

    mode_specs: list of (mode_label, spec_type, spec_draft_n_max)
    """
    all_results: List[SingleRunResult] = []
    for mode_label, spec_type, draft_n in mode_specs:
        for ctx in context_lengths:
            prompt = build_prompt(ctx)
            print(f"\n[m={mode_label} ctx={ctx}] preparing...")
            port = base_port
            server = LlamaServerMTP(
                binary=server_bin,
                model_path=model_path,
                spec_type=spec_type,
                spec_draft_n_max=draft_n,
                n_ctx=N_CTX,
                n_gpu_layers=N_GPU_LAYERS,
                port=port,
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
                        # Use streaming for accurate tok/s; fall back to
                        # non-streaming on protocol error.
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
                            # Fallback: if AR was not reported, compute from
                            # accepted/proposed counts. Only do this when
                            # `draft_proposed` was actually reported (i.e. > 0),
                            # otherwise the ratio is meaningless.
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


def run_chat_endtoend(
    server_bin: Path,
    model_path: str,
    mode_label: str,
    spec_type: str,
    spec_draft_n_max: int,
    turns: List[Tuple[str, int]],
    base_port: int,
) -> ChatEndToEndResult:
    """Run a multi-turn chat and measure end-to-end wall-clock.

    SIG-only = `--spec-type none` (no MTP, but the SIG "philosophy" of
    preserving context across turns is approximated by always passing the
    full message history; we cannot do real KV-cache injection from outside
    the C++ runtime via HTTP alone, so we approximate by serializing the
    full conversation as chat messages).
    SIG+MTP = `--spec-type draft-mtp` with the same message-history strategy.
    """
    server = LlamaServerMTP(
        binary=server_bin,
        model_path=model_path,
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

    history: List[Dict[str, str]] = [
        {"role": "system", "content": "You are a concise, helpful assistant."}
    ]
    turn_results: List[ChatTurnResult] = []
    t_total0 = time.time()
    sum_gen = 0
    tok_s_acc: List[float] = []

    try:
        for tid, (user_msg, max_tok) in enumerate(turns):
            history.append({"role": "user", "content": user_msg})
            try:
                res = server.chat_streaming(history, max_tokens=max_tok, temperature=TEMPERATURE)
                if res.get("error") or res.get("tokens_predicted", 0) == 0:
                    res = server.chat(history, max_tokens=max_tok, temperature=TEMPERATURE)
                ar = _normalize_ar(res.get("acceptance_rate", 0.0) or 0.0)
                if ar == 0 and res.get("draft_accepted", 0) and res.get("draft_proposed", 0):
                    ar = res["draft_accepted"] / max(res.get("draft_proposed", 1), 1)
                    ar = _normalize_ar(ar)
                history.append({"role": "assistant", "content": res.get("text", "")})
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
                history.append({"role": "assistant", "content": f"[error: {e}]"})
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


# ── Statistics helpers ──────────────────────────────────────────────

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


def bootstrap_ci(data: List[float], n_resamples: int = 10000, ci: float = 0.95) -> Tuple[float, float, float]:
    """Bootstrap confidence interval for the mean.

    Returns (mean, ci_low, ci_high).
    """
    if not data:
        return 0.0, 0.0, 0.0
    mean_val = statistics.mean(data)
    if len(data) < 2:
        return mean_val, mean_val, mean_val
    resampled_means = []
    for _ in range(n_resamples):
        sample = random.choices(data, k=len(data))
        resampled_means.append(statistics.mean(sample))
    resampled_means.sort()
    alpha = 1.0 - ci
    lo_idx = int(math.floor((alpha / 2.0) * n_resamples))
    hi_idx = int(math.floor((1.0 - alpha / 2.0) * n_resamples))
    lo_idx = max(lo_idx, 0)
    hi_idx = min(hi_idx, n_resamples - 1)
    return mean_val, resampled_means[lo_idx], resampled_means[hi_idx]


def cohens_d(group1: List[float], group2: List[float]) -> float:
    """Pooled standard deviation Cohen's d.

    d = (mean1 - mean2) / sqrt((var1 + var2) / 2)
    Returns 0.0 if both groups have zero variance.
    """
    if not group1 or not group2:
        return 0.0
    mean1 = statistics.mean(group1)
    mean2 = statistics.mean(group2)
    var1 = statistics.variance(group1) if len(group1) >= 2 else 0.0
    var2 = statistics.variance(group2) if len(group2) >= 2 else 0.0
    pooled_var = (var1 + var2) / 2.0
    if pooled_var == 0.0:
        return 0.0
    return (mean1 - mean2) / math.sqrt(pooled_var)


def approximate_power(d: float, n: int, alpha: float = 0.05) -> float:
    """Approximate statistical power for two-tailed t-test.

    Uses non-centrality parameter: lambda = d * sqrt(n/2).
    Power = 1 - beta where beta = CDF(t_critical - lambda).
    Uses math.erf for approximate normal CDF.
    """
    if n < 2 or d == 0.0:
        return 0.0
    # t_critical for two-tailed test at alpha (approximate via normal)
    z_alpha = _normal_ppf(1.0 - alpha / 2.0)
    ncp = d * math.sqrt(n / 2.0)  # non-centrality parameter
    beta = _normal_cdf(z_alpha - ncp) - _normal_cdf(-z_alpha - ncp)
    power = 1.0 - beta
    return max(0.0, min(1.0, power))


def _normal_cdf(x: float) -> float:
    """Standard normal CDF using math.erf."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _normal_ppf(p: float) -> float:
    """Approximate inverse normal CDF (ppf) using rational approximation.

    Abramowitz & Stegun approximation 26.2.23.
    """
    if p <= 0.0:
        return -8.0
    if p >= 1.0:
        return 8.0
    if p < 0.5:
        return -_normal_ppf(1.0 - p)
    # Rational approximation for 0.5 <= p < 1
    t = math.sqrt(-2.0 * math.log(1.0 - p))
    c0 = 2.515517
    c1 = 0.802853
    c2 = 0.010328
    d1 = 1.432788
    d2 = 0.189269
    d3 = 0.001308
    return t - (c0 + c1 * t + c2 * t * t) / (1.0 + d1 * t + d2 * t * t + d3 * t * t * t)


def compute_statistical_analysis(runs: List[SingleRunResult]) -> Dict:
    """Compute statistical analysis for one condition.

    Filters to non-skipped runs with tok_per_s > 0.
    Returns dict with mean, std, bootstrap_ci for key metrics.
    """
    real = [r for r in runs if not r.skipped and r.tok_per_s > 0]
    result: Dict = {}

    if not real:
        result["n_valid_runs"] = 0
        result["wall_clock"] = {"mean": 0.0, "std": 0.0, "ci_low": 0.0, "ci_high": 0.0}
        result["tok_per_s"] = {"mean": 0.0, "std": 0.0, "ci_low": 0.0, "ci_high": 0.0}
        result["acceptance_rate"] = {"mean": 0.0, "std": 0.0, "ci_low": 0.0, "ci_high": 0.0}
        return result

    result["n_valid_runs"] = len(real)

    wall_vals = [r.wall_clock_s for r in real]
    tps_vals = [r.tok_per_s for r in real]
    ar_runs = [r for r in real if r.draft_proposed > 0]
    ar_vals = [r.acceptance_rate for r in ar_runs]

    wall_mean, wall_ci_lo, wall_ci_hi = bootstrap_ci(wall_vals)
    tps_mean, tps_ci_lo, tps_ci_hi = bootstrap_ci(tps_vals)

    result["wall_clock"] = {
        "mean": wall_mean,
        "std": safe_std(wall_vals),
        "ci_low": wall_ci_lo,
        "ci_high": wall_ci_hi,
    }
    result["tok_per_s"] = {
        "mean": tps_mean,
        "std": safe_std(tps_vals),
        "ci_low": tps_ci_lo,
        "ci_high": tps_ci_hi,
    }

    if ar_vals:
        ar_mean, ar_ci_lo, ar_ci_hi = bootstrap_ci(ar_vals)
        result["acceptance_rate"] = {
            "mean": ar_mean,
            "std": safe_std(ar_vals),
            "ci_low": ar_ci_lo,
            "ci_high": ar_ci_hi,
        }
    else:
        result["acceptance_rate"] = {"mean": 0.0, "std": 0.0, "ci_low": 0.0, "ci_high": 0.0}

    return result


def compute_pairwise_comparison(
    runs_a: List[SingleRunResult],
    runs_b: List[SingleRunResult],
    label_a: str,
    label_b: str,
) -> Dict:
    """Compute Cohen's d and approximate power for wall_clock comparison."""
    real_a = [r for r in runs_a if not r.skipped and r.tok_per_s > 0]
    real_b = [r for r in runs_b if not r.skipped and r.tok_per_s > 0]
    wall_a = [r.wall_clock_s for r in real_a]
    wall_b = [r.wall_clock_s for r in real_b]

    d = cohens_d(wall_a, wall_b)
    n_min = min(len(wall_a), len(wall_b))
    power = approximate_power(d, n_min) if n_min >= 2 else 0.0

    return {
        "label_a": label_a,
        "label_b": label_b,
        "cohens_d": d,
        "power": power,
        "n_a": len(wall_a),
        "n_b": len(wall_b),
        "mean_a": statistics.mean(wall_a) if wall_a else 0.0,
        "mean_b": statistics.mean(wall_b) if wall_b else 0.0,
    }


def compute_orthogonality_ratio(
    apploop_wall: float,
    sig_wall: float,
    apploop_mtp_wall: float,
    sig_mtp_wall: float,
) -> Dict:
    """Compute orthogonality ratio ρ = S_SIG+MTP / (S_SIG × S_MTP).

    Returns a dict with S_SIG, S_MTP, S_SIG+MTP, rho, and validity notes.
    ρ is only meaningful when S_SIG > 1 and S_MTP > 1.
    """
    result: Dict = {
        "apploop_wall_s": apploop_wall,
        "sig_wall_s": sig_wall,
        "apploop_mtp_wall_s": apploop_mtp_wall,
        "sig_mtp_wall_s": sig_mtp_wall,
    }

    if apploop_wall <= 0 or sig_wall <= 0 or apploop_mtp_wall <= 0 or sig_mtp_wall <= 0:
        result.update({"rho": None, "valid": False, "reason": "one_or_more_wall_clock_values_leq_zero"})
        return result

    s_sig = apploop_wall / sig_wall
    s_mtp = apploop_wall / apploop_mtp_wall
    s_sig_mtp = apploop_wall / sig_mtp_wall

    result["S_SIG"] = round(s_sig, 4)
    result["S_MTP"] = round(s_mtp, 4)
    result["S_SIG_MTP"] = round(s_sig_mtp, 4)

    if s_sig <= 1.0:
        result.update({
            "rho": None, "valid": False,
            "reason": f"S_SIG = {s_sig:.3f} <= 1.0; SIG is not accelerating, rho not interpretable",
        })
        return result

    if s_mtp <= 1.0:
        result.update({
            "rho": None, "valid": False,
            "reason": f"S_MTP = {s_mtp:.3f} <= 1.0; MTP is harmful under AppLoop, rho > 1.0 would be a mathematical artifact, not super-multiplicative synergy",
        })
        return result

    rho = s_sig_mtp / (s_sig * s_mtp)
    result["rho"] = round(rho, 4)
    result["valid"] = True
    result["pass_085"] = rho >= 0.85
    result["reason"] = (
        f"rho = {rho:.3f} >= 0.85: PASS (near-multiplicative composition)"
        if rho >= 0.85
        else f"rho = {rho:.3f} < 0.85: sub-multiplicative (shared overhead or interference)"
    )
    return result


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
            ci_low_wall_clock_s=0.0,
            ci_high_wall_clock_s=0.0,
            ci_low_tok_per_s=0.0,
            ci_high_tok_per_s=0.0,
            skip_reason=(runs[0].skip_reason if runs and runs[0].skipped else ""),
        )
    wall_vals = [r.wall_clock_s for r in real]
    tps_vals = [r.tok_per_s for r in real]
    _, wall_ci_lo, wall_ci_hi = bootstrap_ci(wall_vals)
    _, tps_ci_lo, tps_ci_hi = bootstrap_ci(tps_vals)
    return ModeContextSummary(
        mode=runs[0].mode,
        context_length=runs[0].context_length,
        n_runs=len(runs),
        mean_tok_per_s=statistics.mean(r.tok_per_s for r in real),
        std_tok_per_s=safe_std(tps_vals),
        mean_wall_clock_s=statistics.mean(r.wall_clock_s for r in real),
        std_wall_clock_s=safe_std(wall_vals),
        mean_gen_tokens=statistics.mean(r.gen_tokens for r in real),
        mean_acceptance_rate=statistics.mean(r.acceptance_rate for r in real),
        mean_draft_proposed=statistics.mean(r.draft_proposed for r in real),
        mean_draft_accepted=statistics.mean(r.draft_accepted for r in real),
        skipped=False,
        ci_low_wall_clock_s=wall_ci_lo,
        ci_high_wall_clock_s=wall_ci_hi,
        ci_low_tok_per_s=tps_ci_lo,
        ci_high_tok_per_s=tps_ci_hi,
    )


# ── Placeholder mode (no llama-server) ─────────────────────────────

def build_placeholder_runs(
    mode_specs: List[Tuple[str, str, int]],
    context_lengths: List[int],
    n_runs: int,
    reason: str,
) -> List[SingleRunResult]:
    out: List[SingleRunResult] = []
    for mode_label, spec_type, draft_n in mode_specs:
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


def run_kitchen_condition(
    server_bin: Path,
    model_path: str,
    condition: str,  # "apploop", "sig", "apploop_mtp", "sig_mtp"
    spec_type: str,
    spec_draft_n_max: int,
    n_runs: int,
    base_port: int,
    max_new_tokens: int = 60,
) -> List[KitchenSessionResult]:
    """Run EdgeAgent-Kitchen benchmark for one condition across n_runs.

    SIG simulation: Send full message history each turn; llama-server's
    prompt cache will reuse KV-cache from previous turns (only new tokens
    are evaluated). This approximates SIG's KV-cache continuity.

    AppLoop simulation: Restart the server between each turn, forcing
    full re-encoding of the entire conversation each time.
    """
    all_sessions: List[KitchenSessionResult] = []

    for run_id in range(n_runs):
        print(f"\n  [kitchen condition={condition} run={run_id}] starting...")
        is_sig = condition in ("sig", "sig_mtp")

        server = LlamaServerMTP(
            binary=server_bin,
            model_path=model_path,
            spec_type=spec_type,
            spec_draft_n_max=spec_draft_n_max,
            n_ctx=N_CTX,
            n_gpu_layers=N_GPU_LAYERS,
            port=base_port,
        )

        if not server.start():
            all_sessions.append(KitchenSessionResult(
                condition=condition,
                n_turns=len(KITCHEN_SCENARIO_TURNS),
                total_wall_clock_s=0.0,
                mean_tok_per_s=0.0,
                sum_gen_tokens=0,
                skipped=True,
                skip_reason="server_start_failed",
            ))
            continue

        history: List[Dict[str, str]] = [
            {"role": "system", "content": KITCHEN_SYSTEM_PROMPT}
        ]
        turn_results: List[KitchenTurnResult] = []
        t_session0 = time.time()
        sum_gen = 0
        tok_s_acc: List[float] = []

        try:
            for tid, (user_query, tool_name, tool_response) in enumerate(KITCHEN_SCENARIO_TURNS):
                # Build the prompt for this turn
                if is_sig:
                    # SIG: append to existing history (prompt cache reuse)
                    history.append({"role": "user", "content": user_query})
                else:
                    # AppLoop: rebuild full history from scratch each turn
                    history = [
                        {"role": "system", "content": KITCHEN_SYSTEM_PROMPT}
                    ]
                    # Include all prior turns as context
                    for prev_tid in range(tid):
                        prev_query, prev_tool, prev_resp = KITCHEN_SCENARIO_TURNS[prev_tid]
                        history.append({"role": "user", "content": prev_query})
                        history.append({"role": "assistant",
                                        "content": f"[Tool: {prev_tool}] {prev_resp}"})
                    history.append({"role": "user", "content": user_query})

                try:
                    res = server.chat_streaming(
                        history, max_tokens=max_new_tokens, temperature=TEMPERATURE
                    )
                    if res.get("error") or res.get("tokens_predicted", 0) == 0:
                        res = server.chat(
                            history, max_tokens=max_new_tokens, temperature=TEMPERATURE
                        )

                    ar = _normalize_ar(res.get("acceptance_rate", 0.0) or 0.0)
                    if ar == 0 and res.get("draft_accepted", 0) and res.get("draft_proposed", 0):
                        ar = res["draft_accepted"] / max(res.get("draft_proposed", 1), 1)
                        ar = _normalize_ar(ar)

                    # Extract prompt eval time from timings if available
                    prompt_eval_s = 0.0
                    raw = res.get("raw", {})
                    if isinstance(raw, dict):
                        timings = raw.get("timings", {})
                        if isinstance(timings, dict):
                            prompt_eval_s = float(timings.get("prompt_n", 0)) / max(
                                float(timings.get("prompt_per_second", 1)), 1.0
                            ) if timings.get("prompt_per_second") else 0.0
                            # Alternative: prompt_eval_ms from timings
                            if "prompt_eval_ms" in timings:
                                prompt_eval_s = float(timings["prompt_eval_ms"]) / 1000.0

                    assistant_text = res.get("text", "")
                    if is_sig:
                        history.append({"role": "assistant", "content": assistant_text})

                    turn_results.append(KitchenTurnResult(
                        turn_id=tid,
                        condition=condition,
                        user_query=user_query,
                        tool_name=tool_name,
                        prompt_tokens=res["tokens_evaluated"],
                        gen_tokens=res["tokens_predicted"],
                        wall_clock_s=res["wall_clock_s"],
                        tok_per_s=res["tok_per_s"],
                        prompt_eval_s=prompt_eval_s,
                        acceptance_rate=ar,
                        draft_proposed=int(res.get("draft_proposed", 0) or 0),
                        draft_accepted=int(res.get("draft_accepted", 0) or 0),
                    ))
                    sum_gen += res["tokens_predicted"]
                    if res["tok_per_s"] > 0:
                        tok_s_acc.append(res["tok_per_s"])

                except Exception as e:
                    turn_results.append(KitchenTurnResult(
                        turn_id=tid,
                        condition=condition,
                        user_query=user_query,
                        tool_name=tool_name,
                        prompt_tokens=0,
                        gen_tokens=0,
                        wall_clock_s=0.0,
                        tok_per_s=0.0,
                        prompt_eval_s=0.0,
                        acceptance_rate=0.0,
                        draft_proposed=0,
                        draft_accepted=0,
                    ))
                    print(f"    [turn={tid}] ERROR: {e}")

                # For AppLoop: restart server each turn to force full re-encoding
                if not is_sig and tid < len(KITCHEN_SCENARIO_TURNS) - 1:
                    server.stop()
                    server = LlamaServerMTP(
                        binary=server_bin,
                        model_path=model_path,
                        spec_type=spec_type,
                        spec_draft_n_max=spec_draft_n_max,
                        n_ctx=N_CTX,
                        n_gpu_layers=N_GPU_LAYERS,
                        port=base_port,
                    )
                    if not server.start():
                        print(f"    [turn={tid}] Server restart failed")
                        break

        finally:
            server.stop()

        total_wall = time.time() - t_session0
        mean_tps = statistics.mean(tok_s_acc) if tok_s_acc else 0.0
        all_sessions.append(KitchenSessionResult(
            condition=condition,
            n_turns=len(KITCHEN_SCENARIO_TURNS),
            total_wall_clock_s=total_wall,
            mean_tok_per_s=mean_tps,
            sum_gen_tokens=sum_gen,
            turns=turn_results,
        ))
        print(f"  [kitchen condition={condition} run={run_id}] "
              f"total={total_wall:.2f}s, gen={sum_gen} tok, {mean_tps:.1f} tok/s")

    return all_sessions


# ── JSON output assembly ───────────────────────────────────────────

def build_output(
    model_path: str,
    n_runs: int,
    context_lengths: List[int],
    mode_specs: List[Tuple[str, str, int]],
    single_runs: List[SingleRunResult],
    chat_results: List[ChatEndToEndResult],
    llama_server_available: bool,
    llama_server_path: Optional[Path],
    model_available: bool,
    kitchen_results: Optional[Dict[str, List[KitchenSessionResult]]] = None,
) -> Dict:
    # 1. Per-cell detail
    per_cell_detail: Dict[str, Dict[str, List[Dict]]] = {}
    for mode_label, _, _ in mode_specs:
        per_cell_detail[mode_label] = {}
        for ctx in context_lengths:
            cell_runs = [
                asdict(r) for r in single_runs
                if r.mode == mode_label and r.context_length == ctx
            ]
            per_cell_detail[mode_label][str(ctx)] = cell_runs

    # 2. Per-cell summary (mean +/- std)
    per_cell_summary: Dict[str, Dict[str, Dict]] = {}
    for mode_label, _, _ in mode_specs:
        per_cell_summary[mode_label] = {}
        for ctx in context_lengths:
            cell = [r for r in single_runs if r.mode == mode_label and r.context_length == ctx]
            s = summarize_runs(cell)
            per_cell_summary[mode_label][str(ctx)] = asdict(s)

    # 3. Mode-level aggregate (pooling across context lengths)
    mode_aggregate: Dict[str, Dict] = {}
    for mode_label, _, _ in mode_specs:
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

    # 4. SIG-only vs SIG+MTP end-to-end speedup
    sig_only = next((c for c in chat_results if c.mode == "sig_only"), None)
    sig_mtp = next((c for c in chat_results if c.mode == "sig_mtp"), None)
    speedup: Dict[str, float] = {}
    if sig_only and sig_mtp and not sig_only.skipped and not sig_mtp.skipped:
        if sig_mtp.total_wall_clock_s > 0:
            speedup["sig_only_total_wall_s"] = sig_only.total_wall_clock_s
            speedup["sig_mtp_total_wall_s"] = sig_mtp.total_wall_clock_s
            speedup["speedup_x"] = sig_only.total_wall_clock_s / sig_mtp.total_wall_clock_s
            speedup["mean_tok_per_s_sig_only"] = sig_only.mean_tok_per_s
            speedup["mean_tok_per_s_sig_mtp"] = sig_mtp.mean_tok_per_s
        else:
            speedup = {"reason": "sig_mtp_total_wall_s == 0"}
    else:
        speedup = {
            "reason": "one_or_both_modes_skipped",
            "sig_only_skipped": sig_only.skipped if sig_only else None,
            "sig_mtp_skipped": sig_mtp.skipped if sig_mtp else None,
        }

    # 5. Statistical analysis (bootstrap CI per mode, pairwise comparison)
    statistical_analysis: Dict[str, Dict] = {}
    for mode_label, _, _ in mode_specs:
        mode_runs = [r for r in single_runs if r.mode == mode_label]
        statistical_analysis[mode_label] = compute_statistical_analysis(mode_runs)

    # Pairwise comparison: sig_only vs sig_mtp for end-to-end chat
    sig_only_runs = [r for r in single_runs if r.mode == "sig_only"]
    sig_mtp_runs = [r for r in single_runs if r.mode == "sig_mtp"]
    pairwise = compute_pairwise_comparison(sig_only_runs, sig_mtp_runs, "sig_only", "sig_mtp")

    # 6. Kitchen benchmark results
    kitchen_section: Dict = {}
    if kitchen_results:
        kitchen_section["conditions"] = {}
        for cond, sessions in kitchen_results.items():
            real_sessions = [s for s in sessions if not s.skipped]
            if real_sessions:
                walls = [s.total_wall_clock_s for s in real_sessions]
                tps = [s.mean_tok_per_s for s in real_sessions if s.mean_tok_per_s > 0]
                gens = [s.sum_gen_tokens for s in real_sessions]
                kitchen_section["conditions"][cond] = {
                    "n_runs": len(sessions),
                    "n_valid": len(real_sessions),
                    "mean_wall_clock_s": statistics.mean(walls) if walls else 0.0,
                    "std_wall_clock_s": safe_std(walls),
                    "mean_tok_per_s": statistics.mean(tps) if tps else 0.0,
                    "mean_gen_tokens": statistics.mean(gens) if gens else 0.0,
                    "sessions": [asdict(s) for s in sessions],
                }
            else:
                kitchen_section["conditions"][cond] = {
                    "n_runs": len(sessions),
                    "n_valid": 0,
                    "sessions": [asdict(s) for s in sessions],
                }

        # Orthogonality analysis from kitchen results
        ortho = {"source": "kitchen_benchmark"}
        cond_means = {}
        for cond, sessions in kitchen_results.items():
            real = [s for s in sessions if not s.skipped and s.total_wall_clock_s > 0]
            if real:
                cond_means[cond] = statistics.mean(s.total_wall_clock_s for s in real)

        if all(c in cond_means for c in ["apploop", "sig", "apploop_mtp", "sig_mtp"]):
            ortho.update(compute_orthogonality_ratio(
                cond_means["apploop"], cond_means["sig"],
                cond_means["apploop_mtp"], cond_means["sig_mtp"],
            ))
        else:
            ortho["rho"] = None
            ortho["valid"] = False
            ortho["reason"] = "not all 4 conditions have valid data"
        kitchen_section["orthogonality_analysis"] = ortho

    # 7. Orthogonality from chat results (if kitchen not available)
    chat_ortho: Dict = {"source": "chat_end_to_end"}
    if not kitchen_results:
        sig_only = next((c for c in chat_results if c.mode == "sig_only"), None)
        sig_mtp = next((c for c in chat_results if c.mode == "sig_mtp"), None)
        # Chat mode only has sig_only and sig_mtp, not all 4 conditions
        # So we can't compute full orthogonality, but we can compute SIG+MTP speedup
        if sig_only and sig_mtp and not sig_only.skipped and not sig_mtp.skipped:
            if sig_mtp.total_wall_clock_s > 0:
                chat_ortho["sig_only_wall_s"] = sig_only.total_wall_clock_s
                chat_ortho["sig_mtp_wall_s"] = sig_mtp.total_wall_clock_s
                chat_ortho["speedup_x"] = sig_only.total_wall_clock_s / sig_mtp.total_wall_clock_s
                chat_ortho["note"] = "Full orthogonality ratio requires 4 conditions (AppLoop, SIG, AppLoop+MTP, SIG+MTP). Use --task kitchen for complete analysis."
        chat_ortho["rho"] = None
        chat_ortho["valid"] = False
        chat_ortho["reason"] = "chat_mode_only_has_2_conditions"

    return {
        "metadata": {
            "model_path": model_path,
            "model_available": model_available,
            "llama_server_available": llama_server_available,
            "llama_server_path": str(llama_server_path) if llama_server_path else None,
            "n_runs_per_cell": n_runs,
            "context_lengths": list(context_lengths),
            "mode_specs": [
                {"mode": m, "spec_type": st, "spec_draft_n_max": dn}
                for (m, st, dn) in mode_specs
            ],
            "n_ctx": N_CTX,
            "n_gpu_layers": N_GPU_LAYERS,
            "temperature": TEMPERATURE,
            "max_new_tokens": MAX_NEW_TOKENS,
            "skipped_due_to_no_llama_server": (
                not llama_server_available or not model_available
            ),
            "skip_reason": (
                "llama_server binary not found"
                if not llama_server_available
                else "model file not found"
                if not model_available
                else ""
            ),
        },
        "per_cell_detail": per_cell_detail,
        "per_cell_summary": per_cell_summary,
        "mode_aggregate": mode_aggregate,
        "end_to_end_chat": {
            "turns_script": [
                {"turn_id": tid, "prompt": p, "max_new_tokens": m}
                for tid, (p, m) in enumerate(SIG_CHAT_TURNS)
            ],
            "results": [asdict(c) for c in chat_results],
            "sig_only_vs_sig_mtp_speedup": speedup,
        },
        "statistical_analysis": {
            "per_mode": statistical_analysis,
            "pairwise_sig_only_vs_sig_mtp": pairwise,
        },
        "kitchen_benchmark": kitchen_section,
        "orthogonality_analysis": kitchen_section.get("orthogonality_analysis", {}) if kitchen_results else chat_ortho,
    }


# ── Main orchestration ─────────────────────────────────────────────

def parse_int_list(spec: str) -> List[int]:
    return [int(x.strip()) for x in spec.split(",") if x.strip()]


def main():
    parser = argparse.ArgumentParser(
        description="P5 native MTP full sweep v2: 4 modes x 4 ctx x n runs + SIG vs SIG+MTP"
    )
    parser.add_argument(
        "--model", default=DEFAULT_MODEL,
        help="Path to GGUF model (default: %(default)s)",
    )
    parser.add_argument(
        "--mtp-model", default=MTP_MODEL,
        help="Path to MTP-enabled GGUF (fallback to --model if not found)",
    )
    parser.add_argument("--n-runs", type=int, default=DEFAULT_N_RUNS, help="Runs per cell")
    parser.add_argument(
        "--context-lengths", type=str, default="30,500,1000,2000",
        help="Comma-separated target prompt token counts",
    )
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    parser.add_argument(
        "--output", type=str, default="data/exp_mtp_full_v2_results.json",
    )
    parser.add_argument(
        "--no-chat", action="store_true",
        help="Skip the end-to-end multi-turn SIG vs SIG+MTP comparison",
    )
    parser.add_argument(
        "--skip-sweep", action="store_true",
        help="Skip the per-cell sweep (only run the end-to-end chat)",
    )
    parser.add_argument(
        "--mode", type=str, default="all",
        choices=["all", "no_mtp", "mtp_n1", "mtp_n2", "mtp_n3"],
        help="Restrict to a single mode (default: all 4 modes)",
    )
    parser.add_argument(
        "--task", type=str, default="all",
        choices=["all", "sweep", "chat", "kitchen"],
        help="Task to run: all (sweep+chat), sweep, chat, or kitchen (EdgeAgent-Kitchen 4-condition)",
    )
    args = parser.parse_args()

    context_lengths = parse_int_list(args.context_lengths)

    full_mode_specs: List[Tuple[str, str, int]] = [
        ("no_mtp", "none", 0),
        ("mtp_n1", "draft-mtp", 1),
        ("mtp_n2", "draft-mtp", 2),
        ("mtp_n3", "draft-mtp", 3),
    ]
    if args.mode == "all":
        mode_specs = full_mode_specs
    else:
        mode_specs = [m for m in full_mode_specs if m[0] == args.mode]

    # Resolve model: prefer MTP-enabled file, fall back to standard.
    model_path = args.mtp_model if Path(args.mtp_model).exists() else args.model
    model_available = detect_model(model_path)
    llama_available, server_bin = detect_llama_server()

    print("=" * 70)
    print("P5 Native MTP Full Sweep v2")
    print("=" * 70)
    print(f"  model_path           = {model_path}")
    print(f"  model_available      = {model_available}")
    print(f"  llama_server_avail   = {llama_available}")
    if server_bin:
        print(f"  llama_server_path    = {server_bin}")
    print(f"  n_runs               = {args.n_runs}")
    print(f"  context_lengths      = {context_lengths}")
    print(f"  max_new_tokens       = {args.max_new_tokens}")
    print(f"  mode_specs           = {[m[0] for m in mode_specs]}")

    can_run = llama_available and model_available

    if not can_run:
        reason = (
            "llama-server binary not found in PATH or at "
            f"{LLAMA_CPP_BIN_DIR}"
            if not llama_available
            else f"model file not found: {model_path}"
        )
        print(f"\n⚠ Cannot run: {reason}")
        print("  Emitting placeholder JSON with the full cartesian product of cells.")
        single_runs = build_placeholder_runs(
            mode_specs, context_lengths, args.n_runs,
            reason="skipped_due_to_no_llama_server",
        )
        chat_results: List[ChatEndToEndResult] = []
        if not args.no_chat:
            chat_results = build_placeholder_chat(
                ["sig_only", "sig_mtp"], SIG_CHAT_TURNS,
                reason="skipped_due_to_no_llama_server",
            )
        kitchen_results: Optional[Dict[str, List[KitchenSessionResult]]] = None
    else:
        single_runs: List[SingleRunResult] = []
        if args.task in ("all", "sweep") and not args.skip_sweep:
            print("\n>>> Running per-cell sweep")
            single_runs = run_mode_sweep(
                server_bin=server_bin,
                model_path=model_path,
                mode_specs=mode_specs,
                context_lengths=context_lengths,
                n_runs=args.n_runs,
                base_port=args.port,
                max_new_tokens=args.max_new_tokens,
            )
        elif args.task not in ("all", "sweep"):
            print("\n>>> Skipping per-cell sweep (task != sweep/all)")
            single_runs = build_placeholder_runs(
                mode_specs, context_lengths, args.n_runs,
                reason="skipped_by_task_selection",
            )
        else:
            print("\n>>> Skipping per-cell sweep (--skip-sweep)")
            single_runs = build_placeholder_runs(
                mode_specs, context_lengths, args.n_runs,
                reason="skipped_by_user",
            )

        chat_results: List[ChatEndToEndResult] = []
        if args.task in ("all", "chat") and not args.no_chat:
            print("\n>>> Running end-to-end SIG-only vs SIG+MTP chat")
            chat_results = []
            print("\n[chat A] SIG-only (no MTP)...")
            chat_results.append(run_chat_endtoend(
                server_bin=server_bin,
                model_path=model_path,
                mode_label="sig_only",
                spec_type="none",
                spec_draft_n_max=0,
                turns=SIG_CHAT_TURNS,
                base_port=args.port,
            ))
            print("\n[chat B] SIG + MTP (draft-n-max=2)...")
            chat_results.append(run_chat_endtoend(
                server_bin=server_bin,
                model_path=model_path,
                mode_label="sig_mtp",
                spec_type="draft-mtp",
                spec_draft_n_max=2,
                turns=SIG_CHAT_TURNS,
                base_port=args.port,
            ))

        kitchen_results: Optional[Dict[str, List[KitchenSessionResult]]] = None
        if args.task in ("all", "kitchen"):
            print("\n>>> Running EdgeAgent-Kitchen 4-condition benchmark")
            kitchen_results = {}
            kitchen_conditions = [
                ("apploop", "none", 0),
                ("sig", "none", 0),
                ("apploop_mtp", "draft-mtp", 2),
                ("sig_mtp", "draft-mtp", 2),
            ]
            for cond_label, spec_type, draft_n in kitchen_conditions:
                print(f"\n[Kitchen] condition={cond_label} (spec_type={spec_type}, draft_n={draft_n})")
                kitchen_results[cond_label] = run_kitchen_condition(
                    server_bin=server_bin,
                    model_path=model_path,
                    condition=cond_label,
                    spec_type=spec_type,
                    spec_draft_n_max=draft_n,
                    n_runs=args.n_runs,
                    base_port=args.port,
                )

    out_doc = build_output(
        model_path=model_path,
        n_runs=args.n_runs,
        context_lengths=context_lengths,
        mode_specs=mode_specs,
        single_runs=single_runs,
        chat_results=chat_results,
        llama_server_available=llama_available,
        llama_server_path=server_bin,
        model_available=model_available,
        kitchen_results=kitchen_results,
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_doc, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: {out_path}")
    print(f"  total cells: {len(mode_specs) * len(context_lengths) * args.n_runs}")
    print(f"  skipped:     {sum(1 for r in single_runs if r.skipped)}")
    print(f"  completed:   {sum(1 for r in single_runs if not r.skipped)}")
    if not out_doc["end_to_end_chat"]["sig_only_vs_sig_mtp_speedup"].get("reason"):
        sp = out_doc["end_to_end_chat"]["sig_only_vs_sig_mtp_speedup"]
        print(
            f"  SIG vs SIG+MTP:  sig_only={sp['sig_only_total_wall_s']:.2f}s  "
            f"sig_mtp={sp['sig_mtp_total_wall_s']:.2f}s  speedup={sp['speedup_x']:.2f}x"
        )

    # Print kitchen results summary
    if kitchen_results:
        print("\n  Kitchen Benchmark Summary:")
        for cond, sessions in kitchen_results.items():
            real = [s for s in sessions if not s.skipped]
            if real:
                mean_wall = statistics.mean(s.total_wall_clock_s for s in real)
                print(f"    {cond}: {mean_wall:.2f}s (n={len(real)})")
        # Print orthogonality if available
        ortho = out_doc.get("orthogonality_analysis", {})
        if ortho.get("valid"):
            print(f"    ρ = {ortho['rho']:.3f} ({ortho.get('reason', '')})")
        elif ortho.get("reason"):
            print(f"    ρ: {ortho['reason']}")


if __name__ == "__main__":
    main()
