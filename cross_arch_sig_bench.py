#!/usr/bin/env python3
"""
Cross-Architecture SIG Validation Infrastructure
================================================

Reuses the R6 deep-chain experiment from co_benchmark.py to validate
SIG prefill savings on non-Qwen model families (Llama-3.2, Gemma-2, ...).

Tasks
-----
  r6         : 30-tool chain, 15% failure injection. Three modes: SIG, AppLoop,
               AppLoop-PC. Default N=10 paired runs.
  specdec    : SpecDec compatibility test for kv_cache_seq_rm partial deletion.
  arch       : Architecture detection only (GQA ratio, SWA, hybrid attention).
  all        : run all of the above.

Usage
-----
  python cross_arch_sig_bench.py --model models/Llama-3.2-1B-Instruct-Q4_K_M.gguf --task r6 --runs 10
  python cross_arch_sig_bench.py --model models/gemma-2-2b-it-Q4_K_M.gguf --task r6 --runs 10
  python cross_arch_sig_bench.py --model models/Llama-3.2-1B-Instruct-Q4_K_M.gguf --task specdec
  python cross_arch_sig_bench.py --model models/Qwen3.5-4B-Q4_K_M.gguf --task all --runs 5

Outputs JSON to ``data/cross_arch/{model_name}_r6.json`` and
``data/cross_arch/{model_name}_specdec_test.json``.

Notes
-----
* Reuses core/compiler.py (MeaningCompiler, PrefixCache) and
  core/injection.py (InjectionEngine) without modification. If SIG injection
  fails on a non-Qwen model, the failure is recorded but core/ is left intact.
* Architecture features are inferred from the model filename and GGUF
  metadata; nothing is hard-wired to a specific family.
"""

import os
import sys
import json
import time
import argparse
import random
import statistics
import traceback
from typing import Dict, List, Optional, Tuple

from llama_cpp import Llama

from core import (
    MeaningCompiler,
    ToolRegistry,
    GPUMonitor,
    SYSTEM_PROMPT,
)
from core.compiler import PrefixCache


# ---------------------------------------------------------------------------
# Architecture detection
# ---------------------------------------------------------------------------

# Filename-based feature hints. GGUF metadata is the source of truth when
# available, but filenames give a quick first cut.
_ARCH_HINTS = [
    ("nemotron", {
        "family": "nemotron",
        "kind": "hybrid_mamba_attention",
        "gqa": None,
        "swa": False,
        "notes": "Hybrid Mamba/SSM + attention. State-space recurrence compresses history "
                 "into a fixed-size hidden state; SIG prefill saving semantics may diverge "
                 "from pure-Transformer baselines.",
    }),
    ("gemma-4", {
        "family": "gemma-4",
        "arch_type": "gemma4",
        "kind": "dense_attention_swa_shared",
        "gqa": 4,
        "swa": True,
        "shared_kv_layers": True,
        "notes": "gemma4 architecture: SWA layers interleaved with global attention, plus "
                 "shared_kv_layers (a subset of layers share KV cache). GeGLU + pre-norm + GQA. "
                 "Partial-deletion semantics may diverge across SWA window boundaries and "
                 "shared layers — SpecDec compatibility should be cross-checked with the actual "
                 "shared_kv_layers metadata value.",
    }),
    ("gemma4", {
        "family": "gemma-4",
        "arch_type": "gemma4",
        "kind": "dense_attention_swa_shared",
        "gqa": 4,
        "swa": True,
        "shared_kv_layers": True,
        "notes": "gemma4 architecture (alt. filename spelling, no hyphen): SWA + "
                 "shared_kv_layers. GeGLU + pre-norm + GQA.",
    }),
    ("gemma2", {
        "family": "gemma-2",
        "arch_type": "gemma2",
        "kind": "dense_attention_swa",
        "gqa": 2,
        "swa": True,
        "shared_kv_layers": False,
        "notes": "Sliding-window attention (SWA) layers intermixed with global attention. "
                 "SIG injection must respect the local window or be confined to global layers.",
    }),
    ("gemma", {
        "family": "gemma-1/2",
        "kind": "dense_attention",
        "gqa": 1,
        "swa": False,
        "notes": "GeGLU + pre-norm. Pure dense attention in the 1.x line.",
    }),
    ("llama-3.2", {
        "family": "llama-3.2",
        "kind": "gqa_dense",
        "gqa": 4,
        "swa": False,
        "notes": "RoPE, SwiGLU, GQA at 1B/3B. Standard Transformer with grouped KV.",
    }),
    ("llama-3.1", {
        "family": "llama-3.1",
        "kind": "gqa_dense",
        "gqa": 8,
        "swa": False,
        "notes": "RoPE, SwiGLU, GQA at 8B+.",
    }),
    ("llama-3", {
        "family": "llama-3",
        "kind": "gqa_dense",
        "gqa": 4,
        "swa": False,
        "notes": "RoPE, SwiGLU, GQA.",
    }),
    ("llama", {
        "family": "llama",
        "kind": "gqa_dense",
        "gqa": 4,
        "swa": False,
        "notes": "Generic Llama-family: RoPE, SwiGLU, likely GQA.",
    }),
    ("qwen3.5", {
        "family": "qwen3.5",
        "kind": "qwen_hybrid",
        "gqa": 4,
        "swa": False,
        "notes": "Qwen3.5 — the reference architecture the SIG pipeline was originally "
                 "tuned against. SpecDec compatibility is the known baseline.",
    }),
    ("qwen3", {
        "family": "qwen3",
        "kind": "dense_attention",
        "gqa": 7,
        "swa": False,
        "notes": "Qwen3 dense Transformer with GQA.",
    }),
    ("qwen2.5", {
        "family": "qwen2.5",
        "kind": "dense_attention",
        "gqa": 7,
        "swa": False,
        "notes": "Qwen2.5 dense Transformer with GQA.",
    }),
    ("qwen2", {
        "family": "qwen2",
        "kind": "dense_attention",
        "gqa": 7,
        "swa": False,
        "notes": "Qwen2 dense Transformer with GQA.",
    }),
    ("qwen", {
        "family": "qwen",
        "kind": "dense_attention",
        "gqa": 7,
        "swa": False,
        "notes": "Qwen-family default.",
    }),
    ("phi-3.5", {
        "family": "phi-3.5",
        "kind": "dense_attention",
        "gqa": 4,
        "swa": False,
        "notes": "Phi-3.5 dense attention. Closest cousin to the original SIG baseline.",
    }),
    ("phi-3", {
        "family": "phi-3",
        "kind": "dense_attention",
        "gqa": 4,
        "swa": False,
        "notes": "Phi-3 dense attention.",
    }),
    ("phi", {
        "family": "phi",
        "kind": "dense_attention",
        "gqa": 4,
        "swa": False,
        "notes": "Generic Phi family.",
    }),
    ("mistral", {
        "family": "mistral",
        "kind": "swa_dense",
        "gqa": 4,
        "swa": True,
        "notes": "Mistral SWA (sliding window attention) + GQA. SIG may need window-aware logic.",
    }),
]


def detect_architecture(model_path: str) -> Dict:
    """Infer architecture features from filename and GGUF metadata.

    Returns a dict with keys: family, arch_type, kind, gqa, swa, shared_kv_layers,
    is_draft_model, requires_target_arch, n_head, n_kv_head, block_count,
    embedding_length, feed_forward_length, head_count, head_count_kv,
    context_length, sliding_window_pattern, notes, source.
    """
    name = os.path.basename(model_path).lower()
    features = {
        "family": "unknown",
        "arch_type": "unknown",
        "kind": "unknown",
        "gqa": None,
        "swa": False,
        "shared_kv_layers": False,
        "is_draft_model": False,
        "requires_target_arch": None,
        "n_head": None,
        "n_kv_head": None,
        "block_count": None,
        "embedding_length": None,
        "feed_forward_length": None,
        "head_count": None,
        "head_count_kv": None,
        "context_length": None,
        "sliding_window_pattern": None,
        "notes": "Could not infer architecture from filename.",
        "source": "filename_fallback",
    }

    for needle, hint in _ARCH_HINTS:
        if needle in name:
            features.update(hint)
            features["source"] = "filename_hint"
            break

    # If arch_type was not set by the hint, mirror family so downstream consumers
    # always have a stable identifier.
    if features.get("arch_type") in (None, "unknown"):
        features["arch_type"] = features.get("family", "unknown")

    # Draft-model detection: filename suffix "-assistant" indicates an MTP
    # (multi-token prediction) draft model intended to pair with a target model
    # of the same architecture. The required target arch defaults to whatever
    # arch_type we just resolved.
    if "assistant" in name:
        features["is_draft_model"] = True
        if features.get("requires_target_arch") is None:
            features["requires_target_arch"] = features.get("arch_type", features.get("family"))

    # Try to refine with GGUF metadata (file size / magic + llama_cpp probe).
    # We avoid loading the whole model here; metadata only.
    try:
        with open(model_path, "rb") as f:
            magic = f.read(4)
        if magic != b"GGUF":
            features["notes"] += " [WARNING: file magic is not GGUF]"
    except OSError:
        features["notes"] += " [WARNING: file is unreadable]"

    if features["gqa"] is not None and features["n_head"] is None:
        features["gqa_ratio"] = features["gqa"]
    return features


def probe_model_metadata(llm: Llama, features: Dict) -> Dict:
    """Augment feature dict with values pulled from a loaded Llama instance.

    Best-effort — many fields are not exposed via llama-cpp-python and the
    keys differ across versions. Failures are swallowed.

    For Gemma-4 specifically, GGUF metadata keys use the ``gemma4.`` prefix
    (distinct from ``gemma2.`` for Gemma-2). The most important Gemma-4-only
    field is ``gemma4.shared_kv_layers``: a bitmask/list of layer indices
    that share KV cache. A non-zero value means partial KV deletion has to
    account for shared layers, otherwise the cache can drift between the
    shared and non-shared paths.
    """
    md: Dict = {}
    arch_type = features.get("arch_type") or features.get("family") or ""
    is_gemma4 = arch_type == "gemma4"

    try:
        ctx = getattr(llm, "n_ctx", None)
        if ctx is not None:
            md["context_length"] = int(ctx)
    except Exception:
        pass
    try:
        n_embd = getattr(llm, "n_embd", None)
        if n_embd is not None:
            md["embedding_length"] = int(n_embd)
            md["n_embd"] = int(n_embd)
    except Exception:
        pass
    try:
        n_layer = getattr(llm, "n_layer", None)
        if n_layer is not None:
            md["block_count"] = int(n_layer)
            md["n_layer"] = int(n_layer)
    except Exception:
        pass
    try:
        n_head = getattr(llm, "n_head", None)
        n_head_kv = getattr(llm, "n_head_kv", None)
        if n_head is not None:
            md["head_count"] = int(n_head)
            md["n_head"] = int(n_head)
        if n_head_kv is not None:
            md["head_count_kv"] = int(n_head_kv)
            md["n_kv_head"] = int(n_head_kv)
        if n_head and n_head_kv and int(n_head) > 0:
            ratio = int(n_head) // int(n_head_kv) if int(n_head_kv) > 0 else 1
            md["gqa_ratio"] = ratio
            md["is_gqa"] = ratio > 1
    except Exception:
        pass

    # Read the raw GGUF metadata dict if exposed by llama-cpp-python.
    raw_md = getattr(llm, "metadata", None) or {}
    if isinstance(raw_md, dict):
        # feed_forward_length: used by llama.cpp's n_ff computation.
        for key in ("feed_forward_length", "llama.feed_forward_length"):
            if key in raw_md and md.get("feed_forward_length") is None:
                try:
                    md["feed_forward_length"] = int(raw_md[key])
                    break
                except Exception:
                    pass

        # sliding_window_pattern (Gemma-4 only).  Gemma-2 uses
        # ``gemma2.attn.sliding_window``; Gemma-4 adds ``gemma4.sliding_window_pattern``
        # as a comma-separated list of layer indices that use SWA.
        for key in (
            "gemma4.sliding_window_pattern",
            "gemma2.attn.sliding_window",
            "attn.sliding_window",
        ):
            if key in raw_md and md.get("sliding_window_pattern") is None:
                md["sliding_window_pattern"] = raw_md[key]
                break

        # shared_kv_layers (Gemma-4 only).  Distinguish from any legacy
        # ``gemma2.shared_kv_layers`` field by preferring the ``gemma4.`` prefix.
        shared_kv_layers = None
        for key in (
            "gemma4.shared_kv_layers",
            "gemma2.shared_kv_layers",
            "shared_kv_layers",
        ):
            if key in raw_md and shared_kv_layers is None:
                shared_kv_layers = raw_md[key]
                md["shared_kv_layers_source_key"] = key
                break
        if shared_kv_layers is not None:
            md["shared_kv_layers_raw"] = shared_kv_layers
            # Best-effort: if it parses as an int/long, store the count; if it
            # is a string like "1,3,5" or "1;3;5", count the non-empty entries.
            try:
                md["shared_kv_layers_count"] = int(shared_kv_layers)
                if is_gemma4:
                    features["shared_kv_layers"] = True
            except Exception:
                if isinstance(shared_kv_layers, str):
                    parts = [p for p in shared_kv_layers.replace(";", ",").split(",") if p.strip()]
                    md["shared_kv_layers_count"] = len(parts)
                    if parts and is_gemma4:
                        features["shared_kv_layers"] = True
                else:
                    md["shared_kv_layers_count"] = None

    features.update({k: v for k, v in md.items() if v is not None and features.get(k) is None})
    return features


# ---------------------------------------------------------------------------
# SpecDec compatibility test
# ---------------------------------------------------------------------------

def test_specdec_compatibility(llm: Llama, model_path: str, features: Optional[Dict] = None) -> Dict:
    """Test whether kv_cache_seq_rm partial deletion works on this model.

    This is the gateway check for the SpecDec path: a model that rejects
    partial deletion cannot use the partial-draft-rejection optimisation.

    Three probes are run:
      * full_clear    — wipe the entire cache (should always succeed).
      * noop          — no-op on an empty cache range (should be a no-op).
      * partial_delete — fill with N tokens, then drop the first M.

    For Gemma-4 (and any SWA + shared_kv_layers model), the partial_delete
    probe is interpreted with extra caveats:
      * If ``shared_kv_layers`` > 0, partial deletion may not be safe for
        indices that fall on a layer that shares its KV cache with another
        layer — deleting the first M tokens in the shared layer also wipes
        history that the partner layer still references. The probe still
        runs unconditionally so the caller can see the raw outcome; the
        ``shared_kv_layers_partial_delete_applicable`` flag is False in
        that case to signal "do not use partial rejection on this model".
      * If the model uses SWA (e.g. Gemma-2 / Gemma-4 / Mistral), M tokens
        that span the SWA window boundary behave differently from M tokens
        that fit entirely inside the local window. The 50-token fill used
        here is large enough to span a typical 4k window, so a False
        partial_delete on these families is a real signal, not noise.
    """
    results: Dict = {
        "model": model_path,
        "full_clear": None,
        "noop": None,
        "partial_delete": None,
        "shared_kv_layers": None,
        "shared_kv_layers_partial_delete_applicable": True,
        "swa_partial_delete_applicable": True,
        "error": None,
        "compatible": False,
    }
    swa_flag = False
    shared_kv_layers_count = 0
    if features is not None:
        swa_flag = bool(features.get("swa"))
        if features.get("shared_kv_layers"):
            shared_kv_layers_count = int(features.get("shared_kv_layers_count") or 0) \
                if isinstance(features.get("shared_kv_layers_count"), int) else 1
        results["shared_kv_layers"] = bool(features.get("shared_kv_layers"))
        results["swa"] = swa_flag
    try:
        # Full clear should always succeed.
        results["full_clear"] = bool(llm.kv_cache_seq_rm(0, 0, -1))
    except Exception as e:
        results["full_clear"] = f"Error: {e!r}"
    try:
        # Noop on an empty cache should also be safe.
        results["noop"] = bool(llm.kv_cache_seq_rm(0, 100, -1))
    except Exception as e:
        results["noop"] = f"Error: {e!r}"

    try:
        # Fill the cache with 50 tokens, then try to delete the first 10.
        fill_ids = [1, 2, 3, 4, 5] * 10
        llm.eval(fill_ids)
        ok = llm.kv_cache_seq_rm(0, 10, -1)
        results["partial_delete"] = bool(ok)
    except Exception as e:
        results["partial_delete"] = f"Error: {e!r}"

    try:
        partial_ok = results.get("partial_delete") is True
        noop_ok = results.get("noop") is True
        full_ok = results.get("full_clear") is True
        # If the model has shared KV layers, partial_delete is not a safe
        # optimisation, so we report compatible=False even if the raw probe
        # succeeded. We also expose the raw result for the caller to inspect.
        if shared_kv_layers_count > 0:
            results["shared_kv_layers_partial_delete_applicable"] = False
        if swa_flag:
            # SWA models can still work with partial_delete as long as the
            # window is large enough; we leave the flag True but mark a note.
            results.setdefault(
                "swa_note",
                "SWA model: partial_delete spans SWA window boundary; interpret with care."
            )
        results["compatible"] = bool(partial_ok and noop_ok and full_ok and
                                      results["shared_kv_layers_partial_delete_applicable"])
    except Exception:
        pass
    return results


# ---------------------------------------------------------------------------
# R6 deep-chain experiment
# ---------------------------------------------------------------------------

CITIES_POOL = [
    "paris", "london", "rome", "berlin", "tokyo", "newyork",
    "sydney", "dubai", "mumbai", "moscow", "beijing", "cairo",
]


def _generate_tool_chain(n_tools: int) -> List[Tuple[str, Dict]]:
    """Build a deterministic 30-tool chain of mixed travel tools."""
    tools_out: List[Tuple[str, Dict]] = []
    for i in range(n_tools):
        city = CITIES_POOL[i % len(CITIES_POOL)]
        if i % 3 == 0:
            tools_out.append(("search_attractions", {"city": city}))
        elif i % 3 == 1:
            tools_out.append(("get_weather", {"city": city}))
        else:
            origin = CITIES_POOL[i % (len(CITIES_POOL) - 1)]
            dest = CITIES_POOL[(i + 1) % len(CITIES_POOL)]
            tools_out.append(("get_flight_info", {"origin": origin, "destination": dest}))
    return tools_out


def _run_mode_sig(compiler: MeaningCompiler, module: ToolRegistry, sys_ids: List[int],
                  tools: List[Tuple[str, Dict]], failed: set) -> float:
    """SIG: incremental token injection, no cache rebuild per step."""
    compiler.reset_cache()
    t_start = time.time()
    compiler.eval(sys_ids)
    accumulated: List[str] = []
    for step_i, (tool_name, tool_args) in enumerate(tools):
        result = module.execute(tool_name, tool_args)
        if step_i in failed:
            result = "[ERROR] Tool failed - retrying..."
            module.execute(tool_name, tool_args)
        city = list(tool_args.values())[0]
        tool_text = f"\n[Step {step_i+1}] {tool_name}({city}): {result}\n"
        accumulated.append(tool_text)
        t_ids = list(compiler.tokenize(tool_text, add_bos=False))
        compiler.eval(t_ids)
    return time.time() - t_start


def _run_mode_apploop(compiler: MeaningCompiler, module: ToolRegistry, sys_ids: List[int],
                      tools: List[Tuple[str, Dict]], failed: set) -> float:
    """AppLoop: rebuild the entire context cache from scratch at every step."""
    compiler.reset_cache()
    t_start = time.time()
    accumulated: List[str] = []
    compiler.eval(sys_ids)
    for step_i, (tool_name, tool_args) in enumerate(tools):
        result = module.execute(tool_name, tool_args)
        if step_i in failed:
            result = "[ERROR] Tool failed - retrying..."
            module.execute(tool_name, tool_args)
        city = list(tool_args.values())[0]
        tool_text = f"\n[Step {step_i+1}] {tool_name}({city}): {result}\n"
        accumulated.append(tool_text)
        context = "\n".join(p.strip() for p in accumulated if p.strip())
        full_text = SYSTEM_PROMPT + "\n\n" + context
        full_ids = list(compiler.tokenize(full_text, add_bos=False))
        compiler.rebuild_cache(full_ids)
    return time.time() - t_start


def _run_mode_apploop_pc(compiler: MeaningCompiler, module: ToolRegistry, sys_ids: List[int],
                         tools: List[Tuple[str, Dict]], failed: set) -> float:
    """AppLoop-PC: prefix-cache baseline (KV reuse for the system prompt)."""
    compiler.reset_cache()
    t_start = time.time()
    compiler.eval(sys_ids)
    pc = PrefixCache()
    pc.save(compiler, list(sys_ids))
    accumulated: List[str] = []
    for step_i, (tool_name, tool_args) in enumerate(tools):
        result = module.execute(tool_name, tool_args)
        if step_i in failed:
            result = "[ERROR] Tool failed - retrying..."
            module.execute(tool_name, tool_args)
        city = list(tool_args.values())[0]
        tool_text = f"\n[Step {step_i+1}] {tool_name}({city}): {result}\n"
        accumulated.append(tool_text)
        context = "\n".join(p.strip() for p in accumulated if p.strip())
        pc.restore(compiler)
        ctx_ids = list(compiler.tokenize("\n\n" + context, add_bos=False))
        compiler.eval(ctx_ids)
    return time.time() - t_start


def run_r6(args, compiler: MeaningCompiler, module: ToolRegistry, arch: Dict) -> Dict:
    """Drive the R6 deep-chain experiment across the three modes."""
    tool_depth = int(getattr(args, "tool_depth", 30))
    failure_rate = float(getattr(args, "failure_rate", 0.15))
    n_runs = int(getattr(args, "runs", 10))
    base_seed = 42

    tools = _generate_tool_chain(tool_depth)
    sys_ids = list(compiler.tokenize(f"{SYSTEM_PROMPT}\n\n", add_bos=False))

    print(f"\n{'='*80}")
    print(f"  R6 Deep-Chain Cross-Architecture Validation")
    print(f"  Family: {arch.get('family')} | Kind: {arch.get('kind')} | "
          f"GQA: {arch.get('gqa_ratio', arch.get('gqa', '?'))} | SWA: {arch.get('swa')}")
    print(f"  Tool depth: {tool_depth} | Failure rate: {failure_rate:.0%} | N={n_runs}")
    print(f"{'='*80}")

    sig_times: List[float] = []
    app_times: List[float] = []
    apppc_times: List[float] = []
    sig_failures: List[str] = []

    for run_i in range(n_runs):
        rng = random.Random(base_seed + run_i)
        failed = {idx for idx in range(tool_depth) if rng.random() < failure_rate}

        if run_i == 0:
            print(f"  Failed indices (run 0): {sorted(failed)[:8]}{'...' if len(failed) > 8 else ''} "
                  f"({len(failed)}/{tool_depth} failed)")

        # SIG (may raise on non-Qwen — capture and continue).
        try:
            sig_t = _run_mode_sig(compiler, module, sys_ids, tools, failed)
            sig_times.append(sig_t)
        except Exception as e:
            sig_failures.append(f"run {run_i}: {type(e).__name__}: {e}")

        # AppLoop
        try:
            app_t = _run_mode_apploop(compiler, module, sys_ids, tools, failed)
            app_times.append(app_t)
        except Exception as e:
            app_failures = sig_failures  # piggyback for visibility
            piggyback_msg = f"AppLoop failed at run {run_i}: {type(e).__name__}: {e}"
            print(f"  [WARN] {piggyback_msg}")
            app_times.append(float("nan"))

        # AppLoop-PC
        try:
            apppc_t = _run_mode_apploop_pc(compiler, module, sys_ids, tools, failed)
            apppc_times.append(apppc_t)
        except Exception as e:
            print(f"  [WARN] AppLoop-PC failed at run {run_i}: {type(e).__name__}: {e}")
            apppc_times.append(float("nan"))

    def _ms(vals: List[float]) -> Tuple[float, float, int]:
        clean = [v for v in vals if v == v]  # drop NaN
        if not clean:
            return float("nan"), float("nan"), 0
        m = statistics.mean(clean)
        s = statistics.stdev(clean) if len(clean) > 1 else 0.0
        return m, s, len(clean)

    sig_m, sig_s, sig_n = _ms(sig_times)
    app_m, app_s, app_n = _ms(app_times)
    apppc_m, apppc_s, apppc_n = _ms(apppc_times)

    def _safe_div(a: float, b: float) -> float:
        if a != a or b != b or b == 0:
            return float("nan")
        return a / b

    summary_table = {
        "SIG":      {"mean_s": sig_m,   "std_s": sig_s,   "n": sig_n,   "vs_sig": 1.0,  "vs_apploop": _safe_div(app_m, sig_m)},
        "AppLoop":  {"mean_s": app_m,   "std_s": app_s,   "n": app_n,   "vs_sig": _safe_div(sig_m, app_m), "vs_apploop": 1.0},
        "AppLoop-PC": {"mean_s": apppc_m, "std_s": apppc_s, "n": apppc_n,
                       "vs_sig": _safe_div(sig_m, apppc_m),
                       "vs_apploop": _safe_div(app_m, apppc_m)},
    }

    print(f"\n  {'Mode':<14} {'Wall-Clock (s)':<22} {'vs SIG':<10} {'vs AppLoop':<12}")
    print(f"  {'-'*14} {'-'*22} {'-'*10} {'-'*12}")
    for mode, row in summary_table.items():
        wc = f"{row['mean_s']:.3f}±{row['std_s']:.3f}" if row['mean_s'] == row['mean_s'] else "FAIL"
        vs_sig = f"{row['vs_sig']:.2f}x" if row['vs_sig'] == row['vs_sig'] else "n/a"
        vs_app = f"{row['vs_apploop']:.2f}x" if row['vs_apploop'] == row['vs_apploop'] else "n/a"
        print(f"  {mode:<14} {wc:<22} {vs_sig:<10} {vs_app:<12}")

    if sig_failures:
        print(f"\n  [NOTE] {len(sig_failures)} SIG run(s) raised exceptions on this model:")
        for msg in sig_failures[:3]:
            print(f"    - {msg}")
        if len(sig_failures) > 3:
            print(f"    ... and {len(sig_failures) - 3} more")

    payload = {
        "model_path": args.model,
        "architecture": arch,
        "task": "r6",
        "config": {
            "tool_depth": tool_depth,
            "failure_rate": failure_rate,
            "n_runs": n_runs,
            "base_seed": base_seed,
        },
        "summary": summary_table,
        "per_run": {
            "sig_times": sig_times,
            "apploop_times": app_times,
            "apploop_pc_times": apppc_times,
        },
        "sig_exceptions": sig_failures,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    return payload


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _safe_filename(model_path: str) -> str:
    base = os.path.basename(model_path)
    base = base.rsplit(".", 1)[0] if base.lower().endswith(".gguf") else base
    return "".join(c if (c.isalnum() or c in "-_") else "_" for c in base)


def _write_json(path: str, payload: Dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Cross-architecture SIG validation infrastructure")
    parser.add_argument("--model", type=str, required=True,
                        help="Path to a GGUF model file.")
    parser.add_argument("--draft-model", type=str, default=None,
                        help="Optional path to a paired draft (MTP) GGUF model. "
                             "When supplied, the draft model's architecture is also "
                             "detected, and the specdec JSON will report whether the "
                             "draft's requires_target_arch matches the main model's "
                             "arch_type (see target_arch_match).")
    parser.add_argument("--task", type=str, default="all",
                        choices=["r6", "specdec", "arch", "all"],
                        help="Which task(s) to run.")
    parser.add_argument("--runs", type=int, default=10,
                        help="Number of paired runs for the R6 task (default 10).")
    parser.add_argument("--tool-depth", type=int, default=30,
                        help="R6 tool-chain depth (default 30).")
    parser.add_argument("--failure-rate", type=float, default=0.15,
                        help="R6 simulated failure rate (default 0.15).")
    parser.add_argument("--n-ctx", type=int, default=8192)
    parser.add_argument("--n-threads", type=int, default=4)
    parser.add_argument("--n-gpu-layers", type=int, default=0)
    parser.add_argument("--output-dir", type=str, default="data/cross_arch",
                        help="Where to write JSON outputs (default data/cross_arch).")
    parser.add_argument("--quiet", action="store_true", help="Suppress per-step prints.")
    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"ERROR: model file not found: {args.model}", file=sys.stderr)
        return 2

    # 1) Architecture detection (filename + GGUF metadata peek).
    arch = detect_architecture(args.model)
    print(f"\n[ARCH] family={arch['family']}  kind={arch['kind']}  "
          f"gqa={arch.get('gqa', arch.get('gqa_ratio', '?'))}  swa={arch['swa']}")
    print(f"       notes: {arch['notes']}")

    # 2) Load the model and refine architecture features.
    print(f"\n[LOAD] {args.model}  ctx={args.n_ctx}  gpu_layers={args.n_gpu_layers}")
    t_load = time.time()
    try:
        compiler = MeaningCompiler(
            model_path=args.model,
            n_ctx=args.n_ctx,
            n_threads=args.n_threads,
            n_gpu_layers=args.n_gpu_layers,
        )
    except Exception as e:
        print(f"[FATAL] failed to load model: {e!r}", file=sys.stderr)
        return 3
    print(f"[LOAD] done in {time.time() - t_load:.1f}s")

    arch = probe_model_metadata(compiler.llm, arch)
    print(f"[ARCH refined] n_layer={arch.get('n_layer')}  n_head={arch.get('n_head')}  "
          f"n_kv_head={arch.get('n_kv_head')}  gqa_ratio={arch.get('gqa_ratio', arch.get('gqa'))}")

    module = ToolRegistry()
    gpu = GPUMonitor()
    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)
    fname = _safe_filename(args.model)

    # Optional paired draft model: filename-only detection (no GGUF probe, no
    # GPU load) so we can record pairing metadata cheaply. We only load the
    # draft model later if a future task needs it; the current specdec test
    # is driven off the main model's KV cache.
    draft_arch: Optional[Dict] = None
    if args.draft_model is not None:
        if not os.path.exists(args.draft_model):
            print(f"[WARN] draft model not found, ignoring: {args.draft_model}",
                  file=sys.stderr)
        else:
            draft_arch = detect_architecture(args.draft_model)
            print(f"[ARCH draft] family={draft_arch.get('family')}  "
                  f"arch_type={draft_arch.get('arch_type')}  "
                  f"is_draft_model={draft_arch.get('is_draft_model')}  "
                  f"requires_target_arch={draft_arch.get('requires_target_arch')}")

    arch_record = {
        "model_path": args.model,
        "architecture": arch,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    if draft_arch is not None:
        arch_record["draft_model_path"] = args.draft_model
        arch_record["draft_architecture"] = draft_arch
    _write_json(os.path.join(out_dir, f"{fname}_arch.json"), arch_record)

    try:
        # 3) SpecDec compatibility test.
        if args.task in ("specdec", "all"):
            print(f"\n[SPECDEC] testing kv_cache_seq_rm partial-delete on {arch['family']}")
            try:
                spec = test_specdec_compatibility(compiler.llm, args.model, features=arch)
            except Exception as e:
                spec = {"model": args.model, "error": f"{type(e).__name__}: {e}",
                        "compatible": False}
            # Pairing check: a draft model is only usable with this main model
            # if its requires_target_arch matches the main arch_type.
            target_arch_match: Optional[bool] = None
            if draft_arch is not None:
                target_arch_match = (
                    draft_arch.get("requires_target_arch") == arch.get("arch_type")
                )
            spec_record = {
                "model_path": args.model,
                "architecture": arch,
                "shared_kv_layers": arch.get("shared_kv_layers"),
                "target_arch_match": target_arch_match,
                **spec,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            if draft_arch is not None:
                spec_record["draft_model_path"] = args.draft_model
                spec_record["draft_architecture"] = draft_arch
                spec_record["draft_requires_target_arch"] = draft_arch.get("requires_target_arch")
            print(f"[SPECDEC] full_clear={spec.get('full_clear')}  "
                  f"noop={spec.get('noop')}  partial_delete={spec.get('partial_delete')}  "
                  f"compatible={spec.get('compatible')}"
                  + (f"  target_arch_match={target_arch_match}" if target_arch_match is not None else ""))
            _write_json(os.path.join(out_dir, f"{fname}_specdec_test.json"), spec_record)

        # 4) R6 deep-chain experiment.
        if args.task in ("r6", "all"):
            try:
                payload = run_r6(args, compiler, module, arch)
            except Exception as e:
                print(f"[FATAL] R6 crashed: {e!r}", file=sys.stderr)
                traceback.print_exc()
                payload = {
                    "model_path": args.model,
                    "architecture": arch,
                    "task": "r6",
                    "error": f"{type(e).__name__}: {e}",
                    "traceback": traceback.format_exc(),
                }
            _write_json(os.path.join(out_dir, f"{fname}_r6.json"), payload)

        # 5) Optional: just print arch and exit.
        if args.task == "arch":
            print(f"\n[ARCH JSON] {json.dumps(arch_record, indent=2)}")

    finally:
        try:
            gpu.shutdown()
        except Exception:
            pass

    print(f"\n[DONE] outputs written under {os.path.abspath(out_dir)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
