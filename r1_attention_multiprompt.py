#!/usr/bin/env python3
"""R1 Multi-Prompt Attention Analysis with Statistical Validation.

Extends single-prompt R1 attention analysis to 5 diverse prompt types
with bootstrap confidence intervals and Wilcoxon signed-rank tests.

Usage:
    python r1_attention_multiprompt.py --model Qwen/Qwen2.5-0.5B --prompts 5 --output data/r1_multiprompt_results.json
"""

import argparse
import json
import sys
import os
import numpy as np
from typing import Dict, List, Tuple, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.info_theory import head_agreement_rate, cosine_similarity, compute_layer_shifts


# ── Prompt Definitions ──────────────────────────────────────────────────

PROMPTS = [
    {
        "label": "Travel Planning",
        "prefix": (
            "You are a travel assistant helping users plan trips.\n"
            "User: I need to plan a trip from New York to Tokyo next month. "
            "Can you help me find flights and suggest an itinerary?\n"
            "Assistant: I'd be happy to help you plan your trip to Tokyo! "
            "Let me search for available flights and local attractions.\n"
        ),
        "injection": (
            "[Tool Results]\n"
            "Flight search: JFK->NRT on Mar 15 - JAL Flight 5 (12h 40m, $890 economy). "
            "NRT->JFK on Mar 22 - ANA Flight 10 (13h 05m, $920 economy).\n"
            "Hotel: Shinjuku Granbell Hotel - $120/night, 4.3 stars, near train station.\n"
            "Weather Tokyo March: 8-16C, cherry blossom season begins late March.\n\n"
            "Based on these results:\n"
        ),
    },
    {
        "label": "Code Debugging",
        "prefix": (
            "You are a code debugging assistant.\n"
            "User: My Python web server keeps crashing with a 500 error. "
            "Here is the relevant code:\n"
            "```python\n"
            "def get_user(user_id):\n"
            "    conn = db.connect()\n"
            "    result = conn.execute('SELECT * FROM users WHERE id=' + user_id)\n"
            "    return result.fetchone()\n"
            "```\n"
            "Assistant: Let me analyze the code and run diagnostics.\n"
        ),
        "injection": (
            "[Tool Results]\n"
            "Static analysis: SQL injection vulnerability at line 3 - "
            "string concatenation in SQL query. Use parameterized query instead.\n"
            "Runtime error log: TypeError - user_id is int but concatenation expects str. "
            "Traceback: File 'app.py', line 3, in get_user.\n"
            "Test output: 3/5 tests failed - test_get_user_string_input, "
            "test_get_user_sql_injection, test_get_user_none_return.\n\n"
            "Here is the diagnosis:\n"
        ),
    },
    {
        "label": "Medical Consultation",
        "prefix": (
            "You are a medical information assistant. Always remind users to consult "
            "a healthcare professional for medical advice.\n"
            "User: I have been experiencing persistent headaches, dizziness, "
            "and fatigue for the past two weeks. I also noticed some blurred vision "
            "in the mornings. What could this be related to?\n"
            "Assistant: I understand your concerns. These symptoms warrant medical "
            "attention. Let me look up relevant clinical information.\n"
        ),
        "injection": (
            "[Tool Results]\n"
            "Blood test results: Hemoglobin 10.2 g/dL (low, normal 12-16), "
            "Ferritin 8 ng/mL (low, normal 12-150), Vitamin B12 180 pg/mL (low-normal).\n"
            "Blood pressure reading: 95/60 mmHg (low-normal).\n"
            "Clinical reference: Symptoms consistent with iron-deficiency anemia. "
            "Blurred vision may relate to hypotension. Recommend follow-up with PCP.\n\n"
            "Based on the results:\n"
        ),
    },
    {
        "label": "Financial Analysis",
        "prefix": (
            "You are a financial analysis assistant.\n"
            "User: I am considering investing in tech stocks. The market has been "
            "volatile lately with rising interest rates. Can you provide analysis "
            "on the current tech sector outlook and specific stock performance?\n"
            "Assistant: Let me pull up the latest market data and sector analysis.\n"
        ),
        "injection": (
            "[Tool Results]\n"
            "Market indices: S&P 500 -0.8% (week), NASDAQ -1.2% (week), "
            "Russell 2000 +0.3% (week). Fed funds rate: 5.25-5.50%.\n"
            "Stock data: AAPL $178.52 (-2.1% MTD, P/E 28.3), "
            "MSFT $378.91 (+0.8% MTD, P/E 35.2), "
            "NVDA $495.22 (+12.4% MTD, P/E 65.1).\n"
            "Sector analysis: Tech earnings growth +18% YoY but margin pressure "
            "from AI capex. Semiconductor subsector outperforming software.\n\n"
            "Here is the analysis:\n"
        ),
    },
    {
        "label": "General Conversation",
        "prefix": (
            "You are a helpful conversational assistant.\n"
            "User: I am planning a picnic this weekend with some friends. "
            "We are thinking of going to the lakeside park. Do you think "
            "that is a good idea? What should we bring?\n"
            "Assistant: A lakeside picnic sounds wonderful! Let me check "
            "the weather forecast and gather some suggestions for you.\n"
        ),
        "injection": (
            "[Tool Results]\n"
            "Weather forecast Saturday: Sunny, high 24C, low 16C, "
            "UV index 6 (moderate), wind 8 km/h NW, 5% chance of rain.\n"
            "Weather forecast Sunday: Partly cloudy, high 22C, low 15C, "
            "30% chance of afternoon showers.\n"
            "Park info: Lakeside Park - open 6am-9pm, BBQ grills available, "
            "parking $5, swimming area open, no alcohol permitted.\n\n"
            "Based on the forecast:\n"
        ),
    },
]


# ── GQA Compatibility ───────────────────────────────────────────────────

def detect_gqa(config) -> Tuple[bool, int, int, int]:
    """Detect GQA and return (is_gqa, num_attention_heads, num_kv_heads, ratio)."""
    n_heads = config.num_attention_heads
    n_kv_heads = getattr(config, "num_key_value_heads", n_heads)
    is_gqa = n_kv_heads < n_heads
    ratio = n_heads // n_kv_heads if is_gqa else 1
    return is_gqa, n_heads, n_kv_heads, ratio


def aggregate_gqa_attention(
    attn_per_layer: List[np.ndarray],
    num_kv_heads: int,
    ratio: int,
) -> List[np.ndarray]:
    """Aggregate attention weights by kv_head groups for GQA models.

    For each layer, reshape (num_heads, seq, seq) into
    (num_kv_heads, ratio, seq, seq) and average across the ratio axis,
    producing (num_kv_heads, seq, seq).
    """
    aggregated = []
    for attn in attn_per_layer:
        n_heads, seq_q, seq_k = attn.shape
        reshaped = attn.reshape(num_kv_heads, ratio, seq_q, seq_k)
        aggregated.append(reshaped.mean(axis=1))
    return aggregated


# ── Model Loading ───────────────────────────────────────────────────────

def load_model(model_id: str):
    """Load model and tokenizer from modelscope (with HuggingFace fallback)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    try:
        from modelscope import snapshot_download
        print(f"Downloading model from modelscope: {model_id}")
        model_dir = snapshot_download(model_id, cache_dir="./modelscope_cache")
        tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
            attn_implementation="eager",
        )
    except Exception:
        print(f"Modelscope failed, trying HuggingFace hub: {model_id}")
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
            attn_implementation="eager",
        )

    model.eval()
    return model, tokenizer


# ── Attention Extraction ────────────────────────────────────────────────

def extract_attention_full(model, tokenizer, prefix: str, injection: str):
    """Extract per-layer attention weights for full re-encoding (prefix + injection)."""
    import torch

    full_text = prefix + injection
    inputs = tokenizer(full_text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)

    attn_weights = [a.cpu().numpy()[0] for a in outputs.attentions]
    return attn_weights


def extract_attention_sig(model, tokenizer, prefix: str, injection: str):
    """Extract per-layer attention weights for SIG injection (prefix then injection via past_key_values)."""
    import torch

    inputs_prefix = tokenizer(prefix, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out_prefix = model(**inputs_prefix, output_attentions=True, use_cache=True)
    past_kv = out_prefix.past_key_values

    inputs_injection = tokenizer(injection, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out_injection = model(
            input_ids=inputs_injection["input_ids"],
            past_key_values=past_kv,
            output_attentions=True,
            use_cache=True,
        )

    attn_weights = [a.cpu().numpy()[0] for a in out_injection.attentions]
    return attn_weights


# ── Per-Layer Metrics ───────────────────────────────────────────────────

def compute_per_layer_head_agreement(
    attn_full: List[np.ndarray],
    attn_sig: List[np.ndarray],
    n_heads: int,
    k: int = 5,
) -> np.ndarray:
    """Compute head agreement rate per layer between full and SIG attention.

    Returns array of shape (num_layers,) with per-layer agreement rates.
    """
    num_layers = len(attn_full)
    agreements = np.zeros(num_layers)

    for l in range(num_layers):
        ms = min(attn_full[l].shape[1], attn_sig[l].shape[1])
        total_agreement = 0.0
        for h in range(n_heads):
            af = attn_full[l][h, :ms, :ms].mean(axis=0)
            ai = attn_sig[l][h, :ms, :ms].mean(axis=0)
            topk_full = set(np.argsort(-af)[:k])
            topk_sig = set(np.argsort(-ai)[:k])
            total_agreement += len(topk_full & topk_sig) / k
        agreements[l] = total_agreement / n_heads

    return agreements


def compute_per_layer_cosine_similarity(
    attn_full: List[np.ndarray],
    attn_sig: List[np.ndarray],
    n_heads: int,
) -> np.ndarray:
    """Compute per-layer cosine similarity between full and SIG attention patterns.

    Returns array of shape (num_layers,) with per-layer cosine similarities.
    """
    num_layers = len(attn_full)
    similarities = np.zeros(num_layers)

    for l in range(num_layers):
        ms = min(attn_full[l].shape[1], attn_sig[l].shape[1])
        csims = []
        for h in range(n_heads):
            af = attn_full[l][h, :ms, :ms].mean(axis=0)
            ai = attn_sig[l][h, :ms, :ms].mean(axis=0)
            dot = np.dot(af, ai)
            norm = np.linalg.norm(af) * np.linalg.norm(ai)
            csims.append(float(dot / max(norm, 1e-10)))
        similarities[l] = float(np.mean(csims))

    return similarities


def compute_per_layer_head_agreement_gqa(
    attn_full: List[np.ndarray],
    attn_sig: List[np.ndarray],
    num_kv_heads: int,
    ratio: int,
    k: int = 5,
) -> np.ndarray:
    """Compute head agreement rate per layer with GQA aggregation.

    Aggregates query heads sharing the same kv_head before computing agreement.
    Returns array of shape (num_layers,).
    """
    agg_full = aggregate_gqa_attention(attn_full, num_kv_heads, ratio)
    agg_sig = aggregate_gqa_attention(attn_sig, num_kv_heads, ratio)

    num_layers = len(agg_full)
    agreements = np.zeros(num_layers)

    for l in range(num_layers):
        ms = min(agg_full[l].shape[1], agg_sig[l].shape[1])
        total_agreement = 0.0
        for h in range(num_kv_heads):
            af = agg_full[l][h, :ms, :ms].mean(axis=0)
            ai = agg_sig[l][h, :ms, :ms].mean(axis=0)
            topk_full = set(np.argsort(-af)[:k])
            topk_sig = set(np.argsort(-ai)[:k])
            total_agreement += len(topk_full & topk_sig) / k
        agreements[l] = total_agreement / num_kv_heads

    return agreements


def compute_per_layer_cosine_similarity_gqa(
    attn_full: List[np.ndarray],
    attn_sig: List[np.ndarray],
    num_kv_heads: int,
    ratio: int,
) -> np.ndarray:
    """Compute per-layer cosine similarity with GQA aggregation.

    Returns array of shape (num_layers,).
    """
    agg_full = aggregate_gqa_attention(attn_full, num_kv_heads, ratio)
    agg_sig = aggregate_gqa_attention(attn_sig, num_kv_heads, ratio)

    num_layers = len(agg_full)
    similarities = np.zeros(num_layers)

    for l in range(num_layers):
        ms = min(agg_full[l].shape[1], agg_sig[l].shape[1])
        csims = []
        for h in range(num_kv_heads):
            af = agg_full[l][h, :ms, :ms].mean(axis=0)
            ai = agg_sig[l][h, :ms, :ms].mean(axis=0)
            dot = np.dot(af, ai)
            norm = np.linalg.norm(af) * np.linalg.norm(ai)
            csims.append(float(dot / max(norm, 1e-10)))
        similarities[l] = float(np.mean(csims))

    return similarities


# ── Statistical Analysis ────────────────────────────────────────────────

def bootstrap_ci(
    data: np.ndarray,
    n_resamples: int = 10000,
    ci: float = 0.95,
) -> Tuple[float, float, float]:
    """Compute mean with bootstrap confidence interval.

    Returns (mean, ci_lower, ci_upper).
    """
    data = np.asarray(data)
    n = len(data)
    if n == 0:
        return 0.0, 0.0, 0.0

    observed_mean = float(np.mean(data))
    bootstrap_means = np.empty(n_resamples)

    rng = np.random.default_rng(42)
    for i in range(n_resamples):
        sample = rng.choice(data, size=n, replace=True)
        bootstrap_means[i] = np.mean(sample)

    alpha = (1.0 - ci) / 2.0
    ci_lower = float(np.percentile(bootstrap_means, 100 * alpha))
    ci_upper = float(np.percentile(bootstrap_means, 100 * (1.0 - alpha)))

    return observed_mean, ci_lower, ci_upper


def wilcoxon_signed_rank_test(
    x: np.ndarray,
    y: np.ndarray,
) -> Dict:
    """Perform Wilcoxon signed-rank test and compute effect size.

    Tests whether x and y come from the same distribution.
    Effect size r = Z / sqrt(N) where N is total number of observations.
    """
    from scipy.stats import wilcoxon, rankdata

    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    n = len(x)
    if n < 2:
        return {
            "statistic": None,
            "p_value": None,
            "effect_size_r": None,
            "significant_005": False,
            "n_pairs": n,
        }

    try:
        result = wilcoxon(x, y, alternative="two-sided")
        stat = float(result.statistic)
        p_val = float(result.pvalue)
    except ValueError:
        return {
            "statistic": None,
            "p_value": None,
            "effect_size_r": None,
            "significant_005": False,
            "n_pairs": n,
        }

    diffs = x - y
    abs_diffs = np.abs(diffs)
    nonzero = abs_diffs > 0
    if nonzero.sum() == 0:
        effect_r = 0.0
    else:
        ranks = rankdata(abs_diffs[nonzero])
        z_approx = (stat - nonzero.sum() * (nonzero.sum() + 1) / 4.0) / np.sqrt(
            nonzero.sum() * (nonzero.sum() + 1) * (2 * nonzero.sum() + 1) / 24.0
        )
        effect_r = float(abs(z_approx) / np.sqrt(nonzero.sum()))

    return {
        "statistic": stat,
        "p_value": p_val,
        "effect_size_r": effect_r,
        "significant_005": p_val < 0.05,
        "n_pairs": n,
    }


# ── Main Pipeline ───────────────────────────────────────────────────────

def run_multiprompt_analysis(
    model_id: str = "Qwen/Qwen2.5-0.5B",
    n_prompts: int = 5,
    output_path: Optional[str] = None,
) -> Dict:
    """Run R1 multi-prompt attention analysis with statistical validation."""

    print(f"\n{'='*70}")
    print(f"  R1 Multi-Prompt Attention Analysis")
    print(f"  Model: {model_id}")
    print(f"  Prompts: {n_prompts}")
    print(f"{'='*70}\n")

    model, tokenizer = load_model(model_id)

    is_gqa, n_heads, n_kv_heads, gqa_ratio = detect_gqa(model.config)
    n_layers = model.config.num_hidden_layers
    n_params = sum(p.numel() for p in model.parameters()) / 1e9

    print(f"  Model: {n_layers} layers, {n_heads} heads, {n_params:.2f}B params")
    if is_gqa:
        print(f"  GQA detected: {n_kv_heads} kv_heads, ratio={gqa_ratio}")
    else:
        print(f"  No GQA (MHA): {n_kv_heads} kv_heads == {n_heads} attention heads")

    selected_prompts = PROMPTS[:n_prompts]

    all_head_agreements = []
    all_cosine_similarities = []
    per_prompt_results = []

    for pi, prompt in enumerate(selected_prompts):
        label = prompt["label"]
        prefix = prompt["prefix"]
        injection = prompt["injection"]

        print(f"\n  Prompt {pi+1}/{n_prompts}: {label}")

        print(f"    Extracting full re-encoding attention...")
        attn_full = extract_attention_full(model, tokenizer, prefix, injection)

        print(f"    Extracting SIG injection attention...")
        attn_sig = extract_attention_sig(model, tokenizer, prefix, injection)

        if is_gqa:
            ha = compute_per_layer_head_agreement_gqa(
                attn_full, attn_sig, n_kv_heads, gqa_ratio, k=5
            )
            cs = compute_per_layer_cosine_similarity_gqa(
                attn_full, attn_sig, n_kv_heads, gqa_ratio
            )
        else:
            ha = compute_per_layer_head_agreement(attn_full, attn_sig, n_heads, k=5)
            cs = compute_per_layer_cosine_similarity(attn_full, attn_sig, n_heads)

        all_head_agreements.append(ha)
        all_cosine_similarities.append(cs)

        per_prompt_results.append({
            "label": label,
            "head_agreement_per_layer": ha.tolist(),
            "cosine_similarity_per_layer": cs.tolist(),
            "mean_head_agreement": float(np.mean(ha)),
            "mean_cosine_similarity": float(np.mean(cs)),
        })

        print(f"    Mean head agreement: {np.mean(ha):.4f}")
        print(f"    Mean cosine similarity: {np.mean(cs):.4f}")

    ha_matrix = np.array(all_head_agreements)
    cs_matrix = np.array(all_cosine_similarities)

    print(f"\n{'='*70}")
    print(f"  Statistical Aggregation ({n_prompts} prompts)")
    print(f"{'='*70}\n")

    aggregated_layers = []
    for l in range(n_layers):
        ha_vals = ha_matrix[:, l]
        cs_vals = cs_matrix[:, l]
        ha_mean, ha_lo, ha_hi = bootstrap_ci(ha_vals, n_resamples=10000, ci=0.95)
        cs_mean, cs_lo, cs_hi = bootstrap_ci(cs_vals, n_resamples=10000, ci=0.95)
        aggregated_layers.append({
            "layer": l,
            "head_agreement_mean": ha_mean,
            "head_agreement_ci_lower": ha_lo,
            "head_agreement_ci_upper": ha_hi,
            "cosine_similarity_mean": cs_mean,
            "cosine_similarity_ci_lower": cs_lo,
            "cosine_similarity_ci_upper": cs_hi,
        })

    early_end = min(8, n_layers // 3)
    late_start = max(n_layers - 8, 2 * n_layers // 3)
    if late_start <= early_end:
        late_start = early_end + 1

    early_ha = ha_matrix[:, :early_end].flatten()
    late_ha = ha_matrix[:, late_start:].flatten()
    early_cs = cs_matrix[:, :early_end].flatten()
    late_cs = cs_matrix[:, late_start:].flatten()

    ha_test = wilcoxon_signed_rank_test(early_ha, late_ha)
    cs_test = wilcoxon_signed_rank_test(early_cs, late_cs)

    early_ha_summary = bootstrap_ci(early_ha, n_resamples=10000, ci=0.95)
    late_ha_summary = bootstrap_ci(late_ha, n_resamples=10000, ci=0.95)
    early_cs_summary = bootstrap_ci(early_cs, n_resamples=10000, ci=0.95)
    late_cs_summary = bootstrap_ci(late_cs, n_resamples=10000, ci=0.95)

    print(f"  Early layers (0-{early_end-1}):")
    print(f"    Head agreement: {early_ha_summary[0]:.4f} "
          f"[{early_ha_summary[1]:.4f}, {early_ha_summary[2]:.4f}]")
    print(f"    Cosine similarity: {early_cs_summary[0]:.4f} "
          f"[{early_cs_summary[1]:.4f}, {early_cs_summary[2]:.4f}]")

    print(f"\n  Late layers ({late_start}-{n_layers-1}):")
    print(f"    Head agreement: {late_ha_summary[0]:.4f} "
          f"[{late_ha_summary[1]:.4f}, {late_ha_summary[2]:.4f}]")
    print(f"    Cosine similarity: {late_cs_summary[0]:.4f} "
          f"[{late_cs_summary[1]:.4f}, {late_cs_summary[2]:.4f}]")

    print(f"\n  Wilcoxon signed-rank test (early vs late):")
    print(f"    Head agreement: p={ha_test['p_value']:.6f}, "
          f"r={ha_test['effect_size_r']:.4f}, "
          f"significant={ha_test['significant_005']}")
    print(f"    Cosine similarity: p={cs_test['p_value']:.6f}, "
          f"r={cs_test['effect_size_r']:.4f}, "
          f"significant={cs_test['significant_005']}")

    if ha_test["p_value"] is not None:
        if early_ha_summary[0] < late_ha_summary[0]:
            print(f"\n  -> Early layers have LOWER head agreement than late layers "
                  f"(consistent with SIG injection hypothesis)")
        else:
            print(f"\n  -> Early layers do NOT have lower head agreement than late layers")

    overall_ha = bootstrap_ci(ha_matrix.flatten(), n_resamples=10000, ci=0.95)
    overall_cs = bootstrap_ci(cs_matrix.flatten(), n_resamples=10000, ci=0.95)

    result = {
        "task": "r1_multiprompt_attention",
        "model_id": model_id,
        "n_params_b": n_params,
        "n_layers": n_layers,
        "n_heads": n_heads,
        "n_prompts": n_prompts,
        "gqa": {
            "detected": is_gqa,
            "num_kv_heads": n_kv_heads,
            "gqa_ratio": gqa_ratio,
        },
        "overall": {
            "head_agreement_mean": overall_ha[0],
            "head_agreement_ci_lower": overall_ha[1],
            "head_agreement_ci_upper": overall_ha[2],
            "cosine_similarity_mean": overall_cs[0],
            "cosine_similarity_ci_lower": overall_cs[1],
            "cosine_similarity_ci_upper": overall_cs[2],
        },
        "early_layers": {
            "range": f"0-{early_end-1}",
            "head_agreement_mean": early_ha_summary[0],
            "head_agreement_ci_lower": early_ha_summary[1],
            "head_agreement_ci_upper": early_ha_summary[2],
            "cosine_similarity_mean": early_cs_summary[0],
            "cosine_similarity_ci_lower": early_cs_summary[1],
            "cosine_similarity_ci_upper": early_cs_summary[2],
        },
        "late_layers": {
            "range": f"{late_start}-{n_layers-1}",
            "head_agreement_mean": late_ha_summary[0],
            "head_agreement_ci_lower": late_ha_summary[1],
            "head_agreement_ci_upper": late_ha_summary[2],
            "cosine_similarity_mean": late_cs_summary[0],
            "cosine_similarity_ci_lower": late_cs_summary[1],
            "cosine_similarity_ci_upper": late_cs_summary[2],
        },
        "wilcoxon_test": {
            "head_agreement": ha_test,
            "cosine_similarity": cs_test,
        },
        "per_prompt": per_prompt_results,
        "per_layer": aggregated_layers,
    }

    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"\n  Results saved to: {output_path}")

    return result


# ── CLI ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="R1 Multi-Prompt Attention Analysis with Statistical Validation"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-0.5B",
        help="HuggingFace model ID (default: Qwen/Qwen2.5-0.5B)",
    )
    parser.add_argument(
        "--prompts",
        type=int,
        default=5,
        choices=[1, 2, 3, 4, 5],
        help="Number of prompts to evaluate (1-5, default: 5)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/r1_multiprompt_results.json",
        help="Output JSON file path (default: data/r1_multiprompt_results.json)",
    )
    args = parser.parse_args()

    run_multiprompt_analysis(
        model_id=args.model,
        n_prompts=args.prompts,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
