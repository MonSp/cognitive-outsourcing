"""Information-theoretic metrics for SIG injection analysis.

Merged from four sources:
  - r1_info_theory.py  — numpy-array KL/JS/entropy, MI via binning/KDE,
                         cosine similarity, logits helpers
  - r3_core.py         — (architecture model only; no standalone functions
                         to merge here)
  - r5_privacy.py      — text-based Shannon entropy, text-based KL
                         divergence, text-based mutual information
  - research/info_analysis.py — KSG MI estimator, head agreement rate,
                                per-layer cosine similarity, per-layer JS

All numpy-dependent functions degrade gracefully when numpy is unavailable.
"""

import math
from typing import List, Dict, Optional, Tuple
from collections import Counter

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

try:
    from scipy.spatial import cKDTree
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


# ── numpy-array helpers ────────────────────────────────────────────────

def _ensure_numpy(arr) -> "np.ndarray":
    if not NUMPY_AVAILABLE:
        raise RuntimeError("numpy is required for this function")
    if isinstance(arr, np.ndarray):
        return arr.astype(np.float64)
    return np.array(arr, dtype=np.float64)


def kl_divergence(p, q, epsilon: float = 1e-10) -> float:
    """KL divergence D_KL(P || Q) for numpy probability arrays.

    Both *p* and *q* are normalised internally.  *epsilon* is added for
    numerical stability (different from r1's base-parameter version; this
    uses the simpler epsilon-smoothing convention from info_analysis.py).
    """
    p = _ensure_numpy(p)
    q = _ensure_numpy(q)
    p = p + epsilon
    q = q + epsilon
    p = p / p.sum(axis=-1, keepdims=True)
    q = q / q.sum(axis=-1, keepdims=True)
    return float(np.sum(p * np.log(p / q), axis=-1))


def js_divergence(p, q, epsilon: float = 1e-10) -> float:
    """Jensen-Shannon divergence JS(P || Q) for numpy probability arrays.

    JS = 0.5 * KL(P||M) + 0.5 * KL(Q||M)  where M = 0.5*(P+Q).
    Bounded in [0, log(2)].
    """
    p = _ensure_numpy(p)
    q = _ensure_numpy(q)
    p = p + epsilon
    q = q + epsilon
    p = p / p.sum(axis=-1, keepdims=True)
    q = q / q.sum(axis=-1, keepdims=True)
    m = 0.5 * (p + q)
    js = 0.5 * np.sum(p * np.log(p / m), axis=-1) + \
         0.5 * np.sum(q * np.log(q / m), axis=-1)
    return float(np.mean(js))


def shannon_entropy_array(p, base: float = 2.0) -> float:
    """Shannon entropy H(P) for a numpy probability array.

    From r1_info_theory.py.  Zero-mass entries are dropped before
    computing.
    """
    p = _ensure_numpy(p)
    p = p / p.sum()
    p = p[p > 0]
    if base == 2.0:
        return float(-np.sum(p * np.log2(p)))
    elif base == math.e:
        return float(-np.sum(p * np.log(p)))
    else:
        return float(-np.sum(p * np.log(p)) / math.log(base))


def mutual_information_estimate(
    x,
    z,
    k: int = 3,
) -> float:
    """Estimate mutual information I(X; Z) using the KSG estimator.

    From research/info_analysis.py.  Requires scipy for cKDTree.
    Falls back to 0.0 when scipy is unavailable.
    """
    if not SCIPY_AVAILABLE or not NUMPY_AVAILABLE:
        return 0.0

    x = np.asarray(x, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    if z.ndim == 1:
        z = z.reshape(-1, 1)

    n_samples = x.shape[0]
    if n_samples != z.shape[0] or n_samples <= k:
        return 0.0

    xz = np.concatenate([x, z], axis=1)
    tree_xz = cKDTree(xz)
    tree_x = cKDTree(x)
    tree_z = cKDTree(z)

    distances, _ = tree_xz.query(xz, k=k + 1)
    eps = distances[:, k]

    nx = tree_x.query_radius(x, r=eps * 0.99999, count_only=True)
    nz = tree_z.query_radius(z, r=eps * 0.99999, count_only=True)

    nx = np.maximum(nx, 1)
    nz = np.maximum(nz, 1)

    def _digamma(v):
        return np.log(v) + 0.5772156649

    mi = _digamma(k) - np.mean(_digamma(nx) + _digamma(nz)) + _digamma(n_samples)
    return float(mi)


def cosine_similarity(a, b) -> float:
    """Cosine similarity between two numpy arrays (flattened)."""
    a = _ensure_numpy(a).flatten()
    b = _ensure_numpy(b).flatten()
    dot = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(dot / (norm_a * norm_b))


def head_agreement_rate(
    w_apploop,
    w_sig,
    threshold: float = 0.1,
    k: int = 5,
) -> float:
    """Fraction of attention heads whose top-k attended positions agree
    between AppLoop (full re-encoding) and SIG injection.

    From research/info_analysis.py.  *w_apploop* and *w_sig* have shape
    (num_layers, num_heads, seq_len, seq_len).
    """
    if not NUMPY_AVAILABLE:
        return 0.0
    w_apploop = np.asarray(w_apploop)
    w_sig = np.asarray(w_sig)
    num_layers = w_apploop.shape[0]
    num_heads = w_apploop.shape[1]
    total_agreement = 0.0
    total_heads = num_layers * num_heads
    if total_heads == 0:
        return 0.0
    for l in range(num_layers):
        for h in range(num_heads):
            full_topk = np.argsort(w_apploop[l, h], axis=-1)[..., -k:]
            sig_topk = np.argsort(w_sig[l, h], axis=-1)[..., -k:]
            full_sets = [set(full_topk[i].flatten()) for i in range(full_topk.shape[0])]
            sig_sets = [set(sig_topk[i].flatten()) for i in range(sig_topk.shape[0])]
            for fs, ss in zip(full_sets, sig_sets):
                total_agreement += len(fs & ss) / k
    return total_agreement / total_heads


def compute_layer_shifts(att_w, ref_w, epsilon: float = 1e-10) -> "np.ndarray":
    """Per-layer JS divergence between attention-weight distributions.

    From research/info_analysis.py (InjectionInfoAnalyzer._compute_js_per_layer).
    Returns a 1-D array of length num_layers.
    """
    if not NUMPY_AVAILABLE:
        return np.array([])
    att_w = np.asarray(att_w, dtype=np.float64)
    ref_w = np.asarray(ref_w, dtype=np.float64)
    num_layers = att_w.shape[0]
    js_vector = np.zeros(num_layers)
    for l in range(num_layers):
        p = np.abs(att_w[l].flatten())
        q = np.abs(ref_w[l].flatten())
        p = p / (p.sum() + epsilon)
        q = q / (q.sum() + epsilon)
        js_vector[l] = js_divergence(p, q, epsilon)
    return js_vector


# ── text-based helpers (no numpy required) ─────────────────────────────

def shannon_entropy(text: str) -> float:
    """Shannon entropy of a text string (bit units).

    From r5_privacy.py (PrivacyQuantifier.compute_shannon_entropy).
    Higher entropy ⇒ more information content ⇒ potentially more leakage.
    """
    if not text:
        return 0.0
    counter = Counter(text)
    length = len(text)
    entropy = 0.0
    for count in counter.values():
        p = count / length
        if p > 0:
            entropy -= p * math.log2(p)
    return entropy


def mutual_information_text(query: str, response: str) -> float:
    """Text-based mutual information approximation I(Q; R).

    From r5_privacy.py (PrivacyQuantifier.estimate_mutual_information).
    Uses character bigram Jaccard overlap weighted by the smaller entropy.
    """
    if not query or not response:
        return 0.0
    h_query = shannon_entropy(query)
    h_response = shannon_entropy(response)
    bigrams_q = set(zip(query, query[1:]))
    bigrams_r = set(zip(response, response[1:]))
    if not bigrams_q or not bigrams_r:
        return 0.0
    overlap = len(bigrams_q & bigrams_r)
    union = len(bigrams_q | bigrams_r)
    jaccard = overlap / union if union > 0 else 0.0
    return jaccard * min(h_query, h_response)
