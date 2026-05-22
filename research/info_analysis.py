"""
Information Analysis Module for SIG Injection Research

Provides tools for computing information-theoretic metrics between
full re-encoding and SIG injection methods, including KL divergence,
JS divergence, head agreement rate, and information retention probing.
"""

import numpy as np
import torch
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict


def kl_divergence(p: np.ndarray, q: np.ndarray, epsilon: float = 1e-12) -> float:
    """
    Compute KL divergence between two probability distributions.
    
    KL(P || Q) = sum_i P(i) * log(P(i) / Q(i))
    
    Parameters
    ----------
    p : np.ndarray
        Reference probability distribution (P).
    q : np.ndarray
        Target probability distribution (Q).
    epsilon : float, optional
        Small constant for numerical stability. Default is 1e-12.
        
    Returns
    -------
    float
        KL divergence value.
    """
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    
    p = p + epsilon
    q = q + epsilon
    
    p = p / p.sum(axis=-1, keepdims=True)
    q = q / q.sum(axis=-1, keepdims=True)
    
    kl = np.sum(p * np.log(p / q), axis=-1)
    return float(np.mean(kl))


def kl_divergence_symmetric(p: np.ndarray, q: np.ndarray, epsilon: float = 1e-12) -> float:
    """
    Compute symmetric KL divergence between two distributions.
    
    Symmetric KL = 0.5 * (KL(P||Q) + KL(Q||P))
    
    Parameters
    ----------
    p : np.ndarray
        First probability distribution.
    q : np.ndarray
        Second probability distribution.
    epsilon : float, optional
        Small constant for numerical stability.
        
    Returns
    -------
    float
        Symmetric KL divergence value.
    """
    kl_pq = kl_divergence(p, q, epsilon)
    kl_qp = kl_divergence(q, p, epsilon)
    return 0.5 * (kl_pq + kl_qp)


def js_divergence(p: np.ndarray, q: np.ndarray, epsilon: float = 1e-12) -> float:
    """
    Compute Jensen-Shannon divergence between two distributions.
    
    JS(P || Q) = 0.5 * KL(P || M) + 0.5 * KL(Q || M)
    where M = 0.5 * (P + Q)
    
    Parameters
    ----------
    p : np.ndarray
        First probability distribution.
    q : np.ndarray
        Second probability distribution.
    epsilon : float, optional
        Small constant for numerical stability.
        
    Returns
    -------
    float
        JS divergence value in range [0, log(2)].
    """
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    
    p = p + epsilon
    q = q + epsilon
    
    p = p / p.sum(axis=-1, keepdims=True)
    q = q / q.sum(axis=-1, keepdims=True)
    
    m = 0.5 * (p + q)
    
    js = 0.5 * np.sum(p * np.log(p / m), axis=-1) + \
         0.5 * np.sum(q * np.log(q / m), axis=-1)
    return float(np.mean(js))


def js_divergence_sqrt(p: np.ndarray, q: np.ndarray, epsilon: float = 1e-12) -> float:
    """
    Compute Jensen-Shannon distance (sqrt of JS divergence).
    
    This metric is bounded in [0, 1] and satisfies triangle inequality.
    
    Parameters
    ----------
    p : np.ndarray
        First probability distribution.
    q : np.ndarray
        Second probability distribution.
    epsilon : float, optional
        Small constant for numerical stability.
        
    Returns
    -------
    float
        JS distance value in range [0, 1].
    """
    return float(np.sqrt(js_divergence(p, q, epsilon)))


def head_agreement_rate(
    attention_full: np.ndarray,
    attention_inject: np.ndarray,
    k: int = 5
) -> float:
    """
    Compute head agreement rate between full re-encoding and SIG injection.
    
    Measures the fraction of attention heads where top-k attended positions
    agree between the two methods.
    
    Parameters
    ----------
    attention_full : np.ndarray
        Attention weights from full re-encoding.
        Shape: (num_layers, num_heads, seq_len, seq_len)
    attention_inject : np.ndarray
        Attention weights from SIG injection.
        Shape: (num_layers, num_heads, seq_len, seq_len)
    k : int, optional
        Number of top attended positions to compare. Default is 5.
        
    Returns
    -------
    float
        Agreement rate in range [0, 1].
    """
    attention_full = np.asarray(attention_full)
    attention_inject = np.asarray(attention_inject)
    
    num_layers = attention_full.shape[0]
    num_heads = attention_full.shape[1]
    
    total_agreement = 0.0
    total_heads = num_layers * num_heads
    
    for l in range(num_layers):
        for h in range(num_heads):
            full_topk = np.argsort(attention_full[l, h], axis=-1)[..., -k:]
            inject_topk = np.argsort(attention_inject[l, h], axis=-1)[..., -k:]
            
            full_sets = [set(full_topk[i].flatten()) for i in range(full_topk.shape[0])]
            inject_sets = [set(inject_topk[i].flatten()) for i in range(inject_topk.shape[0])]
            
            for fs, ins in zip(full_sets, inject_sets):
                intersection = len(fs & ins)
                total_agreement += intersection / k
    
    return total_agreement / total_heads


def head_agreement_rate_per_layer(
    attention_full: np.ndarray,
    attention_inject: np.ndarray,
    k: int = 5
) -> np.ndarray:
    """
    Compute head agreement rate per layer.
    
    Parameters
    ----------
    attention_full : np.ndarray
        Attention weights from full re-encoding.
        Shape: (num_layers, num_heads, seq_len, seq_len)
    attention_inject : np.ndarray
        Attention weights from SIG injection.
        Shape: (num_layers, num_heads, seq_len, seq_len)
    k : int, optional
        Number of top attended positions to compare.
        
    Returns
    -------
    np.ndarray
        Agreement rates per layer. Shape: (num_layers,)
    """
    attention_full = np.asarray(attention_full)
    attention_inject = np.asarray(attention_inject)
    
    num_layers = attention_full.shape[0]
    num_heads = attention_full.shape[1]
    
    layer_agreements = np.zeros(num_layers)
    
    for l in range(num_layers):
        total_agreement = 0.0
        for h in range(num_heads):
            full_topk = np.argsort(attention_full[l, h], axis=-1)[..., -k:]
            inject_topk = np.argsort(attention_inject[l, h], axis=-1)[..., -k:]
            
            full_sets = [set(full_topk[i].flatten()) for i in range(full_topk.shape[0])]
            inject_sets = [set(inject_topk[i].flatten()) for i in range(inject_topk.shape[0])]
            
            for fs, ins in zip(full_sets, inject_sets):
                intersection = len(fs & ins)
                total_agreement += intersection / k
        
        layer_agreements[l] = total_agreement / num_heads
    
    return layer_agreements


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute cosine similarity between two vectors or matrices.
    
    Parameters
    ----------
    a : np.ndarray
        First array.
    b : np.ndarray
        Second array.
        
    Returns
    -------
    float
        Cosine similarity value in range [-1, 1].
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    
    a_norm = a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-12)
    b_norm = b / (np.linalg.norm(b, axis=-1, keepdims=True) + 1e-12)
    
    sim = np.sum(a_norm * b_norm, axis=-1)
    return float(np.mean(sim))


def cosine_similarity_per_layer(
    hidden_full: np.ndarray,
    hidden_inject: np.ndarray
) -> np.ndarray:
    """
    Compute cosine similarity per layer.
    
    Parameters
    ----------
    hidden_full : np.ndarray
        Hidden states from full re-encoding.
        Shape: (num_layers, seq_len, hidden_dim)
    hidden_inject : np.ndarray
        Hidden states from SIG injection.
        Shape: (num_layers, seq_len, hidden_dim)
        
    Returns
    -------
    np.ndarray
        Cosine similarities per layer. Shape: (num_layers,)
    """
    hidden_full = np.asarray(hidden_full, dtype=np.float64)
    hidden_inject = np.asarray(hidden_inject, dtype=np.float64)
    
    num_layers = hidden_full.shape[0]
    similarities = np.zeros(num_layers)
    
    for l in range(num_layers):
        a = hidden_full[l]
        b = hidden_inject[l]
        
        a_norm = a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-12)
        b_norm = b / (np.linalg.norm(b, axis=-1, keepdims=True) + 1e-12)
        
        similarities[l] = float(np.mean(np.sum(a_norm * b_norm, axis=-1)))
    
    return similarities


def mutual_information_estimate(
    x: np.ndarray,
    z: np.ndarray,
    k_neighbors: int = 6
) -> float:
    """
    Estimate mutual information I(X; Z) using KSG estimator.
    
    Kraskov-Stögbauer-Grassberger (KSG) estimator for mutual information.
    
    Parameters
    ----------
    x : np.ndarray
        Input variable samples. Shape: (n_samples, x_dim)
    z : np.ndarray
        Representation variable samples. Shape: (n_samples, z_dim)
    k_neighbors : int, optional
        Number of neighbors for KSG estimation. Default is 6.
        
    Returns
    -------
    float
        Estimated mutual information.
    """
    from scipy.spatial import cKDTree
    
    x = np.asarray(x, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    
    n_samples = x.shape[0]
    
    xz = np.concatenate([x, z], axis=1)
    tree_xz = cKDTree(xz)
    tree_x = cKDTree(x)
    tree_z = cKDTree(z)
    
    distances, _ = tree_xz.query(xz, k=k_neighbors + 1)
    eps = distances[:, k_neighbors]
    
    nx = tree_x.query_radius(xz[:, :x.shape[1]], r=eps * 0.99999, count_only=True)
    nz = tree_z.query_radius(xz[:, x.shape[1]:], r=eps * 0.99999, count_only=True)
    
    nx = np.maximum(nx, 1)
    nz = np.maximum(nz, 1)
    
    digamma = np.vectorize(lambda v: np.log(v) + 0.5772156649)
    
    mi = digamma(k_neighbors) - np.mean(digamma(nx) + digamma(nz)) + digamma(n_samples)
    
    return float(mi)


class InformationRetentionProbe:
    """
    Probe for measuring information retention across transformer layers.
    
    This class trains linear probes on hidden states to estimate how much
    task-relevant information is preserved at each layer under different
    encoding methods (full re-encoding vs SIG injection).
    
    Parameters
    ----------
    num_layers : int
        Number of transformer layers to probe.
    hidden_dim : int
        Dimensionality of hidden states.
    num_classes : int, optional
        Number of classes for the probe task. Default is 2.
    device : str, optional
        Device to run probes on. Default is 'cpu'.
    """
    
    def __init__(
        self,
        num_layers: int,
        hidden_dim: int,
        num_classes: int = 2,
        device: str = 'cpu'
    ):
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.device = device
        
        self.probes_full = self._init_probes()
        self.probes_inject = self._init_probes()
        
        self.is_fitted = False
    
    def _init_probes(self) -> List[np.ndarray]:
        """Initialize linear probe weights for each layer."""
        probes = []
        for _ in range(self.num_layers):
            w = np.random.randn(self.hidden_dim, self.num_classes) * 0.01
            probes.append(w)
        return probes
    
    def train_probes(
        self,
        hidden_states_full: np.ndarray,
        hidden_states_inject: np.ndarray,
        labels: np.ndarray,
        learning_rate: float = 0.01,
        num_epochs: int = 100,
        reg_lambda: float = 0.01
    ) -> Dict[str, List[float]]:
        """
        Train linear probes on hidden states from both methods.
        
        Parameters
        ----------
        hidden_states_full : np.ndarray
            Hidden states from full re-encoding.
            Shape: (num_layers, n_samples, hidden_dim)
        hidden_states_inject : np.ndarray
            Hidden states from SIG injection.
            Shape: (num_layers, n_samples, hidden_dim)
        labels : np.ndarray
            Labels for the probe task. Shape: (n_samples,)
        learning_rate : float, optional
            Learning rate for gradient descent.
        num_epochs : int, optional
            Number of training epochs.
        reg_lambda : float, optional
            L2 regularization strength.
            
        Returns
        -------
        Dict[str, List[float]]
            Training history with accuracies per layer per epoch.
        """
        hidden_states_full = np.asarray(hidden_states_full)
        hidden_states_inject = np.asarray(hidden_states_inject)
        labels = np.asarray(labels)
        
        history = {
            'accuracy_full': [[] for _ in range(self.num_layers)],
            'accuracy_inject': [[] for _ in range(self.num_layers)],
        }
        
        for layer in range(self.num_layers):
            for epoch in range(num_epochs):
                acc_f = self._train_single_probe(
                    self.probes_full[layer],
                    hidden_states_full[layer],
                    labels,
                    learning_rate,
                    reg_lambda
                )
                acc_i = self._train_single_probe(
                    self.probes_inject[layer],
                    hidden_states_inject[layer],
                    labels,
                    learning_rate,
                    reg_lambda
                )
                
                history['accuracy_full'][layer].append(acc_f)
                history['accuracy_inject'][layer].append(acc_i)
        
        self.is_fitted = True
        return history
    
    def _train_single_probe(
        self,
        probe: np.ndarray,
        hidden: np.ndarray,
        labels: np.ndarray,
        lr: float,
        reg_lambda: float
    ) -> float:
        """Train a single linear probe using logistic regression."""
        logits = hidden @ probe
        
        if self.num_classes == 2:
            probs = 1.0 / (1.0 + np.exp(-logits))
            preds = (probs > 0.5).astype(int).flatten()
            
            error = (probs.flatten() - labels)
            grad = (hidden.T @ error[:, np.newaxis]) / len(labels)
            grad += reg_lambda * probe
            probe -= lr * grad
            
            accuracy = float(np.mean(preds == labels))
        else:
            exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
            probs = exp_logits / (exp_logits.sum(axis=-1, keepdims=True) + 1e-12)
            preds = np.argmax(probs, axis=-1)
            
            one_hot = np.zeros_like(probs)
            one_hot[np.arange(len(labels)), labels] = 1.0
            error = probs - one_hot
            grad = (hidden.T @ error) / len(labels)
            grad += reg_lambda * probe
            probe -= lr * grad
            
            accuracy = float(np.mean(preds == labels))
        
        return accuracy
    
    def get_retention_scores(self) -> np.ndarray:
        """
        Compute information retention scores per layer.
        
        Retention score = accuracy_inject / accuracy_full
        
        Returns
        -------
        np.ndarray
            Retention scores per layer. Shape: (num_layers,)
        """
        if not self.is_fitted:
            raise ValueError("Probes must be trained first. Call train_probes().")
        
        final_acc_full = np.array([acc[-1] for acc in self.history['accuracy_full']])
        final_acc_inject = np.array([acc[-1] for acc in self.history['accuracy_inject']])
        
        retention = final_acc_inject / (final_acc_full + 1e-12)
        return retention
    
    def evaluate(
        self,
        hidden_states_full: np.ndarray,
        hidden_states_inject: np.ndarray,
        labels: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        Evaluate trained probes on test data.
        
        Parameters
        ----------
        hidden_states_full : np.ndarray
            Test hidden states from full re-encoding.
        hidden_states_inject : np.ndarray
            Test hidden states from SIG injection.
        labels : np.ndarray
            Test labels.
            
        Returns
        -------
        Dict[str, np.ndarray]
            Dictionary with accuracies per layer for both methods.
        """
        if not self.is_fitted:
            raise ValueError("Probes must be trained first.")
        
        accuracies_full = np.zeros(self.num_layers)
        accuracies_inject = np.zeros(self.num_layers)
        
        for layer in range(self.num_layers):
            logits_f = hidden_states_full[layer] @ self.probes_full[layer]
            logits_i = hidden_states_inject[layer] @ self.probes_inject[layer]
            
            if self.num_classes == 2:
                preds_f = (logits_f > 0).astype(int).flatten()
                preds_i = (logits_i > 0).astype(int).flatten()
            else:
                preds_f = np.argmax(logits_f, axis=-1)
                preds_i = np.argmax(logits_i, axis=-1)
            
            accuracies_full[layer] = float(np.mean(preds_f == labels))
            accuracies_inject[layer] = float(np.mean(preds_i == labels))
        
        return {
            'accuracy_full': accuracies_full,
            'accuracy_inject': accuracies_inject,
            'retention_rate': accuracies_inject / (accuracies_full + 1e-12)
        }


class InjectionInfoAnalyzer:
    """
    Comprehensive analyzer for SIG injection information-theoretic properties.
    
    This class orchestrates the computation of all information metrics
    between full re-encoding and SIG injection methods, providing
    structured reports and statistical summaries.
    
    Parameters
    ----------
    num_layers : int
        Number of transformer layers in the model.
    num_heads : int
        Number of attention heads per layer.
    epsilon : float, optional
        Numerical stability constant for divergence computations.
    """
    
    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        epsilon: float = 1e-12
    ):
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.epsilon = epsilon
        
        self.results = {}
    
    def analyze(
        self,
        hidden_states_full: np.ndarray,
        hidden_states_inject: np.ndarray,
        attention_full: np.ndarray,
        attention_inject: np.ndarray,
        output_dist_full: Optional[np.ndarray] = None,
        output_dist_inject: Optional[np.ndarray] = None
    ) -> Dict:
        """
        Run comprehensive information analysis.
        
        Parameters
        ----------
        hidden_states_full : np.ndarray
            Hidden states from full re-encoding.
            Shape: (num_layers, seq_len, hidden_dim)
        hidden_states_inject : np.ndarray
            Hidden states from SIG injection.
            Shape: (num_layers, seq_len, hidden_dim)
        attention_full : np.ndarray
            Attention weights from full re-encoding.
            Shape: (num_layers, num_heads, seq_len, seq_len)
        attention_inject : np.ndarray
            Attention weights from SIG injection.
            Shape: (num_layers, num_heads, seq_len, seq_len)
        output_dist_full : np.ndarray, optional
            Output distribution from full re-encoding.
        output_dist_inject : np.ndarray, optional
            Output distribution from SIG injection.
            
        Returns
        -------
        Dict
            Comprehensive analysis results.
        """
        self.results = {}
        
        self.results['kl_divergence'] = self._compute_kl_per_head(
            attention_full, attention_inject
        )
        
        self.results['kl_divergence_symmetric'] = self._compute_kl_symmetric_per_head(
            attention_full, attention_inject
        )
        
        self.results['js_divergence'] = self._compute_js_per_layer(
            hidden_states_full, hidden_states_inject
        )
        
        self.results['head_agreement_rate'] = head_agreement_rate(
            attention_full, attention_inject, k=5
        )
        
        self.results['head_agreement_per_layer'] = head_agreement_rate_per_layer(
            attention_full, attention_inject, k=5
        )
        
        self.results['head_agreement_per_layer_k1'] = head_agreement_rate_per_layer(
            attention_full, attention_inject, k=1
        )
        
        self.results['head_agreement_per_layer_k3'] = head_agreement_rate_per_layer(
            attention_full, attention_inject, k=3
        )
        
        self.results['cosine_similarity'] = cosine_similarity_per_layer(
            hidden_states_full, hidden_states_inject
        )
        
        if output_dist_full is not None and output_dist_inject is not None:
            self.results['output_kl'] = kl_divergence_symmetric(
                output_dist_full, output_dist_inject, self.epsilon
            )
            self.results['output_js'] = js_divergence(
                output_dist_full, output_dist_inject, self.epsilon
            )
        
        self.results['summary'] = self._generate_summary()
        
        return self.results
    
    def _compute_kl_per_head(
        self,
        attention_full: np.ndarray,
        attention_inject: np.ndarray
    ) -> np.ndarray:
        """Compute KL divergence per head per layer."""
        kl_matrix = np.zeros((self.num_layers, self.num_heads))
        
        for l in range(self.num_layers):
            for h in range(self.num_heads):
                p = attention_full[l, h].mean(axis=0)
                q = attention_inject[l, h].mean(axis=0)
                kl_matrix[l, h] = kl_divergence(p, q, self.epsilon)
        
        return kl_matrix
    
    def _compute_kl_symmetric_per_head(
        self,
        attention_full: np.ndarray,
        attention_inject: np.ndarray
    ) -> np.ndarray:
        """Compute symmetric KL divergence per head per layer."""
        kl_matrix = np.zeros((self.num_layers, self.num_heads))
        
        for l in range(self.num_layers):
            for h in range(self.num_heads):
                p = attention_full[l, h].mean(axis=0)
                q = attention_inject[l, h].mean(axis=0)
                kl_matrix[l, h] = kl_divergence_symmetric(p, q, self.epsilon)
        
        return kl_matrix
    
    def _compute_js_per_layer(
        self,
        hidden_full: np.ndarray,
        hidden_inject: np.ndarray
    ) -> np.ndarray:
        """Compute JS divergence per layer."""
        js_vector = np.zeros(self.num_layers)
        
        for l in range(self.num_layers):
            p = hidden_full[l].flatten()
            q = hidden_inject[l].flatten()
            
            p = np.abs(p)
            q = np.abs(q)
            
            p = p / (p.sum() + self.epsilon)
            q = q / (q.sum() + self.epsilon)
            
            js_vector[l] = js_divergence(p, q, self.epsilon)
        
        return js_vector
    
    def _generate_summary(self) -> Dict:
        """Generate statistical summary of analysis results."""
        kl = self.results['kl_divergence']
        js = self.results['js_divergence']
        cos_sim = self.results['cosine_similarity']
        
        early_layers = slice(0, max(1, self.num_layers // 3))
        mid_layers = slice(self.num_layers // 3, 2 * self.num_layers // 3)
        late_layers = slice(2 * self.num_layers // 3, self.num_layers)
        
        summary = {
            'overall': {
                'mean_kl': float(np.mean(kl)),
                'std_kl': float(np.std(kl)),
                'max_kl': float(np.max(kl)),
                'mean_js': float(np.mean(js)),
                'mean_cosine_similarity': float(np.mean(cos_sim)),
            },
            'by_region': {
                'early': {
                    'mean_kl': float(np.mean(kl[early_layers])),
                    'mean_js': float(np.mean(js[early_layers])),
                    'mean_cosine_sim': float(np.mean(cos_sim[early_layers])),
                    'head_agreement': float(np.mean(
                        self.results['head_agreement_per_layer'][early_layers]
                    )),
                },
                'mid': {
                    'mean_kl': float(np.mean(kl[mid_layers])),
                    'mean_js': float(np.mean(js[mid_layers])),
                    'mean_cosine_sim': float(np.mean(cos_sim[mid_layers])),
                    'head_agreement': float(np.mean(
                        self.results['head_agreement_per_layer'][mid_layers]
                    )),
                },
                'late': {
                    'mean_kl': float(np.mean(kl[late_layers])),
                    'mean_js': float(np.mean(js[late_layers])),
                    'mean_cosine_sim': float(np.mean(cos_sim[late_layers])),
                    'head_agreement': float(np.mean(
                        self.results['head_agreement_per_layer'][late_layers]
                    )),
                },
            },
            'layer_sensitivity_ranking': np.argsort(-np.mean(kl, axis=1)).tolist(),
        }
        
        return summary
    
    def get_layer_sensitivity_profile(self) -> Dict:
        """
        Get layer sensitivity profile with hypothesis testing.
        
        Returns
        -------
        Dict
            Sensitivity analysis with statistical tests.
        """
        if 'kl_divergence' not in self.results:
            raise ValueError("Run analyze() first.")
        
        kl = self.results['kl_divergence']
        js = self.results['js_divergence']
        
        layer_means_kl = np.mean(kl, axis=1)
        layer_means_js = js
        
        early_mean_kl = np.mean(layer_means_kl[:self.num_layers // 3])
        mid_mean_kl = np.mean(layer_means_kl[self.num_layers // 3:2 * self.num_layers // 3])
        late_mean_kl = np.mean(layer_means_kl[2 * self.num_layers // 3:])
        
        early_mean_js = np.mean(layer_means_js[:self.num_layers // 3])
        mid_mean_js = np.mean(layer_means_js[self.num_layers // 3:2 * self.num_layers // 3])
        late_mean_js = np.mean(layer_means_js[2 * self.num_layers // 3:])
        
        profile = {
            'h1_early_most_sensitive': bool(
                early_mean_kl > mid_mean_kl and early_mean_kl > late_mean_kl
            ),
            'h2_mid_recovery': bool(mid_mean_kl < early_mean_kl),
            'h3_late_least_sensitive': bool(
                late_mean_kl < mid_mean_kl and late_mean_kl < early_mean_kl
            ),
            'kl_by_region': {
                'early': float(early_mean_kl),
                'mid': float(mid_mean_kl),
                'late': float(late_mean_kl),
            },
            'js_by_region': {
                'early': float(early_mean_js),
                'mid': float(mid_mean_js),
                'late': float(late_mean_js),
            },
            'information_loss_upper_bound': float(
                np.sum(layer_means_kl) / (self.num_layers * self.num_heads)
            ),
        }
        
        return profile
    
    def compute_information_bottleneck_ratio(
        self,
        input_samples: np.ndarray,
        hidden_full: np.ndarray,
        hidden_inject: np.ndarray,
        k_neighbors: int = 6
    ) -> np.ndarray:
        """
        Compute information bottleneck ratio per layer.
        
        Ratio = I(X; Z_inject) / I(X; Z_full)
        
        Parameters
        ----------
        input_samples : np.ndarray
            Input token embeddings. Shape: (n_samples, input_dim)
        hidden_full : np.ndarray
            Hidden states from full re-encoding.
            Shape: (num_layers, n_samples, hidden_dim)
        hidden_inject : np.ndarray
            Hidden states from SIG injection.
            Shape: (num_layers, n_samples, hidden_dim)
        k_neighbors : int, optional
            Number of neighbors for MI estimation.
            
        Returns
        -------
        np.ndarray
            Information bottleneck ratios per layer.
        """
        ratios = np.zeros(self.num_layers)
        
        for l in range(self.num_layers):
            mi_full = mutual_information_estimate(
                input_samples, hidden_full[l].reshape(hidden_full[l].shape[0], -1),
                k_neighbors
            )
            mi_inject = mutual_information_estimate(
                input_samples, hidden_inject[l].reshape(hidden_inject[l].shape[0], -1),
                k_neighbors
            )
            
            ratios[l] = mi_inject / (mi_full + 1e-12)
        
        return ratios
    
    def export_results(self, filepath: str) -> None:
        """
        Export analysis results to a JSON file.
        
        Parameters
        ----------
        filepath : str
            Path to save the results.
        """
        import json
        
        serializable_results = {}
        for key, value in self.results.items():
            if isinstance(value, np.ndarray):
                serializable_results[key] = value.tolist()
            elif isinstance(value, dict):
                serializable_results[key] = self._make_serializable(value)
            else:
                serializable_results[key] = value
        
        with open(filepath, 'w') as f:
            json.dump(serializable_results, f, indent=2)
    
    def _make_serializable(self, obj) -> any:
        """Recursively convert numpy types to Python native types."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._make_serializable(item) for item in obj]
        return obj
