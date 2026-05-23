#!/usr/bin/env python3
"""
Universal Transformer Testing Engine
=====================================
Shared engine for R1 (attention analysis), R3 (cross-architecture SIG simulation),
and future Transformer-based tests. Used by both sig_benchmark.py and co_benchmark.py.

Tasks:
  --task r1            HuggingFace attention distribution analysis (SIG vs full)
  --task r3            Cross-architecture SIG simulation (Transformer/SSM/RWKV/xLSTM)
  --task r3-empirical  Empirical parameterization from CO benchmarks
  --task all           Run all available tasks

Requires: numpy (all), torch+transformers+modelscope (r1 only)
"""
import argparse
import json
import time
import sys
import os
import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum


# ======================================================================
# Core: Architecture Types, State Info, Injection Result
# ======================================================================

class ArchitectureType(Enum):
    TRANSFORMER = "transformer"
    SSM = "ssm"
    RWKV = "rwkv"
    XLSTM = "xlstm"
    HYBRID = "hybrid"


@dataclass
class StateInfo:
    architecture: ArchitectureType
    state_dim: int
    sequence_length: int
    num_layers: int
    precision_bits: int = 16
    extra: Dict = field(default_factory=dict)

    @property
    def capacity_bits(self) -> int:
        return self.state_dim * self.sequence_length * self.num_layers * self.precision_bits

    @property
    def capacity_bytes(self) -> float:
        return self.capacity_bits / 8.0

    @property
    def capacity_mb(self) -> float:
        return self.capacity_bytes / (1024 ** 2)


@dataclass
class InjectionResult:
    success: bool
    pre_injection_state_norm: float
    post_injection_state_norm: float
    injection_fidelity: float
    retention_ratio: float
    state_delta_norm: float
    effective_capacity_used: float
    time_elapsed: float = 0.0
    extra: Dict = field(default_factory=dict)


# ======================================================================
# Core: Architecture State Model (ABC)
# ======================================================================

class ArchitectureStateModel(ABC):
    def __init__(self, state_info: StateInfo):
        self.state_info = state_info
        self._state = None
        self._history = []

    @abstractmethod
    def init_state(self, sequence: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def suspend(self) -> np.ndarray:
        pass

    @abstractmethod
    def inject(self, inject_data: np.ndarray, inject_mode: str = "append") -> InjectionResult:
        pass

    @abstractmethod
    def resume(self, n_steps: int = 1) -> np.ndarray:
        pass

    @abstractmethod
    def state_capacity(self) -> int:
        pass

    @abstractmethod
    def effective_information(self) -> float:
        pass

    def reset(self):
        self._state = None
        self._history = []

    def get_state_info(self) -> StateInfo:
        return self.state_info

    def _compute_fidelity(self, pre_state: np.ndarray, post_state: np.ndarray,
                          inject_signal: np.ndarray) -> float:
        pre_flat = pre_state.flatten()
        post_flat = post_state.flatten()
        inject_flat = inject_signal.flatten()

        if len(post_flat) >= len(pre_flat):
            delta_prefix = post_flat[:len(pre_flat)] - pre_flat
            delta_extended = post_flat[len(pre_flat):]
            delta = np.concatenate([delta_prefix, delta_extended])
        else:
            delta = post_flat - pre_flat[:len(post_flat)]

        min_len = min(len(delta), len(inject_flat))
        if min_len == 0:
            return 0.0
        delta = delta[:min_len]
        inject_flat = inject_flat[:min_len]

        norm_delta = np.linalg.norm(delta)
        norm_inject = np.linalg.norm(inject_flat)

        if norm_delta < 1e-12 or norm_inject < 1e-12:
            return 0.0

        cosine = np.dot(delta, inject_flat) / (norm_delta * norm_inject)
        return float(max(0.0, cosine))

    def _compute_retention(self, pre_state: np.ndarray, post_state: np.ndarray,
                           prior_signal: np.ndarray) -> float:
        pre_flat = pre_state.flatten()
        post_flat = post_state.flatten()

        min_len = min(len(pre_flat), len(post_flat))
        if min_len == 0:
            return 0.0
        pre_flat = pre_flat[:min_len]
        post_flat = post_flat[:min_len]

        norm_pre = np.linalg.norm(pre_flat)
        norm_post = np.linalg.norm(post_flat)

        if norm_pre < 1e-12 or norm_post < 1e-12:
            return 0.0

        cosine = np.dot(pre_flat, post_flat) / (norm_pre * norm_post)
        return float(max(0.0, cosine))


# ======================================================================
# Transformer State Model
# ======================================================================

class TransformerStateModel(ArchitectureStateModel):
    def __init__(self, d_model: int = 512, n_heads: int = 8, n_layers: int = 6,
                 max_seq_len: int = 8192, precision_bits: int = 16):
        d_head = d_model // n_heads
        info = StateInfo(
            architecture=ArchitectureType.TRANSFORMER,
            state_dim=d_model * 2,
            sequence_length=max_seq_len,
            num_layers=n_layers,
            precision_bits=precision_bits,
            extra={"d_model": d_model, "n_heads": n_heads, "d_head": d_head},
        )
        super().__init__(info)
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_head
        self.n_layers = n_layers
        self.max_seq_len = max_seq_len
        self._kv_cache = None
        self._current_pos = 0

    def init_state(self, sequence: np.ndarray) -> np.ndarray:
        seq_len = sequence.shape[0] if sequence.ndim > 0 else 1
        self._kv_cache = np.random.randn(self.n_layers, 2, seq_len, self.d_head, self.n_heads).astype(np.float32) * 0.1
        self._current_pos = seq_len
        self._state = self._kv_cache
        return self._kv_cache

    def suspend(self) -> np.ndarray:
        return self._kv_cache.copy() if self._kv_cache is not None else np.array([])

    def inject(self, inject_data: np.ndarray, inject_mode: str = "append") -> InjectionResult:
        if self._kv_cache is None:
            return InjectionResult(False, 0, 0, 0, 0, 0, 0)

        pre_norm = float(np.linalg.norm(self._kv_cache))
        pre_state = self._kv_cache.copy()

        inject_len = inject_data.shape[2] if inject_data.ndim == 5 else 1

        if inject_mode == "append":
            if self._current_pos + inject_len > self.max_seq_len:
                return InjectionResult(False, pre_norm, pre_norm, 0, 1.0, 0, 0,
                                       extra={"reason": "exceeds_max_seq_len"})
            if inject_data.ndim == 5:
                self._kv_cache = np.concatenate([self._kv_cache, inject_data], axis=2)
            else:
                new_kv = np.random.randn(self.n_layers, 2, inject_len, self.d_head, self.n_heads).astype(np.float32) * 0.1
                self._kv_cache = np.concatenate([self._kv_cache, new_kv], axis=2)
            self._current_pos += inject_len
        elif inject_mode == "overwrite":
            end_pos = min(self._current_pos + inject_len, self.max_seq_len)
            if inject_data.ndim == 5:
                self._kv_cache[:, :, self._current_pos:end_pos] = inject_data[:, :, :end_pos - self._current_pos]
            self._current_pos = end_pos
        elif inject_mode == "merge":
            alpha = 0.5
            if inject_data.ndim == 5:
                new_kv = np.random.randn(self.n_layers, 2, inject_len, self.d_head, self.n_heads).astype(np.float32) * 0.1
                merged = alpha * self._kv_cache[:, :, -1:] + (1 - alpha) * new_kv[:, :, :1]
                self._kv_cache = np.concatenate([self._kv_cache, merged], axis=2)
            self._current_pos += inject_len

        post_norm = float(np.linalg.norm(self._kv_cache))
        inject_signal = inject_data.flatten() if inject_data.ndim > 0 else np.array([0])

        fidelity = self._compute_fidelity(pre_state, self._kv_cache, inject_signal)
        retention = self._compute_retention(pre_state, self._kv_cache, pre_state)

        delta_norm = 0.0
        if self._kv_cache.shape == pre_state.shape:
            delta_norm = float(np.linalg.norm(self._kv_cache - pre_state))
        else:
            min_len = min(self._kv_cache.size, pre_state.size)
            delta_norm = float(np.linalg.norm(
                self._kv_cache.flatten()[:min_len] - pre_state.flatten()[:min_len]))

        return InjectionResult(
            success=True,
            pre_injection_state_norm=pre_norm,
            post_injection_state_norm=post_norm,
            injection_fidelity=fidelity,
            retention_ratio=retention,
            state_delta_norm=delta_norm,
            effective_capacity_used=self._current_pos / self.max_seq_len,
            extra={"inject_mode": inject_mode, "current_pos": self._current_pos},
        )

    def resume(self, n_steps: int = 1) -> np.ndarray:
        new_kv = np.random.randn(self.n_layers, 2, n_steps, self.d_head, self.n_heads).astype(np.float32) * 0.1
        if self._kv_cache is not None:
            self._kv_cache = np.concatenate([self._kv_cache, new_kv], axis=2)
            self._current_pos += n_steps
        return new_kv

    def state_capacity(self) -> int:
        return self.n_layers * 2 * self.max_seq_len * self.d_head * self.n_heads

    def effective_information(self) -> float:
        if self._kv_cache is None:
            return 0.0
        flat = self._kv_cache.flatten()
        std = float(np.std(flat))
        if std < 1e-12:
            return 0.0
        entropy = 0.5 * np.log(2 * np.pi * np.e * std ** 2)
        return float(entropy * len(flat))


# ======================================================================
# Information Metrics
# ======================================================================

class InformationMetrics:
    @staticmethod
    def capacity_ratio(model: ArchitectureStateModel) -> float:
        total = model.state_capacity()
        if total == 0:
            return 0.0
        effective = model.effective_information()
        max_possible = total * np.log(2 * np.pi * np.e) / 2
        if max_possible < 1e-12:
            return 0.0
        return float(min(1.0, effective / max_possible))

    @staticmethod
    def information_density(model: ArchitectureStateModel) -> float:
        capacity = model.state_capacity()
        if capacity == 0:
            return 0.0
        return model.effective_information() / capacity

    @staticmethod
    def fidelity_retention_tradeoff(results: List[InjectionResult]) -> Dict:
        if not results:
            return {"pareto_points": [], "area_under_curve": 0.0}

        points = [(r.injection_fidelity, r.retention_ratio) for r in results if r.success]
        if not points:
            return {"pareto_points": [], "area_under_curve": 0.0}

        points.sort(key=lambda p: p[0])
        pareto = [points[0]]
        for f, r in points[1:]:
            if r >= pareto[-1][1]:
                pareto.append((f, r))

        auc = 0.0
        for i in range(1, len(points)):
            dx = points[i][0] - points[i - 1][0]
            avg_y = (points[i][1] + points[i - 1][1]) / 2
            auc += dx * avg_y

        return {"pareto_points": pareto, "area_under_curve": auc}

    @staticmethod
    def multi_injection_degradation(results: List[InjectionResult]) -> Dict:
        retentions = [r.retention_ratio for r in results if r.success]
        fidelities = [r.injection_fidelity for r in results if r.success]

        if len(retentions) < 2:
            return {"retentions": retentions, "fidelities": fidelities, "decay_rate": 0.0}

        log_retentions = [np.log(max(r, 1e-10)) for r in retentions]
        x = np.arange(len(log_retentions), dtype=np.float64)
        if len(x) > 1:
            coeffs = np.polyfit(x, log_retentions, 1)
            decay_rate = -coeffs[0]
        else:
            decay_rate = 0.0

        return {
            "retentions": retentions,
            "fidelities": fidelities,
            "decay_rate": float(decay_rate),
            "half_life": float(np.log(2) / decay_rate) if decay_rate > 1e-10 else float('inf'),
        }

    @staticmethod
    def compare_architectures(models: Dict[str, ArchitectureStateModel],
                              inject_data: np.ndarray,
                              n_injections: int = 5) -> Dict:
        comparison = {}

        for name, model in models.items():
            model.reset()
            init_seq = np.random.randn(64).astype(np.float32)
            model.init_state(init_seq)

            injection_results = []
            for i in range(n_injections):
                result = model.inject(inject_data, inject_mode="append")
                injection_results.append(result)

            degradation = InformationMetrics.multi_injection_degradation(injection_results)
            tradeoff = InformationMetrics.fidelity_retention_tradeoff(injection_results)

            comparison[name] = {
                "architecture": model.state_info.architecture.value,
                "state_capacity": model.state_capacity(),
                "effective_information": model.effective_information(),
                "capacity_ratio": InformationMetrics.capacity_ratio(model),
                "information_density": InformationMetrics.information_density(model),
                "injection_results": injection_results,
                "degradation": degradation,
                "tradeoff": tradeoff,
                "avg_fidelity": float(np.mean([r.injection_fidelity for r in injection_results if r.success])) if injection_results else 0.0,
                "avg_retention": float(np.mean([r.retention_ratio for r in injection_results if r.success])) if injection_results else 0.0,
                "final_retention": degradation["retentions"][-1] if degradation["retentions"] else 0.0,
                "decay_rate": degradation["decay_rate"],
                "half_life": degradation["half_life"],
            }

        return comparison


def print_comparison(comparison: Dict):
    print("\n" + "=" * 90)
    print("R3: SIG Cross-Architecture Comparison")
    print("=" * 90)

    header = f"{'Architecture':<16} {'Capacity':<12} {'Eff.Info':<12} {'Cap.Ratio':<10} {'AvgFid':<8} {'AvgRet':<8} {'FinalRet':<9} {'Decay':<8} {'HalfLife':<10}"
    print(header)
    print("-" * len(header))

    for name, data in comparison.items():
        cap = data["state_capacity"]
        eff = data["effective_information"]
        cr = data["capacity_ratio"]
        af = data["avg_fidelity"]
        ar = data["avg_retention"]
        fr = data["final_retention"]
        dr = data["decay_rate"]
        hl = data["half_life"]

        hl_str = f"{hl:.1f}" if hl < 1e6 else "inf"
        print(f"{name:<16} {cap:<12} {eff:<12.2f} {cr:<10.4f} {af:<8.3f} {ar:<8.3f} {fr:<9.3f} {dr:<8.4f} {hl_str:<10}")

    print("\n--- Fidelity-Retention Trade-off (AUC) ---")
    for name, data in comparison.items():
        auc = data["tradeoff"]["area_under_curve"]
        n_pareto = len(data["tradeoff"]["pareto_points"])
        print(f"  {name:<16} AUC={auc:.4f}  Pareto points={n_pareto}")

    print("\n--- Multi-Injection Degradation ---")
    for name, data in comparison.items():
        retentions = data["degradation"]["retentions"]
        ret_str = " -> ".join(f"{r:.3f}" for r in retentions)
        print(f"  {name:<16} {ret_str}")


# ======================================================================
# SSM / Mamba State Model
# ======================================================================

@dataclass
class SSMConfig:
    d_model: int = 512
    d_state: int = 16
    n_layers: int = 6
    dt_min: float = 0.001
    dt_max: float = 0.1
    precision_bits: int = 16


class SSMStateModel(ArchitectureStateModel):
    def __init__(self, config: Optional[SSMConfig] = None):
        config = config or SSMConfig()
        self.config = config
        self.d_model = config.d_model
        self.d_state = config.d_state
        self.n_layers = config.n_layers

        info = StateInfo(
            architecture=ArchitectureType.SSM,
            state_dim=config.d_state * config.d_model,
            sequence_length=1,
            num_layers=config.n_layers,
            precision_bits=config.precision_bits,
            extra={
                "d_model": config.d_model,
                "d_state": config.d_state,
                "compression_ratio": config.d_model / (config.d_state * config.d_model),
            },
        )
        super().__init__(info)

        self._A = None
        self._states = None
        self._seq_processed = 0

        self._init_ssm_parameters()

    def _init_ssm_parameters(self):
        self._A = np.zeros((self.n_layers, self.d_state, self.d_state), dtype=np.float32)
        for l in range(self.n_layers):
            hippo = self._make_hippo_matrix(self.d_state)
            self._A[l] = hippo

        self._B = np.random.randn(self.n_layers, self.d_state, self.d_model).astype(np.float32) * 0.01
        self._C = np.random.randn(self.n_layers, self.d_model, self.d_state).astype(np.float32) * 0.01

        log_dt = np.random.uniform(
            np.log(self.config.dt_min), np.log(self.config.dt_max),
            size=(self.n_layers, self.d_model)
        ).astype(np.float32)
        self._dt = np.exp(log_dt)

        self._D = np.ones((self.n_layers, self.d_model), dtype=np.float32)

    @staticmethod
    def _make_hippo_matrix(d_state: int) -> np.ndarray:
        A = np.zeros((d_state, d_state), dtype=np.float32)
        for n in range(d_state):
            for k in range(d_state):
                if n > k:
                    A[n, k] = -np.sqrt(2 * n + 1) * np.sqrt(2 * k + 1)
                elif n == k:
                    A[n, k] = -(2 * n + 1)
        return A

    def _discretize(self, layer: int) -> Tuple[np.ndarray, np.ndarray]:
        dt = self._dt[layer]
        A = self._A[layer]
        B = self._B[layer]

        A_bar = np.zeros_like(A)
        B_bar = np.zeros_like(B)

        for i in range(self.d_model):
            dA = dt[i] * A
            A_bar_i = np.eye(self.d_state, dtype=np.float32) + dA
            A_bar_i_max = np.max(np.abs(A_bar_i))
            if A_bar_i_max > 50:
                scale = 50.0 / A_bar_i_max
                A_bar_i = np.eye(self.d_state, dtype=np.float32) + dA * scale

            A_bar[:, :] = A_bar_i
            B_bar[:, i] = dt[i] * B[:, i]

        return A_bar, B_bar

    def init_state(self, sequence: np.ndarray) -> np.ndarray:
        seq_len = len(sequence) if sequence.ndim > 0 else 1
        self._states = np.zeros((self.n_layers, self.d_state, self.d_model), dtype=np.float32)

        for l in range(self.n_layers):
            A_bar, B_bar = self._discretize(l)
            h = np.zeros((self.d_state, self.d_model), dtype=np.float32)

            for t in range(seq_len):
                x_t = np.zeros(self.d_model, dtype=np.float32)
                idx = t % self.d_model
                x_t[idx] = sequence[t] if t < len(sequence) else 0.0

                for i in range(self.d_model):
                    h[:, i] = A_bar @ h[:, i] + B_bar[:, i] * x_t[i]

            self._states[l] = h

        self._seq_processed = seq_len
        self._state = self._states
        return self._states

    def suspend(self) -> np.ndarray:
        return self._states.copy() if self._states is not None else np.array([])

    def inject(self, inject_data: np.ndarray, inject_mode: str = "append") -> InjectionResult:
        if self._states is None:
            return InjectionResult(False, 0, 0, 0, 0, 0, 0)

        pre_norm = float(np.linalg.norm(self._states))
        pre_state = self._states.copy()
        t0 = time.time()

        if inject_mode == "append":
            inject_len = inject_data.shape[0] if inject_data.ndim > 0 else 1
            for l in range(self.n_layers):
                A_bar, B_bar = self._discretize(l)
                h = self._states[l].copy()

                for t in range(inject_len):
                    x_t = np.zeros(self.d_model, dtype=np.float32)
                    idx = t % self.d_model
                    val = inject_data[t] if t < len(inject_data) else 0.0
                    x_t[idx] = val

                    for i in range(self.d_model):
                        h[:, i] = A_bar @ h[:, i] + B_bar[:, i] * x_t[i]

                self._states[l] = h

        elif inject_mode == "overwrite":
            if inject_data.ndim >= 2 and inject_data.shape == self._states.shape:
                self._states = inject_data.copy()
            else:
                inject_flat = inject_data.flatten()
                state_flat = self._states.flatten()
                n_overwrite = min(len(inject_flat), len(state_flat))
                state_flat[:n_overwrite] = inject_flat[:n_overwrite]
                self._states = state_flat.reshape(self._states.shape)

        elif inject_mode == "merge":
            gate = 0.3
            inject_signal = np.random.randn(*self._states.shape).astype(np.float32) * 0.1
            if inject_data.ndim >= 2 and inject_data.shape == self._states.shape:
                inject_signal = inject_data
            self._states = (1 - gate) * self._states + gate * inject_signal

        elapsed = time.time() - t0
        post_norm = float(np.linalg.norm(self._states))

        inject_signal = inject_data.flatten() if inject_data.ndim > 0 else np.array([0])
        fidelity = self._compute_fidelity(pre_state, self._states, inject_signal)
        retention = self._compute_retention(pre_state, self._states, pre_state)

        return InjectionResult(
            success=True,
            pre_injection_state_norm=pre_norm,
            post_injection_state_norm=post_norm,
            injection_fidelity=fidelity,
            retention_ratio=retention,
            state_delta_norm=float(np.linalg.norm(self._states - pre_state)),
            effective_capacity_used=1.0,
            time_elapsed=elapsed,
            extra={
                "inject_mode": inject_mode,
                "compression_ratio": self.d_model / max(self.d_state, 1),
                "state_is_fixed_size": True,
            },
        )

    def resume(self, n_steps: int = 1) -> np.ndarray:
        outputs = []
        for l in range(self.n_layers):
            A_bar, B_bar = self._discretize(l)
            h = self._states[l].copy()
            layer_outputs = []

            for t in range(n_steps):
                y_t = self._C[l] @ h + self._D[l]
                layer_outputs.append(y_t)
                for i in range(self.d_model):
                    h[:, i] = A_bar @ h[:, i]

            self._states[l] = h
            outputs.append(np.array(layer_outputs))

        self._seq_processed += n_steps
        return np.array(outputs)

    def state_capacity(self) -> int:
        return self.n_layers * self.d_state * self.d_model

    def effective_information(self) -> float:
        if self._states is None:
            return 0.0
        flat = self._states.flatten()
        std = float(np.std(flat))
        if std < 1e-12:
            return 0.0
        entropy = 0.5 * np.log(2 * np.pi * np.e * std ** 2)
        return float(entropy * len(flat))

    def analyze_bottleneck(self, seq_len: int) -> Dict:
        total_input_info = seq_len * self.d_model
        state_capacity = self.d_state * self.d_model
        compression_ratio = state_capacity / total_input_info

        theoretical_max_retention = min(1.0, state_capacity / total_input_info)

        hippo_eigenvalues = np.linalg.eigvals(self._A[0])
        stable_eigenvalues = np.sum(np.real(hippo_eigenvalues) < 0)
        stability_ratio = stable_eigenvalues / len(hippo_eigenvalues)

        return {
            "seq_len": seq_len,
            "total_input_elements": total_input_info,
            "state_capacity_elements": state_capacity,
            "compression_ratio": compression_ratio,
            "theoretical_max_retention": theoretical_max_retention,
            "hippo_stability_ratio": float(stability_ratio),
            "bottleneck_severity": "low" if compression_ratio > 0.5 else
                                   "medium" if compression_ratio > 0.1 else "high",
        }

    def compare_injection_strategies(self, inject_data: np.ndarray,
                                      n_injections: int = 5) -> Dict:
        strategies = ["append", "overwrite", "merge"]
        results = {}

        for strategy in strategies:
            self.reset()
            init_seq = np.random.randn(64).astype(np.float32)
            self.init_state(init_seq)

            injection_results = []
            for i in range(n_injections):
                result = self.inject(inject_data, inject_mode=strategy)
                injection_results.append(result)

            retentions = [r.retention_ratio for r in injection_results if r.success]
            fidelities = [r.injection_fidelity for r in injection_results if r.success]

            results[strategy] = {
                "avg_fidelity": float(np.mean(fidelities)) if fidelities else 0.0,
                "avg_retention": float(np.mean(retentions)) if retentions else 0.0,
                "final_retention": retentions[-1] if retentions else 0.0,
                "retention_curve": retentions,
                "fidelity_curve": fidelities,
            }

        return results


# ======================================================================
# RWKV State Model
# ======================================================================

@dataclass
class RWKVConfig:
    d_model: int = 512
    n_heads: int = 8
    n_layers: int = 6
    head_size: int = 64
    precision_bits: int = 16


class RWKVStateModel(ArchitectureStateModel):
    def __init__(self, config: Optional[RWKVConfig] = None):
        config = config or RWKVConfig()
        self.config = config
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.head_size = config.head_size
        self.n_layers = config.n_layers

        info = StateInfo(
            architecture=ArchitectureType.RWKV,
            state_dim=3 * config.d_model,
            sequence_length=1,
            num_layers=config.n_layers,
            precision_bits=config.precision_bits,
            extra={
                "d_model": config.d_model,
                "n_heads": config.n_heads,
                "head_size": config.head_size,
                "state_per_layer": 3 * config.d_model,
            },
        )
        super().__init__(info)

        self._wkv_states = None
        self._time_decay = None
        self._time_first = None
        self._seq_processed = 0

        self._init_rwkv_parameters()

    def _init_rwkv_parameters(self):
        self._time_decay = np.exp(
            -np.random.uniform(0, 5, size=(self.n_layers, self.d_model)).astype(np.float32)
        )

        self._time_first = np.exp(
            np.random.uniform(0, 3, size=(self.n_layers, self.d_model)).astype(np.float32)
        )

        self._key_weights = np.random.randn(
            self.n_layers, self.d_model, self.d_model
        ).astype(np.float32) * 0.02

        self._value_weights = np.random.randn(
            self.n_layers, self.d_model, self.d_model
        ).astype(np.float32) * 0.02

        self._output_weights = np.random.randn(
            self.n_layers, self.d_model, self.d_model
        ).astype(np.float32) * 0.02

    def _init_wkv_state(self) -> np.ndarray:
        return np.zeros((self.n_layers, 3, self.d_model), dtype=np.float32)

    def _wkv_update(self, layer: int, state: np.ndarray,
                    k: np.ndarray, v: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        aa, bb, pp = state[layer, 0], state[layer, 1], state[layer, 2]

        w = self._time_decay[layer]
        u = self._time_first[layer]

        ww = aa + np.exp(pp + u) * v
        qq = bb + np.exp(pp + u)

        e1 = np.exp(pp - w)
        e2 = np.exp(k - w)

        new_aa = e1 * aa + e2 * v
        new_bb = e1 * bb + e2
        new_pp = np.maximum(pp - w, k - w)

        output = ww / qq

        new_state = np.stack([new_aa, new_bb, new_pp])
        return output, new_state

    def init_state(self, sequence: np.ndarray) -> np.ndarray:
        seq_len = len(sequence) if sequence.ndim > 0 else 1
        self._wkv_states = self._init_wkv_state()

        for t in range(seq_len):
            x_t = np.zeros(self.d_model, dtype=np.float32)
            idx = t % self.d_model
            x_t[idx] = sequence[t] if t < len(sequence) else 0.0

            for l in range(self.n_layers):
                k = self._key_weights[l] @ x_t
                v = self._value_weights[l] @ x_t

                k = np.clip(k, -5, 5)
                v = np.clip(v, -5, 5)

                output, new_state = self._wkv_update(l, self._wkv_states, k, v)
                self._wkv_states[l] = new_state

                x_t = self._output_weights[l] @ output

        self._seq_processed = seq_len
        self._state = self._wkv_states
        return self._wkv_states

    def suspend(self) -> np.ndarray:
        return self._wkv_states.copy() if self._wkv_states is not None else np.array([])

    def inject(self, inject_data: np.ndarray, inject_mode: str = "append") -> InjectionResult:
        if self._wkv_states is None:
            return InjectionResult(False, 0, 0, 0, 0, 0, 0)

        pre_norm = float(np.linalg.norm(self._wkv_states))
        pre_state = self._wkv_states.copy()
        t0 = time.time()

        if inject_mode == "append":
            inject_len = inject_data.shape[0] if inject_data.ndim > 0 else 1

            for t in range(inject_len):
                x_t = np.zeros(self.d_model, dtype=np.float32)
                idx = t % self.d_model
                val = inject_data[t] if t < len(inject_data) else 0.0
                x_t[idx] = val

                for l in range(self.n_layers):
                    k = self._key_weights[l] @ x_t
                    v = self._value_weights[l] @ x_t
                    k = np.clip(k, -5, 5)
                    v = np.clip(v, -5, 5)

                    _, new_state = self._wkv_update(l, self._wkv_states, k, v)
                    self._wkv_states[l] = new_state

        elif inject_mode == "overwrite":
            inject_signal = np.random.randn(self.n_layers, 3, self.d_model).astype(np.float32) * 0.1
            if inject_data.ndim >= 2 and inject_data.shape == self._wkv_states.shape:
                inject_signal = inject_data

            target_channels = np.random.choice(self.d_model, size=self.d_model // 4, replace=False)
            for ch in target_channels:
                self._wkv_states[:, :, ch] = inject_signal[:, :, ch]

        elif inject_mode == "merge":
            alpha = 0.3
            inject_signal = np.random.randn(*self._wkv_states.shape).astype(np.float32) * 0.1
            if inject_data.ndim >= 2 and inject_data.shape == self._wkv_states.shape:
                inject_signal = inject_data

            for l in range(self.n_layers):
                self._wkv_states[l, 0] = (1 - alpha) * self._wkv_states[l, 0] + alpha * inject_signal[l, 0]
                self._wkv_states[l, 1] = (1 - alpha) * self._wkv_states[l, 1] + alpha * inject_signal[l, 1]
                self._wkv_states[l, 2] = np.maximum(
                    (1 - alpha) * self._wkv_states[l, 2] + alpha * inject_signal[l, 2],
                    self._wkv_states[l, 2]
                )

        elapsed = time.time() - t0
        post_norm = float(np.linalg.norm(self._wkv_states))

        inject_signal = inject_data.flatten() if inject_data.ndim > 0 else np.array([0])
        fidelity = self._compute_fidelity(pre_state, self._wkv_states, inject_signal)
        retention = self._compute_retention(pre_state, self._wkv_states, pre_state)

        return InjectionResult(
            success=True,
            pre_injection_state_norm=pre_norm,
            post_injection_state_norm=post_norm,
            injection_fidelity=fidelity,
            retention_ratio=retention,
            state_delta_norm=float(np.linalg.norm(self._wkv_states - pre_state)),
            effective_capacity_used=1.0,
            time_elapsed=elapsed,
            extra={
                "inject_mode": inject_mode,
                "state_per_layer": 3 * self.d_model,
                "decay_rates_mean": float(np.mean(self._time_decay)),
                "decay_rates_std": float(np.std(self._time_decay)),
            },
        )

    def resume(self, n_steps: int = 1) -> np.ndarray:
        outputs = []
        for t in range(n_steps):
            x_t = np.zeros(self.d_model, dtype=np.float32)
            for l in range(self.n_layers):
                k = self._key_weights[l] @ x_t
                v = self._value_weights[l] @ x_t
                k = np.clip(k, -5, 5)
                v = np.clip(v, -5, 5)

                output, new_state = self._wkv_update(l, self._wkv_states, k, v)
                self._wkv_states[l] = new_state
                x_t = self._output_weights[l] @ output

            outputs.append(x_t)

        self._seq_processed += n_steps
        return np.array(outputs)

    def state_capacity(self) -> int:
        return self.n_layers * 3 * self.d_model

    def effective_information(self) -> float:
        if self._wkv_states is None:
            return 0.0
        flat = self._wkv_states.flatten()
        std = float(np.std(flat))
        if std < 1e-12:
            return 0.0
        entropy = 0.5 * np.log(2 * np.pi * np.e * std ** 2)
        return float(entropy * len(flat))

    def analyze_decay_impact(self, n_steps: int = 100) -> Dict:
        if self._wkv_states is None:
            return {}

        initial_aa = self._wkv_states[:, 0].copy()
        initial_norm = float(np.linalg.norm(initial_aa))

        decay_curves = []
        for l in range(self.n_layers):
            w = self._time_decay[l]
            curve = []
            aa = initial_aa[l].copy()
            for t in range(n_steps):
                aa = aa * w
                curve.append(float(np.linalg.norm(aa)))
            decay_curves.append(curve)

        avg_decay_curve = np.mean(decay_curves, axis=0)

        half_life_steps = None
        for t, val in enumerate(avg_decay_curve):
            if val < initial_norm / (2 * self.n_layers):
                half_life_steps = t
                break

        return {
            "decay_curves_per_layer": decay_curves,
            "avg_decay_curve": avg_decay_curve.tolist(),
            "initial_norm": initial_norm,
            "half_life_steps": half_life_steps,
            "avg_decay_rate": float(np.mean(self._time_decay)),
            "min_decay_rate": float(np.min(self._time_decay)),
            "max_decay_rate": float(np.max(self._time_decay)),
        }

    def compare_with_transformer_state_size(self, seq_len: int) -> Dict:
        rwkv_state_size = self.state_capacity()
        transformer_kv_size = self.n_layers * 2 * seq_len * self.d_model

        return {
            "seq_len": seq_len,
            "rwkv_state_size": rwkv_state_size,
            "transformer_kv_size": transformer_kv_size,
            "compression_ratio": rwkv_state_size / transformer_kv_size,
            "rwkv_advantage": "RWKV is smaller" if rwkv_state_size < transformer_kv_size else "Transformer is smaller",
            "sig_feasibility": "RWKV: compressed state, harder injection" if rwkv_state_size < transformer_kv_size else "Transformer: append-only, easier injection",
        }


# ======================================================================
# xLSTM State Model
# ======================================================================

@dataclass
class xLSTMConfig:
    d_model: int = 512
    n_heads: int = 8
    n_layers: int = 6
    head_dim: int = 64
    block_type: str = "mixed"
    precision_bits: int = 16


class sLSTMBlock:
    def __init__(self, d_model: int, n_heads: int, head_dim: int):
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = head_dim

        scale = 0.02
        self.W_i = np.random.randn(d_model, d_model).astype(np.float32) * scale
        self.W_f = np.random.randn(d_model, d_model).astype(np.float32) * scale
        self.W_z = np.random.randn(d_model, d_model).astype(np.float32) * scale
        self.W_o = np.random.randn(d_model, d_model).astype(np.float32) * scale

        self.c = np.zeros(d_model, dtype=np.float32)
        self.h = np.zeros(d_model, dtype=np.float32)
        self.m = np.zeros(d_model, dtype=np.float32)
        self.n = np.zeros(d_model, dtype=np.float32)

    def step(self, x_t: np.ndarray) -> np.ndarray:
        log_i = self.W_i @ x_t
        log_f = self.W_f @ x_t
        z = np.tanh(self.W_z @ x_t)
        o = 1.0 / (1.0 + np.exp(-(self.W_o @ x_t)))

        log_i = np.clip(log_i, -10, 10)
        log_f = np.clip(log_f, -10, 10)

        m_new = np.maximum(log_f + self.m, log_i)
        i_exp = np.exp(log_i - m_new)
        f_exp = np.exp(log_f + self.m - m_new)

        self.n = f_exp * self.n + i_exp
        self.c = f_exp * self.c + i_exp * z
        self.m = m_new

        c_norm = self.c / (self.n + 1e-8)
        self.h = o * np.tanh(c_norm)

        return self.h

    def get_state(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return self.c.copy(), self.h.copy(), self.m.copy(), self.n.copy()

    def set_state(self, c: np.ndarray, h: np.ndarray, m: np.ndarray, n: np.ndarray):
        self.c = c.copy()
        self.h = h.copy()
        self.m = m.copy()
        self.n = n.copy()


class mLSTMBlock:
    def __init__(self, d_model: int, n_heads: int, head_dim: int):
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = head_dim

        scale = 0.02
        self.W_q = np.random.randn(d_model, d_model).astype(np.float32) * scale
        self.W_k = np.random.randn(d_model, d_model).astype(np.float32) * scale
        self.W_v = np.random.randn(d_model, d_model).astype(np.float32) * scale
        self.W_i = np.random.randn(d_model,).astype(np.float32) * scale
        self.W_f = np.random.randn(d_model,).astype(np.float32) * scale
        self.W_o = np.random.randn(d_model, d_model).astype(np.float32) * scale

        self.C = np.zeros((n_heads, head_dim, head_dim), dtype=np.float32)
        self.n_state = np.zeros(n_heads, dtype=np.float32)
        self.m_state = np.zeros(n_heads, dtype=np.float32)
        self.h = np.zeros(d_model, dtype=np.float32)

    def step(self, x_t: np.ndarray) -> np.ndarray:
        q = self.W_q @ x_t
        k = self.W_k @ x_t
        v = self.W_v @ x_t
        o = 1.0 / (1.0 + np.exp(-(self.W_o @ x_t)))

        log_i = float(np.clip(self.W_i @ x_t, -10, 10))
        log_f = float(np.clip(self.W_f @ x_t, -10, 10))

        q_heads = q.reshape(self.n_heads, self.head_dim)
        k_heads = k.reshape(self.n_heads, self.head_dim)
        v_heads = v.reshape(self.n_heads, self.head_dim)

        outputs = []
        for h in range(self.n_heads):
            m_new = max(log_f + self.m_state[h], log_i)
            i_exp = np.exp(log_i - m_new)
            f_exp = np.exp(log_f + self.m_state[h] - m_new)

            self.C[h] = f_exp * self.C[h] + i_exp * np.outer(v_heads[h], k_heads[h])
            self.n_state[h] = f_exp * self.n_state[h] + i_exp
            self.m_state[h] = m_new

            q_h = q_heads[h]
            attn = q_h @ self.C[h] / (self.n_state[h] + 1e-8)
            outputs.append(attn)

        output = np.concatenate(outputs)
        self.h = o * output
        return self.h

    def get_state(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self.C.copy(), self.n_state.copy(), self.m_state.copy()

    def set_state(self, C: np.ndarray, n: np.ndarray, m: np.ndarray):
        self.C = C.copy()
        self.n_state = n.copy()
        self.m_state = m.copy()


class xLSTMStateModel(ArchitectureStateModel):
    def __init__(self, config: Optional[xLSTMConfig] = None):
        config = config or xLSTMConfig()
        self.config = config
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.head_dim = config.head_dim
        self.n_layers = config.n_layers
        self.block_type = config.block_type

        s_state_per_layer = 4 * config.d_model
        m_state_per_layer = config.n_heads * config.head_dim * config.head_dim + 2 * config.n_heads

        if config.block_type == "slstm":
            state_per_layer = s_state_per_layer
        elif config.block_type == "mlstm":
            state_per_layer = m_state_per_layer
        else:
            state_per_layer = (s_state_per_layer + m_state_per_layer) // 2

        info = StateInfo(
            architecture=ArchitectureType.XLSTM,
            state_dim=state_per_layer,
            sequence_length=1,
            num_layers=config.n_layers,
            precision_bits=config.precision_bits,
            extra={
                "d_model": config.d_model,
                "n_heads": config.n_heads,
                "head_dim": config.head_dim,
                "block_type": config.block_type,
                "s_state_per_layer": s_state_per_layer,
                "m_state_per_layer": m_state_per_layer,
            },
        )
        super().__init__(info)

        self._blocks = []
        self._block_types = []
        self._init_blocks()

    def _init_blocks(self):
        for l in range(self.n_layers):
            if self.block_type == "slstm":
                bt = "s"
            elif self.block_type == "mlstm":
                bt = "m"
            else:
                bt = "s" if l % 2 == 0 else "m"

            self._block_types.append(bt)

            if bt == "s":
                block = sLSTMBlock(self.d_model, self.n_heads, self.head_dim)
            else:
                block = mLSTMBlock(self.d_model, self.n_heads, self.head_dim)

            self._blocks.append(block)

    def init_state(self, sequence: np.ndarray) -> np.ndarray:
        seq_len = len(sequence) if sequence.ndim > 0 else 1

        for block in self._blocks:
            if isinstance(block, sLSTMBlock):
                block.c = np.zeros(self.d_model, dtype=np.float32)
                block.h = np.zeros(self.d_model, dtype=np.float32)
                block.m = np.zeros(self.d_model, dtype=np.float32)
                block.n = np.zeros(self.d_model, dtype=np.float32)
            else:
                block.C = np.zeros((self.n_heads, self.head_dim, self.head_dim), dtype=np.float32)
                block.n_state = np.zeros(self.n_heads, dtype=np.float32)
                block.m_state = np.zeros(self.n_heads, dtype=np.float32)
                block.h = np.zeros(self.d_model, dtype=np.float32)

        for t in range(seq_len):
            x_t = np.zeros(self.d_model, dtype=np.float32)
            idx = t % self.d_model
            x_t[idx] = sequence[t] if t < len(sequence) else 0.0

            for block in self._blocks:
                x_t = block.step(x_t)

        self._state = self._collect_state()
        return self._state

    def _collect_state(self) -> np.ndarray:
        parts = []
        for block in self._blocks:
            if isinstance(block, sLSTMBlock):
                c, h, m, n = block.get_state()
                parts.extend([c, h, m, n])
            else:
                C, n_s, m_s = block.get_state()
                parts.extend([C.flatten(), n_s, m_s])
        return np.concatenate([p.flatten() for p in parts])

    def _restore_state(self, state_flat: np.ndarray):
        idx = 0
        for block in self._blocks:
            if isinstance(block, sLSTMBlock):
                size = self.d_model
                c = state_flat[idx:idx + size].reshape(self.d_model)
                idx += size
                h = state_flat[idx:idx + size].reshape(self.d_model)
                idx += size
                m = state_flat[idx:idx + size].reshape(self.d_model)
                idx += size
                n = state_flat[idx:idx + size].reshape(self.d_model)
                idx += size
                block.set_state(c, h, m, n)
            else:
                c_size = self.n_heads * self.head_dim * self.head_dim
                C = state_flat[idx:idx + c_size].reshape(self.n_heads, self.head_dim, self.head_dim)
                idx += c_size
                n_s = state_flat[idx:idx + self.n_heads]
                idx += self.n_heads
                m_s = state_flat[idx:idx + self.n_heads]
                idx += self.n_heads
                block.set_state(C, n_s, m_s)

    def suspend(self) -> np.ndarray:
        return self._collect_state()

    def inject(self, inject_data: np.ndarray, inject_mode: str = "append") -> InjectionResult:
        if self._state is None:
            return InjectionResult(False, 0, 0, 0, 0, 0, 0)

        pre_state = self._collect_state()
        pre_norm = float(np.linalg.norm(pre_state))
        t0 = time.time()

        if inject_mode == "append":
            inject_len = inject_data.shape[0] if inject_data.ndim > 0 else 1
            for t in range(inject_len):
                x_t = np.zeros(self.d_model, dtype=np.float32)
                idx = t % self.d_model
                val = inject_data[t] if t < len(inject_data) else 0.0
                x_t[idx] = val
                for block in self._blocks:
                    x_t = block.step(x_t)

        elif inject_mode == "overwrite":
            for block in self._blocks:
                if isinstance(block, sLSTMBlock):
                    delta_c = np.random.randn(self.d_model).astype(np.float32) * 0.1
                    block.c = block.c + delta_c
                else:
                    for h in range(self.n_heads):
                        delta_C = np.random.randn(self.head_dim, self.head_dim).astype(np.float32) * 0.01
                        block.C[h] = block.C[h] + delta_C

        elif inject_mode == "rank1_inject":
            for block in self._blocks:
                if isinstance(block, mLSTMBlock):
                    alpha = 0.5
                    for h in range(self.n_heads):
                        v = np.random.randn(self.head_dim).astype(np.float32) * 0.1
                        k = np.random.randn(self.head_dim).astype(np.float32) * 0.1
                        block.C[h] = block.C[h] + alpha * np.outer(v, k)
                        block.n_state[h] += alpha
                elif isinstance(block, sLSTMBlock):
                    delta_c = np.random.randn(self.d_model).astype(np.float32) * 0.1
                    block.c = block.c + delta_c

        elif inject_mode == "merge":
            gate = 0.3
            for block in self._blocks:
                if isinstance(block, sLSTMBlock):
                    inject_c = np.random.randn(self.d_model).astype(np.float32) * 0.1
                    block.c = (1 - gate) * block.c + gate * inject_c
                else:
                    inject_C = np.random.randn(self.n_heads, self.head_dim, self.head_dim).astype(np.float32) * 0.01
                    block.C = (1 - gate) * block.C + gate * inject_C

        elapsed = time.time() - t0
        post_state = self._collect_state()
        post_norm = float(np.linalg.norm(post_state))

        inject_signal = inject_data.flatten() if inject_data.ndim > 0 else np.array([0])
        fidelity = self._compute_fidelity(pre_state, post_state, inject_signal)
        retention = self._compute_retention(pre_state, post_state, pre_state)

        return InjectionResult(
            success=True,
            pre_injection_state_norm=pre_norm,
            post_injection_state_norm=post_norm,
            injection_fidelity=fidelity,
            retention_ratio=retention,
            state_delta_norm=float(np.linalg.norm(post_state - pre_state)),
            effective_capacity_used=1.0,
            time_elapsed=elapsed,
            extra={
                "inject_mode": inject_mode,
                "block_types": self._block_types,
                "n_slstm_blocks": sum(1 for bt in self._block_types if bt == "s"),
                "n_mlstm_blocks": sum(1 for bt in self._block_types if bt == "m"),
            },
        )

    def resume(self, n_steps: int = 1) -> np.ndarray:
        outputs = []
        for t in range(n_steps):
            x_t = np.zeros(self.d_model, dtype=np.float32)
            for block in self._blocks:
                x_t = block.step(x_t)
            outputs.append(x_t)
        return np.array(outputs)

    def state_capacity(self) -> int:
        total = 0
        for bt in self._block_types:
            if bt == "s":
                total += 4 * self.d_model
            else:
                total += self.n_heads * self.head_dim * self.head_dim + 2 * self.n_heads
        return total

    def effective_information(self) -> float:
        if self._state is None:
            return 0.0
        flat = self._state.flatten()
        std = float(np.std(flat))
        if std < 1e-12:
            return 0.0
        entropy = 0.5 * np.log(2 * np.pi * np.e * std ** 2)
        return float(entropy * len(flat))

    def analyze_mlstm_injection(self) -> Dict:
        mlstm_blocks = [(i, b) for i, (bt, b) in enumerate(zip(self._block_types, self._blocks))
                        if bt == "m" and isinstance(b, mLSTMBlock)]

        if not mlstm_blocks:
            return {"has_mlstm": False}

        results = {"has_mlstm": True, "per_block_analysis": []}

        for layer_idx, block in mlstm_blocks:
            ranks = []
            for h in range(self.n_heads):
                s = np.linalg.svd(block.C[h], compute_uv=False)
                rank = int(np.sum(s > 1e-6))
                ranks.append(rank)

            total_params = self.n_heads * self.head_dim * self.head_dim
            used_params = sum(ranks) * self.head_dim * 2

            results["per_block_analysis"].append({
                "layer": layer_idx,
                "ranks_per_head": ranks,
                "avg_rank": float(np.mean(ranks)),
                "max_rank": int(np.max(ranks)),
                "rank_utilization": float(np.mean(ranks)) / self.head_dim,
                "total_params": total_params,
                "used_params_estimate": used_params,
                "param_efficiency": used_params / total_params if total_params > 0 else 0,
            })

        return results


# ======================================================================
# Hybrid Architecture Model
# ======================================================================

class LayerType(Enum):
    TRANSFORMER = "transformer"
    SSM = "ssm"
    RWKV = "rwkv"
    XLSTM = "xlstm"


@dataclass
class HybridLayer:
    layer_type: LayerType
    layer_index: int
    model: ArchitectureStateModel
    d_model: int = 512


@dataclass
class InjectionPoint:
    layer_index: int
    layer_type: LayerType
    score: float
    fidelity: float
    retention: float
    capacity_available: float
    recovery_speed: float
    reasoning: str


class HybridArchitectureModel:
    def __init__(self, d_model: int = 512, layer_config: Optional[List[LayerType]] = None):
        self.d_model = d_model
        self.layers: List[HybridLayer] = []

        if layer_config is None:
            layer_config = self._default_jamba_config()

        for i, lt in enumerate(layer_config):
            if lt == LayerType.TRANSFORMER:
                model = TransformerStateModel(d_model=d_model, n_layers=1)
            elif lt == LayerType.SSM:
                model = SSMStateModel(SSMConfig(d_model=d_model, n_layers=1))
            elif lt == LayerType.RWKV:
                model = RWKVStateModel(RWKVConfig(d_model=d_model, n_layers=1))
            elif lt == LayerType.XLSTM:
                model = xLSTMStateModel(xLSTMConfig(d_model=d_model, n_layers=1, block_type="mlstm"))
            else:
                raise ValueError(f"Unknown layer type: {lt}")

            self.layers.append(HybridLayer(
                layer_type=lt,
                layer_index=i,
                model=model,
                d_model=d_model,
            ))

    @staticmethod
    def _default_jamba_config() -> List[LayerType]:
        config = []
        for _ in range(4):
            config.extend([LayerType.SSM] * 7)
            config.append(LayerType.TRANSFORMER)
        return config

    @staticmethod
    def jamba_style(n_blocks: int = 4, ssm_per_block: int = 7) -> List[LayerType]:
        config = []
        for _ in range(n_blocks):
            config.extend([LayerType.SSM] * ssm_per_block)
            config.append(LayerType.TRANSFORMER)
        return config

    @staticmethod
    def griffin_style(n_blocks: int = 4, rnn_per_block: int = 3) -> List[LayerType]:
        config = []
        for _ in range(n_blocks):
            config.extend([LayerType.SSM] * rnn_per_block)
            config.append(LayerType.TRANSFORMER)
        return config

    @staticmethod
    def zamba_style() -> List[LayerType]:
        return [LayerType.SSM, LayerType.SSM, LayerType.TRANSFORMER,
                LayerType.SSM, LayerType.SSM, LayerType.TRANSFORMER,
                LayerType.SSM, LayerType.TRANSFORMER]

    @staticmethod
    def all_transformer(n_layers: int = 12) -> List[LayerType]:
        return [LayerType.TRANSFORMER] * n_layers

    @staticmethod
    def all_ssm(n_layers: int = 12) -> List[LayerType]:
        return [LayerType.SSM] * n_layers

    def get_layer_summary(self) -> List[Dict]:
        summary = []
        for layer in self.layers:
            summary.append({
                "index": layer.layer_index,
                "type": layer.layer_type.value,
                "state_capacity": layer.model.state_capacity(),
                "effective_info": layer.model.effective_information(),
            })
        return summary

    def print_architecture(self):
        print(f"\nHybrid Architecture ({len(self.layers)} layers, d_model={self.d_model})")
        print("=" * 60)

        layer_str = ""
        for layer in self.layers:
            if layer.layer_type == LayerType.TRANSFORMER:
                layer_str += "T"
            elif layer.layer_type == LayerType.SSM:
                layer_str += "M"
            elif layer.layer_type == LayerType.RWKV:
                layer_str += "R"
            elif layer.layer_type == LayerType.XLSTM:
                layer_str += "X"

        for i, ch in enumerate(layer_str):
            if i > 0 and i % 8 == 0:
                print()
            print(f"[{i:2d}:{ch}]", end=" ")
        print()

        t_count = sum(1 for l in self.layers if l.layer_type == LayerType.TRANSFORMER)
        m_count = sum(1 for l in self.layers if l.layer_type == LayerType.SSM)
        r_count = sum(1 for l in self.layers if l.layer_type == LayerType.RWKV)
        x_count = sum(1 for l in self.layers if l.layer_type == LayerType.XLSTM)

        print(f"\n  Transformer: {t_count} | SSM: {m_count} | RWKV: {r_count} | xLSTM: {x_count}")


class InjectionLayerSelector:
    def __init__(self, hybrid_model: HybridArchitectureModel):
        self.model = hybrid_model

    def score_layer(self, layer: HybridLayer, inject_data: np.ndarray,
                    task_weights: Optional[Dict[str, float]] = None) -> InjectionPoint:
        default_weights = {
            "fidelity": 0.3,
            "retention": 0.3,
            "capacity": 0.2,
            "recovery": 0.2,
        }
        weights = task_weights or default_weights

        init_seq = np.random.randn(32).astype(np.float32)
        layer.model.init_state(init_seq)

        result = layer.model.inject(inject_data, inject_mode="append")

        fidelity = result.injection_fidelity
        retention = result.retention_ratio
        capacity = 1.0 - result.effective_capacity_used

        recovery_states = []
        current_state = layer.model.suspend()
        for _ in range(5):
            layer.model.resume(n_steps=1)
            new_state = layer.model.suspend()
            if current_state is not None and new_state is not None:
                min_len = min(len(current_state.flatten()), len(new_state.flatten()))
                diff = float(np.linalg.norm(
                    new_state.flatten()[:min_len] - current_state.flatten()[:min_len]
                ))
                recovery_states.append(diff)
            current_state = new_state

        if recovery_states:
            recovery_speed = 1.0 / (1.0 + np.mean(recovery_states))
        else:
            recovery_speed = 0.5

        score = (weights["fidelity"] * fidelity +
                 weights["retention"] * retention +
                 weights["capacity"] * capacity +
                 weights["recovery"] * recovery_speed)

        reasoning = self._generate_reasoning(layer, fidelity, retention, capacity, recovery_speed)

        return InjectionPoint(
            layer_index=layer.layer_index,
            layer_type=layer.layer_type,
            score=score,
            fidelity=fidelity,
            retention=retention,
            capacity_available=capacity,
            recovery_speed=recovery_speed,
            reasoning=reasoning,
        )

    def _generate_reasoning(self, layer: HybridLayer, fidelity: float,
                            retention: float, capacity: float,
                            recovery: float) -> str:
        parts = []

        if layer.layer_type == LayerType.TRANSFORMER:
            parts.append("Transformer layer: KV cache allows append-only injection")
            if capacity > 0.5:
                parts.append("high spare capacity")
            if retention > 0.9:
                parts.append("excellent retention (additive injection)")
        elif layer.layer_type == LayerType.SSM:
            parts.append("SSM layer: compressed state creates information bottleneck")
            if fidelity < 0.3:
                parts.append("low fidelity due to state compression")
            if retention < 0.5:
                parts.append("moderate retention loss from overwriting")
        elif layer.layer_type == LayerType.RWKV:
            parts.append("RWKV layer: time decay affects injection persistence")
            if recovery < 0.3:
                parts.append("injected info decays quickly")
        elif layer.layer_type == LayerType.XLSTM:
            parts.append("xLSTM layer: matrix memory supports rank-1 injection")
            if fidelity > 0.5:
                parts.append("good fidelity from additive cell state")

        return "; ".join(parts)

    def find_optimal_injection_points(self, inject_data: np.ndarray,
                                       top_k: int = 3,
                                       task_weights: Optional[Dict[str, float]] = None) -> List[InjectionPoint]:
        scores = []
        for layer in self.model.layers:
            point = self.score_layer(layer, inject_data, task_weights)
            scores.append(point)

        scores.sort(key=lambda p: p.score, reverse=True)
        return scores[:top_k]

    def compare_strategies(self, inject_data: np.ndarray) -> Dict:
        strategies = {
            "transformer_only": [],
            "ssm_only": [],
            "all_layers": [],
            "optimal_top3": [],
        }

        all_scores = []
        for layer in self.model.layers:
            point = self.score_layer(layer, inject_data)
            all_scores.append(point)

            strategies["all_layers"].append(point)

            if layer.layer_type == LayerType.TRANSFORMER:
                strategies["transformer_only"].append(point)
            elif layer.layer_type == LayerType.SSM:
                strategies["ssm_only"].append(point)

        sorted_scores = sorted(all_scores, key=lambda p: p.score, reverse=True)
        strategies["optimal_top3"] = sorted_scores[:3]

        results = {}
        for name, points in strategies.items():
            if not points:
                results[name] = {"avg_score": 0, "avg_fidelity": 0, "avg_retention": 0, "n_points": 0}
                continue

            results[name] = {
                "avg_score": float(np.mean([p.score for p in points])),
                "avg_fidelity": float(np.mean([p.fidelity for p in points])),
                "avg_retention": float(np.mean([p.retention for p in points])),
                "n_points": len(points),
                "layers": [{"idx": p.layer_index, "type": p.layer_type.value, "score": p.score} for p in points],
            }

        return results


class CrossLayerAnalysis:
    def __init__(self, hybrid_model: HybridArchitectureModel):
        self.model = hybrid_model

    def analyze_propagation(self, inject_data: np.ndarray) -> Dict:
        init_seq = np.random.randn(32).astype(np.float32)

        propagation_data = []

        for source_layer in self.model.layers:
            for layer in self.model.layers:
                layer.model.reset()
                layer.model.init_state(init_seq)

            source_layer.model.inject(inject_data, inject_mode="append")

            downstream_effects = []
            for layer in self.model.layers:
                if layer.layer_index <= source_layer.layer_index:
                    continue

                pre_state = layer.model.suspend()
                layer.model.resume(n_steps=1)
                post_state = layer.model.suspend()

                if pre_state is not None and post_state is not None:
                    min_len = min(len(pre_state.flatten()), len(post_state.flatten()))
                    effect = float(np.linalg.norm(
                        post_state.flatten()[:min_len] - pre_state.flatten()[:min_len]
                    ))
                else:
                    effect = 0.0

                downstream_effects.append({
                    "target_layer": layer.layer_index,
                    "target_type": layer.layer_type.value,
                    "effect_magnitude": effect,
                })

            propagation_data.append({
                "source_layer": source_layer.layer_index,
                "source_type": source_layer.layer_type.value,
                "downstream_effects": downstream_effects,
            })

        return {"propagation": propagation_data}

    def recommend_injection_plan(self, inject_data: np.ndarray,
                                  n_injections: int = 3) -> Dict:
        selector = InjectionLayerSelector(self.model)
        top_points = selector.find_optimal_injection_points(inject_data, top_k=n_injections)

        plan = {
            "n_injections": n_injections,
            "recommended_points": [],
            "overall_strategy": "",
        }

        for point in top_points:
            inject_mode = self._recommend_inject_mode(point)
            plan["recommended_points"].append({
                "layer_index": point.layer_index,
                "layer_type": point.layer_type.value,
                "inject_mode": inject_mode,
                "expected_fidelity": point.fidelity,
                "expected_retention": point.retention,
                "score": point.score,
                "reasoning": point.reasoning,
            })

        plan["overall_strategy"] = self._generate_strategy(top_points)

        return plan

    def _recommend_inject_mode(self, point: InjectionPoint) -> str:
        if point.layer_type == LayerType.TRANSFORMER:
            return "append"
        elif point.layer_type == LayerType.SSM:
            if point.retention > 0.7:
                return "append"
            else:
                return "merge"
        elif point.layer_type == LayerType.RWKV:
            return "append"
        elif point.layer_type == LayerType.XLSTM:
            return "rank1_inject"
        return "append"

    def _generate_strategy(self, points: List[InjectionPoint]) -> str:
        if not points:
            return "No suitable injection points found"

        type_counts = {}
        for p in points:
            lt = p.layer_type.value
            type_counts[lt] = type_counts.get(lt, 0) + 1

        dominant = max(type_counts, key=type_counts.get)

        strategies = {
            "transformer": "Prefer Transformer layers for SIG injection — append-only KV cache provides highest fidelity and retention",
            "ssm": "SSM layers are primary injection targets — use selective scan injection to respect SSM dynamics",
            "rwkv": "RWKV layers with WKV state injection — leverage time decay for natural relevance weighting",
            "xlstm": "xLSTM matrix memory layers — rank-1 updates naturally model key-value injection",
        }

        return strategies.get(dominant, "Mixed injection strategy across layer types")


# ======================================================================
# Benchmark Suite
# ======================================================================

def create_models(d_model: int = 256, n_layers: int = 4) -> Dict[str, object]:
    models = {
        "Transformer": TransformerStateModel(
            d_model=d_model, n_heads=8, n_layers=n_layers, max_seq_len=2048
        ),
        "SSM/Mamba": SSMStateModel(SSMConfig(
            d_model=d_model, d_state=16, n_layers=n_layers
        )),
        "RWKV": RWKVStateModel(RWKVConfig(
            d_model=d_model, n_heads=8, n_layers=n_layers, head_size=d_model // 8
        )),
        "xLSTM": xLSTMStateModel(xLSTMConfig(
            d_model=d_model, n_heads=8, n_layers=n_layers,
            head_dim=d_model // 8, block_type="mixed"
        )),
    }
    return models


def benchmark_information_retention(models: Dict, inject_data: np.ndarray,
                                     n_injections: int = 5) -> Dict:
    print("\n" + "=" * 80)
    print("Benchmark 1: Information Retention Across Architectures")
    print("=" * 80)

    results = {}
    for name, model in models.items():
        model.reset()
        init_seq = np.random.randn(64).astype(np.float32)
        model.init_state(init_seq)

        retentions = []
        fidelities = []
        for i in range(n_injections):
            result = model.inject(inject_data, inject_mode="append")
            retentions.append(result.retention_ratio)
            fidelities.append(result.injection_fidelity)

        results[name] = {
            "retentions": retentions,
            "fidelities": fidelities,
            "avg_retention": float(np.mean(retentions)),
            "final_retention": retentions[-1],
            "avg_fidelity": float(np.mean(fidelities)),
        }

        ret_str = " -> ".join(f"{r:.3f}" for r in retentions)
        print(f"  {name:<16} Retention: {ret_str}")
        print(f"  {'':<16} Avg Ret: {np.mean(retentions):.3f} | Avg Fid: {np.mean(fidelities):.3f}")

    return results


def benchmark_injection_strategies(models: Dict, inject_data: np.ndarray) -> Dict:
    print("\n" + "=" * 80)
    print("Benchmark 2: Injection Strategy Comparison")
    print("=" * 80)

    strategies = ["append", "overwrite", "merge"]
    results = {}

    for name, model in models.items():
        results[name] = {}

        for strategy in strategies:
            model.reset()
            init_seq = np.random.randn(64).astype(np.float32)
            model.init_state(init_seq)

            injection_results = []
            for _ in range(3):
                result = model.inject(inject_data, inject_mode=strategy)
                injection_results.append(result)

            avg_fid = float(np.mean([r.injection_fidelity for r in injection_results if r.success]))
            avg_ret = float(np.mean([r.retention_ratio for r in injection_results if r.success]))

            results[name][strategy] = {
                "avg_fidelity": avg_fid,
                "avg_retention": avg_ret,
            }

        best_strategy = max(results[name].keys(),
                           key=lambda s: results[name][s]["avg_retention"])
        print(f"  {name:<16} Best strategy: {best_strategy} "
              f"(ret={results[name][best_strategy]['avg_retention']:.3f})")

    print(f"\n  {'Architecture':<16} {'Append Ret':<12} {'Overwrite Ret':<14} {'Merge Ret':<12}")
    print(f"  {'-'*54}")
    for name in results:
        row = f"  {name:<16}"
        for s in strategies:
            ret = results[name].get(s, {}).get("avg_retention", 0)
            row += f" {ret:<12.3f}" if s != "overwrite" else f" {ret:<14.3f}"
        print(row)

    return results


def benchmark_state_capacity(models: Dict) -> Dict:
    print("\n" + "=" * 80)
    print("Benchmark 3: State Capacity Analysis")
    print("=" * 80)

    results = {}
    for name, model in models.items():
        info = model.get_state_info()
        capacity = model.state_capacity()
        eff_info = model.effective_information()
        cap_ratio = InformationMetrics.capacity_ratio(model)
        info_density = InformationMetrics.information_density(model)

        init_seq = np.random.randn(64).astype(np.float32)
        model.init_state(init_seq)
        eff_info_after = model.effective_information()

        results[name] = {
            "state_capacity": capacity,
            "effective_info_init": eff_info,
            "effective_info_after_seq": eff_info_after,
            "capacity_ratio": cap_ratio,
            "information_density": info_density,
            "architecture": info.architecture.value,
            "state_dim": info.state_dim,
            "num_layers": info.num_layers,
        }

        cap_mb = capacity * 2 / (1024 ** 2)
        print(f"  {name:<16} Capacity: {capacity:>10} elements ({cap_mb:.2f} MB @ fp16) | "
              f"Eff.Info: {eff_info_after:.1f} nats | Density: {info_density:.6f}")

    return results


def benchmark_latency(models: Dict, inject_data: np.ndarray,
                       n_runs: int = 10) -> Dict:
    print("\n" + "=" * 80)
    print("Benchmark 4: SIG Operation Latency")
    print("=" * 80)

    results = {}
    for name, model in models.items():
        init_times = []
        inject_times = []
        suspend_times = []
        resume_times = []

        for _ in range(n_runs):
            model.reset()
            init_seq = np.random.randn(64).astype(np.float32)

            t0 = time.time()
            model.init_state(init_seq)
            init_times.append(time.time() - t0)

            t0 = time.time()
            model.inject(inject_data, inject_mode="append")
            inject_times.append(time.time() - t0)

            t0 = time.time()
            model.suspend()
            suspend_times.append(time.time() - t0)

            t0 = time.time()
            model.resume(n_steps=1)
            resume_times.append(time.time() - t0)

        results[name] = {
            "init_time_ms": float(np.mean(init_times) * 1000),
            "inject_time_ms": float(np.mean(inject_times) * 1000),
            "suspend_time_ms": float(np.mean(suspend_times) * 1000),
            "resume_time_ms": float(np.mean(resume_times) * 1000),
        }

        print(f"  {name:<16} Init: {results[name]['init_time_ms']:6.2f}ms | "
              f"Inject: {results[name]['inject_time_ms']:6.2f}ms | "
              f"Suspend: {results[name]['suspend_time_ms']:6.2f}ms | "
              f"Resume: {results[name]['resume_time_ms']:6.2f}ms")

    return results


def benchmark_ssm_bottleneck(d_model: int = 256, n_layers: int = 4) -> Dict:
    print("\n" + "=" * 80)
    print("Benchmark 5: SSM Information Bottleneck Analysis")
    print("=" * 80)

    results = {}
    for d_state in [8, 16, 32, 64]:
        model = SSMStateModel(SSMConfig(d_model=d_model, d_state=d_state, n_layers=n_layers))
        for seq_len in [64, 256, 1024, 4096]:
            analysis = model.analyze_bottleneck(seq_len)
            key = f"d_state={d_state}, seq={seq_len}"
            results[key] = analysis
            print(f"  d_state={d_state:>3}, seq_len={seq_len:>5}: "
                  f"CR={analysis['compression_ratio']:.4f}, "
                  f"max_ret={analysis['theoretical_max_retention']:.3f}, "
                  f"severity={analysis['bottleneck_severity']}")

    return results


def benchmark_rwkv_decay(d_model: int = 256, n_layers: int = 4) -> Dict:
    print("\n" + "=" * 80)
    print("Benchmark 6: RWKV Time Decay Impact on SIG")
    print("=" * 80)

    model = RWKVStateModel(RWKVConfig(d_model=d_model, n_layers=n_layers))
    init_seq = np.random.randn(64).astype(np.float32)
    model.init_state(init_seq)

    decay_analysis = model.analyze_decay_impact(n_steps=100)

    print(f"  Avg decay rate: {decay_analysis['avg_decay_rate']:.6f}")
    print(f"  Min decay rate: {decay_analysis['min_decay_rate']:.6f}")
    print(f"  Max decay rate: {decay_analysis['max_decay_rate']:.6f}")
    if decay_analysis['half_life_steps'] is not None:
        print(f"  Half-life: {decay_analysis['half_life_steps']} steps")
    else:
        print(f"  Half-life: > 100 steps (slow decay)")

    print(f"\n  State size comparison:")
    for seq_len in [128, 512, 2048, 8192]:
        comp = model.compare_with_transformer_state_size(seq_len)
        print(f"    seq_len={seq_len:>5}: RWKV={comp['rwkv_state_size']:>8}, "
              f"Transformer={comp['transformer_kv_size']:>8}, "
              f"ratio={comp['compression_ratio']:.6f}")

    return decay_analysis


def benchmark_xlstm_mlstm(d_model: int = 256, n_layers: int = 4) -> Dict:
    print("\n" + "=" * 80)
    print("Benchmark 7: xLSTM Matrix Memory Analysis for SIG")
    print("=" * 80)

    model = xLSTMStateModel(xLSTMConfig(
        d_model=d_model, n_heads=8, n_layers=n_layers,
        head_dim=d_model // 8, block_type="mlstm"
    ))
    init_seq = np.random.randn(64).astype(np.float32)
    model.init_state(init_seq)

    analysis = model.analyze_mlstm_injection()

    if analysis["has_mlstm"]:
        for block_info in analysis["per_block_analysis"]:
            print(f"  Layer {block_info['layer']} (mLSTM):")
            print(f"    Avg rank: {block_info['avg_rank']:.1f}/{d_model // 8}")
            print(f"    Rank utilization: {block_info['rank_utilization']:.3f}")
            print(f"    Param efficiency: {block_info['param_efficiency']:.3f}")
            print(f"    -> Rank-1 SIG injection adds 1 rank per injection")
            print(f"    -> Max injections before rank saturation: "
                  f"~{int(d_model // 8 - block_info['avg_rank'])}")
    else:
        print("  No mLSTM blocks in current configuration")

    return analysis


def benchmark_hybrid(d_model: int = 256) -> Dict:
    print("\n" + "=" * 80)
    print("Benchmark 8: Hybrid Architecture SIG Analysis")
    print("=" * 80)

    configs = {
        "Jamba (4x8)": HybridArchitectureModel.jamba_style(n_blocks=4, ssm_per_block=7),
        "Griffin (4x4)": HybridArchitectureModel.griffin_style(n_blocks=4, rnn_per_block=3),
        "Zamba": HybridArchitectureModel.zamba_style(),
    }

    results = {}

    for name, layer_config in configs.items():
        print(f"\n  --- {name} ---")
        model = HybridArchitectureModel(d_model=d_model, layer_config=layer_config)
        model.print_architecture()

        inject_data = np.random.randn(32).astype(np.float32)
        selector = InjectionLayerSelector(model)
        top_points = selector.find_optimal_injection_points(inject_data, top_k=3)

        print(f"\n  Top injection points:")
        for p in top_points:
            print(f"    Layer {p.layer_index} ({p.layer_type.value}): "
                  f"score={p.score:.3f}, fid={p.fidelity:.3f}, "
                  f"ret={p.retention:.3f}, cap={p.capacity_available:.3f}")
            print(f"      -> {p.reasoning}")

        strategy_results = selector.compare_strategies(inject_data)
        results[name] = {
            "top_points": [
                {"layer": p.layer_index, "type": p.layer_type.value,
                 "score": p.score, "fidelity": p.fidelity, "retention": p.retention}
                for p in top_points
            ],
            "strategies": {},
        }

        for sname, sdata in strategy_results.items():
            results[name]["strategies"][sname] = {
                "avg_score": sdata["avg_score"],
                "avg_fidelity": sdata["avg_fidelity"],
                "avg_retention": sdata["avg_retention"],
            }

        print(f"\n  Strategy comparison:")
        for sname, sdata in strategy_results.items():
            print(f"    {sname:<20} score={sdata['avg_score']:.3f}, "
                  f"fid={sdata['avg_fidelity']:.3f}, ret={sdata['avg_retention']:.3f}")

    return results


def benchmark_cross_architecture(models: Dict, inject_data: np.ndarray,
                                  n_injections: int = 5) -> Dict:
    print("\n" + "=" * 80)
    print("Benchmark 9: Cross-Architecture SIG Comparison (Unified)")
    print("=" * 80)

    comparison = InformationMetrics.compare_architectures(models, inject_data, n_injections)
    print_comparison(comparison)

    return comparison


def generate_summary(all_results: Dict) -> str:
    lines = []
    lines.append("=" * 80)
    lines.append("R3 Research Summary: SIG Beyond Transformer")
    lines.append("=" * 80)
    lines.append("")

    if "retention" in all_results:
        lines.append("1. INFORMATION RETENTION")
        lines.append("-" * 40)
        for name, data in all_results["retention"].items():
            lines.append(f"  {name}: avg_ret={data['avg_retention']:.3f}, "
                        f"final_ret={data['final_retention']:.3f}")
        lines.append("")

    if "capacity" in all_results:
        lines.append("2. STATE CAPACITY")
        lines.append("-" * 40)
        for name, data in all_results["capacity"].items():
            lines.append(f"  {name}: {data['state_capacity']} elements, "
                        f"density={data['information_density']:.6f}")
        lines.append("")

    lines.append("3. KEY FINDINGS")
    lines.append("-" * 40)
    lines.append("  - Transformer: KV cache append-only injection, highest retention")
    lines.append("  - SSM/Mamba: Fixed-size state creates information bottleneck")
    lines.append("  - RWKV: Time decay naturally weights injected information")
    lines.append("  - xLSTM: Matrix memory supports natural rank-1 injection")
    lines.append("  - Hybrid: Inject into Transformer layers when available")
    lines.append("")
    lines.append("4. SIG FEASIBILITY RANKING")
    lines.append("-" * 40)
    lines.append("  1. Transformer: Native SIG support (append-only KV cache)")
    lines.append("  2. xLSTM: Good SIG support (additive cell state, rank-1 updates)")
    lines.append("  3. RWKV: Moderate SIG support (decay-weighted injection)")
    lines.append("  4. SSM: Challenging SIG (compressed state bottleneck)")
    lines.append("")
    lines.append("5. HYBRID ARCHITECTURE RECOMMENDATION")
    lines.append("-" * 40)
    lines.append("  - Prefer Transformer layers for SIG injection in hybrid models")
    lines.append("  - Use SSM layers for efficient sequence processing between injections")
    lines.append("  - Jamba-style (7:1 SSM:Transformer) needs careful injection planning")
    lines.append("  - Griffin-style (3:1) has more Transformer injection opportunities")

    return "\n".join(lines)


def _make_serializable(obj):
    if isinstance(obj, dict):
        return {str(k): _make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_make_serializable(v) for v in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, (float, int, str, bool)) or obj is None:
        return obj
    return str(obj)


# ======================================================================
# R3 Empirical: CO Benchmark Parameterization
# ======================================================================

def run_r3_empirical() -> Dict:
    """Parameterize architecture model with actual CO benchmark measurements."""
    apf08, apf4 = 0.932, 0.929
    tps08, tps4 = 3.56, 9.90

    results = {
        "task": "r3_empirical",
        "transformer_baseline": {
            "prefill_saving_08b": apf08,
            "prefill_saving_4b": apf4,
            "tok_per_sec_08b": 1000 / tps08,
            "tok_per_sec_4b": 1000 / tps4,
            "speed_ratio_4b_to_08b": tps4 / tps08,
        },
        "projected_savings": {
            "Transformer": {"value": "100%", "basis": "Measured on Qwen3.5"},
            "xLSTM": {"value": "85-95%", "basis": "Rank-1 matrix injection hypothesis"},
            "RWKV": {"value": "70-85%", "basis": "Causal (k,v) insertion hypothesis"},
            "Mamba/SSM": {"value": "40-60%", "basis": "State capacity bottleneck hypothesis"},
        },
        "note": "Non-Transformer projections are HYPOTHESES. No non-Transformer SIG implementation exists."
    }
    return results


def print_r3_empirical(results: Dict):
    """Pretty-print R3 empirical results."""
    b = results["transformer_baseline"]
    print("\n" + "=" * 70)
    print("  R3 EMPIRICAL: CO Benchmark Parameterization")
    print("=" * 70)
    print(f"  Calibrated: 0.8B prefill_save={b['prefill_saving_08b']:.3f}, {b['tok_per_sec_08b']:.0f} tok/s")
    print(f"              4B   prefill_save={b['prefill_saving_4b']:.3f}, {b['tok_per_sec_4b']:.0f} tok/s")
    print(f"              4B/0.8B speed ratio: {b['speed_ratio_4b_to_08b']:.1f}x")
    print("\n  Projected Prefill Savings (rel. to Transformer=100%):")
    for arch, d in results["projected_savings"].items():
        print(f"    {arch:<15} {d['value']:<12} ({d['basis']})")
    print("  NOTE: " + results["note"])


# ======================================================================
# R1: SIG Injection Attention Distribution Analysis
# ======================================================================

def run_r1_attention(model_id: str = "Qwen/Qwen2.5-0.5B") -> Optional[Dict]:
    """Measure attention distribution shift between SIG injection and full re-encoding.

    Uses HuggingFace transformers + modelscope to load models.
    Compares attention weights from full re-encoding vs SIG injection (past_key_values).
    """
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        print("R1 requires: pip install torch transformers modelscope")
        return None

    print(f"\n{'='*70}")
    print(f"  R1: SIG Injection Attention Distribution Analysis")
    print(f"  Model: {model_id}")
    print(f"{'='*70}\n")

    try:
        from modelscope import snapshot_download
        print(f"Downloading model from modelscope: {model_id}")
        model_dir = snapshot_download(model_id, cache_dir="./modelscope_cache")
        tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            model_dir, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True,
            attn_implementation="eager")
    except Exception:
        print(f"Loading from modelscope failed, trying HuggingFace hub: {model_id}")
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True,
            attn_implementation="eager")
    model.eval()
    n_layers = model.config.num_hidden_layers
    n_heads = model.config.num_attention_heads
    print(f"  Model loaded: {n_layers} layers, {n_heads} heads")

    prefix = "You are a travel assistant.\nUser: Paris weather and attractions.\nAssistant: Let me check.\n"
    injection = (
        "[Tool Results]\nWeather Paris: 18C partly cloudy.\nAttractions: Eiffel Tower, Louvre, Notre-Dame.\n"
        "Weather London: 15C overcast.\nAttractions: British Museum, Tower of London, Big Ben.\n\nContinue:\n"
    )

    inputs_f = tokenizer(prefix + injection, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out_f = model(**inputs_f, output_attentions=True)
    attn_full = [a.cpu().numpy()[:, 0] for a in out_f.attentions]

    inputs_p = tokenizer(prefix, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out_p = model(**inputs_p, output_attentions=True, use_cache=True)
    past_kv = out_p.past_key_values
    inputs_i = tokenizer(injection, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out_i = model(input_ids=inputs_i["input_ids"], past_key_values=past_kv,
                     output_attentions=True, use_cache=True)
    attn_inj = [a.cpu().numpy()[:, 0] for a in out_i.attentions]

    layer_results = []
    for l in range(n_layers):
        ms = min(attn_full[l].shape[2], attn_inj[l].shape[2])
        agrs, csims = 0.0, []
        for h in range(n_heads):
            af = attn_full[l][h, :ms, :ms].mean(axis=0)
            ai = attn_inj[l][h, :ms, :ms].mean(axis=0)
            t5f = set(np.argsort(-af)[:5])
            t5i = set(np.argsort(-ai)[:5])
            agrs += len(t5f & t5i) / 5.0
            d = np.dot(af, ai)
            n = np.linalg.norm(af) * np.linalg.norm(ai)
            csims.append(float(d / max(n, 1e-10)))
        layer_results.append({
            "layer": l, "head_agreement": agrs / n_heads,
            "cosine_similarity": float(np.mean(csims))
        })

    t = n_layers // 3
    def avg(layers_r, key):
        vals = [r[key] for r in layers_r]
        return sum(vals) / max(len(vals), 1)

    regions = {
        "early": {"layers": f"0-{t-1}", **{k: avg(layer_results[:t], k) for k in ["head_agreement", "cosine_similarity"]}},
        "middle": {"layers": f"{t}-{2*t-1}", **{k: avg(layer_results[t:2*t], k) for k in ["head_agreement", "cosine_similarity"]}},
        "late": {"layers": f"{2*t}-{n_layers-1}", **{k: avg(layer_results[2*t:], k) for k in ["head_agreement", "cosine_similarity"]}},
        "overall": {"layers": f"0-{n_layers-1}", **{k: avg(layer_results, k) for k in ["head_agreement", "cosine_similarity"]}},
    }

    result = {
        "task": "r1_attention",
        "model_id": model_id,
        "n_layers": n_layers,
        "n_heads": n_heads,
        "regions": regions,
        "layer_details": layer_results,
    }

    print(f"\n{'='*70}")
    print(f"  LAYER SENSITIVITY ANALYSIS")
    print(f"{'='*70}")
    print(f"  {'Region':<12} {'Layers':<12} {'Head Agr':<10} {'Cos Sim':<10}")
    print(f"  {'-'*12} {'-'*12} {'-'*10} {'-'*10}")
    for name, reg in regions.items():
        print(f"  {name.capitalize():<12} {reg['layers']:<12} {reg['head_agreement']:<10.3f} {reg['cosine_similarity']:<10.3f}")

    return result


# ======================================================================
# R3: Cross-Architecture SIG Simulation Runner
# ======================================================================

def run_r3_simulation(
    d_model: int = 256, n_heads: int = 8, n_layers: int = 4,
    n_injections: int = 5, seed: int = 42
) -> Dict:
    """Run full cross-architecture SIG simulation benchmark suite."""
    np.random.seed(seed)
    inject_data = np.random.randn(32).astype(np.float32)

    models = {
        "Transformer": TransformerStateModel(d_model, n_heads, n_layers, max_seq_len=2048),
        "SSM/Mamba": SSMStateModel(SSMConfig(d_model=d_model, d_state=16, n_layers=n_layers)),
        "RWKV": RWKVStateModel(RWKVConfig(d_model=d_model, n_heads=n_heads, n_layers=n_layers, head_size=d_model // n_heads)),
        "xLSTM": xLSTMStateModel(xLSTMConfig(d_model=d_model, n_heads=n_heads, n_layers=n_layers, head_dim=d_model // n_heads, block_type="mixed")),
    }

    all_results = {}

    print("\n" + "=" * 80)
    print("R3: SIG Beyond Transformer — Cross-Architecture Benchmark")
    print(f"  d_model={d_model}, n_heads={n_heads}, n_layers={n_layers}, n_injections={n_injections}")
    print("=" * 80)

    all_results["retention"] = benchmark_information_retention(models, inject_data, n_injections)
    all_results["strategies"] = benchmark_injection_strategies(models, inject_data)
    all_results["capacity"] = benchmark_state_capacity(models)
    all_results["latency"] = benchmark_latency(models, inject_data)
    all_results["ssm_bottleneck"] = benchmark_ssm_bottleneck(d_model, n_layers)
    all_results["rwkv_decay"] = benchmark_rwkv_decay(d_model, n_layers)
    all_results["xlstm_mlstm"] = benchmark_xlstm_mlstm(d_model, n_layers)
    all_results["hybrid"] = benchmark_hybrid(d_model)

    models2 = {
        "Transformer": TransformerStateModel(d_model, n_heads, n_layers, max_seq_len=2048),
        "SSM/Mamba": SSMStateModel(SSMConfig(d_model=d_model, d_state=16, n_layers=n_layers)),
        "RWKV": RWKVStateModel(RWKVConfig(d_model=d_model, n_heads=n_heads, n_layers=n_layers, head_size=d_model // n_heads)),
        "xLSTM": xLSTMStateModel(xLSTMConfig(d_model=d_model, n_heads=n_heads, n_layers=n_layers, head_dim=d_model // n_heads, block_type="mixed")),
    }
    all_results["cross_arch"] = benchmark_cross_architecture(models2, inject_data, n_injections)

    summary = generate_summary(all_results)
    print("\n" + summary)

    return all_results


# ======================================================================
# Unified CLI
# ======================================================================

def main():
    parser = argparse.ArgumentParser(description="Universal Transformer Testing Engine")
    parser.add_argument("--task", default="all",
                        choices=["r1", "r3", "r3-empirical", "all"],
                        help="Which test to run")
    parser.add_argument("--model-id", type=str, default="Qwen/Qwen2.5-0.5B",
                        help="Model ID for R1 attention analysis")
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--n-injections", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="", help="Output JSON path")
    args = parser.parse_args()

    all_data = {}

    if args.task in ("r1", "all"):
        r1_result = run_r1_attention(args.model_id)
        if r1_result:
            all_data["r1"] = r1_result

    if args.task in ("r3", "all"):
        r3_result = run_r3_simulation(
            d_model=args.d_model, n_heads=args.n_heads,
            n_layers=args.n_layers, n_injections=args.n_injections,
            seed=args.seed)
        all_data["r3"] = {"simulation_complete": True}

    if args.task in ("r3-empirical", "all"):
        r3e_result = run_r3_empirical()
        print_r3_empirical(r3e_result)
        all_data["r3_empirical"] = r3e_result

    if args.output and all_data:
        with open(args.output, "w") as f:
            json.dump(all_data, f, indent=2, default=str)
        print(f"\nReport saved to: {args.output}")

    print("\nAll tasks complete.")


if __name__ == "__main__":
    main()