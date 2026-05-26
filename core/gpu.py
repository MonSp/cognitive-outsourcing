"""GPU memory and utilization monitor using pynvml (optional dependency).

Tracks VRAM usage, SM occupancy, and memory bandwidth utilization when
available.  Gracefully degrades when pynvml is unavailable or no GPU is found.
"""

import time
import warnings

try:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        import pynvml
    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False


class GPUMonitor:
    """Monitor NVIDIA GPU memory usage via pynvml.

    Tracks baseline memory at init and reports deltas on each snapshot.
    Optionally samples SM (streaming multiprocessor) occupancy and memory
    bandwidth utilisation when the driver exposes the relevant counters.

    Gracefully degrades when pynvml is unavailable or no GPU is found.
    """

    def __init__(self):
        self.handle = None
        self.baseline_mb = 0.0
        self.total_mb = 0.0
        self.enabled = False
        self._sm_available = False
        if not PYNVML_AVAILABLE:
            return
        try:
            pynvml.nvmlInit()
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            self.enabled = True
            info = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
            self.total_mb = info.total / (1024 ** 2)
            self.baseline_mb = info.used / (1024 ** 2)
            name = pynvml.nvmlDeviceGetName(self.handle)
            if isinstance(name, bytes):
                name = name.decode()
            print(f"[GPU] {name}, Total {self.total_mb:.0f} MB, Baseline {self.baseline_mb:.0f} MB")

            try:
                util = pynvml.nvmlDeviceGetUtilizationRates(self.handle)
                _ = util.gpu
                self._sm_available = True
            except Exception:
                self._sm_available = False
        except Exception as e:
            print(f"[GPU] Init failed: {e}")

    def snapshot(self):
        """Return a dict with current used_mb and delta_mb from baseline."""
        if not self.enabled:
            return {"used_mb": 0.0, "delta_mb": 0.0}
        info = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
        used = info.used / (1024 ** 2)
        return {"used_mb": used, "delta_mb": used - self.baseline_mb}

    def utilization_snapshot(self):
        """Return SM occupancy (%) and memory-bandwidth occupancy (%) if available.

        SM occupancy reflects compute-unit busy time; memory-bandwidth
        occupancy reflects the fraction of time the memory controller is
        saturated.  Together they discriminate between compute-bound and
        memory-bound inference phases — the key distinction between SIG's
        fragmented small evals and AppLoop's batched prefill.
        """
        if not self.enabled or not self._sm_available:
            return {"sm_pct": 0.0, "mem_pct": 0.0, "available": False}
        try:
            util = pynvml.nvmlDeviceGetUtilizationRates(self.handle)
            return {"sm_pct": util.gpu, "mem_pct": util.memory, "available": True}
        except Exception:
            return {"sm_pct": 0.0, "mem_pct": 0.0, "available": False}

    def bandwidth_profile(self, eval_function, token_count, label=""):
        """Run *eval_function(token_count)* and return utilisation deltas.

        Returns (wall_ms, sm_pct, mem_pct) so that the caller can
        distinguish whether a prefill/injection step is compute-bound
        (high SM, high mem → good batching) or memory-bandwidth-bound
        (low SM, high mem → fragmented small-kernel launches, SIG
        pattern).
        """
        util_before = self.utilization_snapshot()
        t0 = time.time()
        eval_function()
        wall_ms = (time.time() - t0) * 1000
        util_after = self.utilization_snapshot()

        sm_delta = util_after["sm_pct"] - util_before["sm_pct"] if util_after["available"] else 0.0
        mem_delta = util_after["mem_pct"] - util_before["mem_pct"] if util_after["available"] else 0.0

        if sm_delta < 0:
            sm_delta = util_after["sm_pct"]
        if mem_delta < 0:
            mem_delta = util_after["mem_pct"]

        return wall_ms, sm_delta, mem_delta

    def shutdown(self):
        """Release the pynvml handle."""
        if self.enabled:
            pynvml.nvmlShutdown()
