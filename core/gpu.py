"""GPU memory monitor using pynvml (optional dependency)."""

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
    Gracefully degrades when pynvml is unavailable or no GPU is found.
    """

    def __init__(self):
        self.handle = None
        self.baseline_mb = 0.0
        self.total_mb = 0.0
        self.enabled = False
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
        except Exception as e:
            print(f"[GPU] Init failed: {e}")

    def snapshot(self):
        """Return a dict with current used_mb and delta_mb from baseline."""
        if not self.enabled:
            return {"used_mb": 0.0, "delta_mb": 0.0}
        info = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
        used = info.used / (1024 ** 2)
        return {"used_mb": used, "delta_mb": used - self.baseline_mb}

    def shutdown(self):
        """Release the pynvml handle."""
        if self.enabled:
            pynvml.nvmlShutdown()
