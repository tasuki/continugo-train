"""Optional sampled driver memory, including native CUDA allocations outside Torch.

nvidia-smi is queried off the compute thread. This is a sampled process-tree
high-water mark, not an exact allocation peak, and excludes unrelated GPU users.
"""

import os
import shutil
import subprocess
import threading
from pathlib import Path


def descendant(pid, root):
    """Linux parent walk; processes that have already exited are not counted."""
    seen = set()
    while pid > 0 and pid not in seen:
        if pid == root:
            return True
        seen.add(pid)
        try:
            status = Path(f"/proc/{pid}/status").read_text()
            pid = int(
                next(line.split()[1] for line in status.splitlines() if line.startswith("PPid:"))
            )
        except (OSError, StopIteration, ValueError):
            return False
    return False


class GpuMemorySampler:
    def __init__(self, pid=None, *, enabled=True, interval=0.1):
        self.pid = os.getpid() if pid is None else pid
        self.interval = interval
        self.available = (
            enabled and Path("/proc").is_dir() and shutil.which("nvidia-smi") is not None
        )
        self.peak = None
        self.samples = 0
        self.stop = threading.Event()
        self.thread = None

    def _sample(self):
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-compute-apps=pid,used_gpu_memory",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=2,
                check=False,
            )
            if result.returncode:
                return
            total = 0.0
            for line in result.stdout.splitlines():
                pid, memory = line.split(",")
                if descendant(int(pid), self.pid):
                    total += float(memory)
        except (OSError, ValueError, subprocess.TimeoutExpired):
            return
        self.peak = max(self.peak or 0.0, total)
        self.samples += 1

    def _run(self):
        while not self.stop.is_set():
            self._sample()
            if self.stop.wait(self.interval):
                break

    def __enter__(self):
        if self.available:
            self.thread = threading.Thread(target=self._run, daemon=True)
            self.thread.start()
        return self

    def __exit__(self, *_):
        self.stop.set()
        if self.thread is not None:
            self.thread.join()

    def summary(self):
        return {
            "sampled_gpu_peak_mib": self.peak,
            "gpu_memory_samples": self.samples,
            "gpu_poll_interval_seconds": self.interval,
        }
