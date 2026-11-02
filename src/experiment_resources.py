"""Wall time, Linux process RSS and sampled process GPU memory (not allocator telemetry)."""
import os
import resource
import subprocess
import threading
import time


class Resources:
    def __init__(self):
        self.stop = threading.Event()
        self.gpu_peak_mib = None

    def _sample(self):
        while not self.stop.is_set():
            try:
                result = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,used_gpu_memory',
                                         '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=3)
                used = sum(float(line.split(',')[1].strip()) for line in result.stdout.splitlines()
                           if line.split(',')[0].strip() == str(os.getpid()))
                if result.returncode == 0:
                    self.gpu_peak_mib = max(self.gpu_peak_mib or 0, used)
            except (OSError, ValueError, subprocess.TimeoutExpired):
                pass
            self.stop.wait(0.5)

    def __enter__(self):
        self.started = time.perf_counter()
        self.thread = threading.Thread(target=self._sample, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.stop.set()
        self.thread.join(timeout=4)
        self.result = {'wall_seconds': time.perf_counter() - self.started,
                       'peak_process_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                       'sampled_peak_process_gpu_mib': self.gpu_peak_mib,
                       'rss_scope': 'Linux process lifetime high-water mark; includes loading/imports',
                       'gpu_scope': 'nvidia-smi process total across GPUs, sampled every 0.5s; may miss short peaks'}
