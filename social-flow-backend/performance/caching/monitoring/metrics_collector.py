import time
import psutil
import threading
from typing import Dict, Any
from collections import deque

class CacheMetricsCollector:
    """
    Collects cache-related metrics like hit/miss ratio, latency,
    memory usage, and system-level stats.
    """

    def __init__(self, max_samples: int = 1000):
        self.hits = 0
        self.misses = 0
        self.latencies = deque(maxlen=max_samples)
        self.sample_interval = 1  # seconds
        self._stop_event = threading.Event()
        self._thread = None

    def record_hit(self, latency: float):
        """Record a cache hit with observed latency in ms."""
        self.hits += 1
        self.latencies.append(latency)

    def record_miss(self, latency: float):
        """Record a cache miss with observed latency in ms."""
        self.misses += 1
        self.latencies.append(latency)

    def get_metrics(self) -> Dict[str, Any]:
        """Return current cache metrics snapshot."""
        total = self.hits + self.misses
        hit_ratio = (self.hits / total) if total > 0 else 0.0
        avg_latency = (sum(self.latencies) / len(self.latencies)) if self.latencies else 0.0
        memory_usage = psutil.virtual_memory().percent
        cpu_usage = psutil.cpu_percent(interval=0.1)

        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_ratio": hit_ratio,
            "avg_latency_ms": avg_latency,
            "memory_usage_percent": memory_usage,
            "cpu_usage_percent": cpu_usage,
            "timestamp": time.time(),
        }

    def start_background_collection(self, callback):
        """
        Continuously collect metrics and send to callback.
        Useful for exporters or dashboards.
        """
        def run():
            while not self._stop_event.is_set():
                metrics = self.get_metrics()
                callback(metrics)
                time.sleep(self.sample_interval)

        self._thread = threading.Thread(target=run, daemon=True)
        self._thread.start()

    def stop_background_collection(self):
        """Stop background metrics collection."""
        self._stop_event.set()
        if self._thread:
            self._thread.join()
