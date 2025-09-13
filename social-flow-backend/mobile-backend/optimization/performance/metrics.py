# Performance metrics collector
"""
Performance Metrics Collector
Aggregates latency, cache hit ratio, and load distribution.
"""

import time
from collections import defaultdict
from .config import CONFIG


class PerformanceMetrics:
    def __init__(self):
        self.metrics = defaultdict(list)
        self.last_flush_time = time.time()

    def log(self, key: str, value: float):
        self.metrics[key].append(value)

    def flush(self) -> dict:
        """Flush metrics periodically."""
        now = time.time()
        if now - self.last_flush_time >= CONFIG.metrics_flush_interval_sec:
            report = {k: sum(v) / len(v) for k, v in self.metrics.items() if v}
            self.metrics.clear()
            self.last_flush_time = now
            return report
        return {}
