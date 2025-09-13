# Metrics collector for React Native flows
"""
React Native metrics collector.

Buffers metrics in memory and flushes aggregated snapshots.
Replace flush() to push to monitoring (Prometheus/Cloud Monitoring).
"""

import time
from collections import defaultdict
from .config import CONFIG

class RNMetrics:
    def __init__(self):
        self._metrics = defaultdict(list)
        self._last_flush = time.time()

    def log(self, key: str, value: float = 1.0):
        self._metrics[key].append(value)

    def flush(self) -> dict:
        now = time.time()
        if now - self._last_flush >= CONFIG.metrics_flush_interval_sec:
            report = {}
            for k, vals in self._metrics.items():
                # expose counts for integer-like metrics and averages for latencies if needed
                report[k] = sum(vals) if vals else 0
            self._metrics.clear()
            self._last_flush = now
            return report
        return {}
