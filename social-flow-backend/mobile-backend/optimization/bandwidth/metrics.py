# Collects metrics and reports efficiency
"""
Bandwidth Metrics Collector
Monitors compression ratios, bitrate shifts, batch sizes, and cache hit-rates.
"""

import time
from collections import defaultdict
from .config import CONFIG


class BandwidthMetrics:
    def __init__(self):
        self.metrics = defaultdict(list)
        self.last_flush_time = time.time()

    def log(self, key: str, value: float):
        """Log a metric value."""
        self.metrics[key].append(value)

    def flush(self) -> dict:
        """Flush metrics periodically to external monitoring."""
        now = time.time()
        if now - self.last_flush_time >= CONFIG.metrics_flush_interval_sec:
            report = {k: sum(v) / len(v) for k, v in self.metrics.items() if v}
            self.metrics.clear()
            self.last_flush_time = now
            return report
        return {}
