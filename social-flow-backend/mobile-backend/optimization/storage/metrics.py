# Collects storage metrics and usage statistics
"""
Storage Metrics Collector

Collects metrics such as:
 - put/get latency (simulated here)
 - compression ratios
 - deduplication hit rates
 - tier transitions
 - storage size per tier

This example stores metrics in memory and provides a flush() that returns aggregated
metrics. Replace flush() to push to Prometheus/Datadog/CloudWatch.
"""

import time
from collections import defaultdict
from typing import Dict
from .config import CONFIG

class StorageMetrics:
    def __init__(self):
        self._metrics = defaultdict(list)
        self._last_flush = time.time()

    def log(self, key: str, value: float):
        self._metrics[key].append(value)

    def incr(self, key: str, amount: float = 1.0):
        self._metrics[key].append(amount)

    def flush(self) -> Dict[str, float]:
        now = time.time()
        if now - self._last_flush >= CONFIG.metrics_flush_interval_sec:
            report = {}
            for k, vals in self._metrics.items():
                if not vals:
                    continue
                report[k] = sum(vals) / len(vals) if k.endswith("_latency") else sum(vals)
            self._metrics.clear()
            self._last_flush = now
            return report
        return {}
