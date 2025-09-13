# Metrics collector for Flutter flows
"""
Flutter metrics collector.

Collects metrics for:
 - bundle uploads / diff creation success
 - analytics ingestion rates
 - push sends / errors
 - device registration counts

Buffer metrics in memory and flush on request. Replace flush() with push to monitoring in prod.
"""

import time
from collections import defaultdict
from .config import CONFIG

class FlutterMetrics:
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
                report[k] = sum(vals) if vals else 0
            self._metrics.clear()
            self._last_flush = now
            return report
        return {}
