# Metrics collector for iOS flows
"""
iOS metrics collector.

Collects metrics for:
 - APNs sends / failures
 - IPA diff production success rate
 - Analytics ingestion rates
 - Signature verification pass/fail

The collector buffers and returns aggregated snapshots on flush().
Replace flush() with integration to Prometheus/Cloud Monitoring or push to timeseries DB.
"""

import time
from collections import defaultdict
from .config import CONFIG

class iOSMetrics:
    def __init__(self):
        self.metrics = defaultdict(list)
        self.last_flush = time.time()

    def log(self, key: str, value: float = 1.0):
        self.metrics[key].append(value)

    def flush(self) -> dict:
        now = time.time()
        if now - self.last_flush >= CONFIG.metrics_flush_interval_sec:
            report = {}
            for k, vals in self.metrics.items():
                report[k] = sum(vals) if vals else 0
            self.metrics.clear()
            self.last_flush = now
            return report
        return {}
