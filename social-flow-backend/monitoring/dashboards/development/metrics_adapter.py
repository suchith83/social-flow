# Adapter to collect and normalize metrics
"""
Metrics Adapter

Provides a unified interface for fetching metrics from multiple backends
(e.g., Prometheus, Datadog, AWS CloudWatch). Uses caching for performance.
"""

import random
import time
from typing import List, Dict


class MetricsAdapter:
    def __init__(self):
        self.cache: Dict[str, List[float]] = {}

    def fetch_metric(self, metric_name: str, points: int = 10) -> List[float]:
        """
        Fetches latest values of a metric.
        In production, connect to Prometheus/CloudWatch APIs here.
        """
        now = int(time.time())
        key = f"{metric_name}:{now // 30}"  # Cache by 30s bucket

        if key in self.cache:
            return self.cache[key]

        # Simulate data with randomness
        data = [round(random.uniform(50, 500), 2) for _ in range(points)]
        self.cache[key] = data
        return data
