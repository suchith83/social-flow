# Collects and aggregates metrics from the application
"""
Collects application-level metrics (latency, throughput, error rates, cache hit ratios, etc.)
Exposes APIs for developers to record custom metrics.
"""

import time
import threading
from prometheus_client import Counter, Histogram, Gauge
from typing import Callable, Dict


class MetricsCollector:
    """Thread-safe application metrics collector."""

    def __init__(self):
        # Core metrics
        self.request_latency = Histogram(
            "app_request_latency_seconds",
            "Request latency in seconds",
            ["endpoint", "method"]
        )
        self.request_count = Counter(
            "app_request_total",
            "Total number of requests",
            ["endpoint", "method", "status"]
        )
        self.error_count = Counter(
            "app_errors_total",
            "Total number of application errors",
            ["type"]
        )
        self.cache_hit_ratio = Gauge(
            "app_cache_hit_ratio",
            "Cache hit ratio",
        )
        # Internal lock for thread safety
        self._lock = threading.Lock()

    def observe_request(self, endpoint: str, method: str, status: str, duration: float):
        """Record request metrics."""
        with self._lock:
            self.request_latency.labels(endpoint, method).observe(duration)
            self.request_count.labels(endpoint, method, status).inc()

    def record_error(self, error_type: str):
        """Increment error counter."""
        with self._lock:
            self.error_count.labels(error_type).inc()

    def update_cache_hit_ratio(self, hits: int, misses: int):
        """Update cache hit ratio gauge."""
        with self._lock:
            total = hits + misses
            ratio = hits / total if total > 0 else 0
            self.cache_hit_ratio.set(ratio)
