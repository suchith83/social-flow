# Monitoring and metrics collection
import time
import statistics
from collections import defaultdict


class APIMetricsCollector:
    """
    Advanced Metrics Collector.

    - Records latency, error rates, throughput.
    - Supports percentile calculation (P50, P95, P99).
    """

    def __init__(self):
        self.latencies = []
        self.errors = 0
        self.requests = 0
        self.routes = defaultdict(list)

    def record_request(self, route: str, latency: float, success: bool):
        self.requests += 1
        if not success:
            self.errors += 1
        self.latencies.append(latency)
        self.routes[route].append(latency)

    def get_summary(self):
        if not self.latencies:
            return {"requests": 0}
        return {
            "requests": self.requests,
            "errors": self.errors,
            "error_rate": self.errors / self.requests,
            "avg_latency": statistics.mean(self.latencies),
            "p95_latency": statistics.quantiles(self.latencies, n=100)[94],
            "p99_latency": statistics.quantiles(self.latencies, n=100)[98],
        }
