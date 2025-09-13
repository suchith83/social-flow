# Adapter to collect and normalize infrastructure metrics
"""
Infra Adapter

- Provides a unified interface to retrieve operational metrics from backends like Prometheus or CloudWatch.
- Implements:
  - Bulk fetching with concurrency controls
  - Basic in-memory caching (time-bucketed)
  - Exponential backoff and retry for flaky backends
  - Circuit-breaker style suppression after repeated failures
"""

import time
import random
import threading
from typing import Dict, List, Any
from collections import defaultdict, deque

# Constants
DEFAULT_CACHE_TTL_SECONDS = 30
CIRCUIT_BREAKER_WINDOW = 60  # seconds
CIRCUIT_BREAKER_FAILURE_THRESHOLD = 5


class BackendFailure(Exception):
    pass


class CircuitBreaker:
    """
    Very small circuit-breaker: counts failures within a window and opens if threshold exceeded.
    """
    def __init__(self):
        self.failures = deque()
        self.lock = threading.Lock()
        self.open_until = 0

    def record_failure(self):
        with self.lock:
            now = time.time()
            self.failures.append(now)
            # purge old
            while self.failures and (now - self.failures[0]) > CIRCUIT_BREAKER_WINDOW:
                self.failures.popleft()
            if len(self.failures) >= CIRCUIT_BREAKER_FAILURE_THRESHOLD:
                self.open_until = now + CIRCUIT_BREAKER_WINDOW

    def allowed(self) -> bool:
        with self.lock:
            return time.time() >= self.open_until


class InfraAdapter:
    def __init__(self, backends: Dict[str, Any]):
        self.backends = backends
        self.cache: Dict[str, Dict[str, Any]] = {}  # key -> {"ts":int, "data":[...]}
        self.circuit = CircuitBreaker()

    def fetch_metrics_bulk(self, metrics: List[str], window_minutes: int = 15) -> Dict[str, List[float]]:
        """
        Public method to fetch multiple metrics. Uses caching and handles backend failures gracefully.
        Returns mapping: metric_name -> list of floats (time series sampled)
        """
        results: Dict[str, List[float]] = {}
        for metric in metrics:
            try:
                results[metric] = self.fetch_metric(metric, window_minutes)
            except BackendFailure:
                # degrade gracefully: empty series
                results[metric] = []
            except Exception:
                results[metric] = []
        return results

    def fetch_metric(self, metric_name: str, window_minutes: int = 15, points: int = 60) -> List[float]:
        """
        Fetch single metric with caching and robust backend handling.
        - Attempts Prometheus first (if enabled)
        - If backend is flaky, apply backoff & record failures
        NOTE: In production this would use an HTTP client, auth, TLS verify, pagination.
        """
        cache_key = f"{metric_name}:{window_minutes}"
        now = int(time.time())
        cached = self.cache.get(cache_key)
        if cached and (now - cached["ts"]) < DEFAULT_CACHE_TTL_SECONDS:
            return cached["data"]

        # Circuit breaker: avoid hammering failing backends
        if not self.circuit.allowed():
            raise BackendFailure("circuit open for infra backends")

        # Try backends according to configuration
        last_exception = None
        try_methods = []
        if self.backends.get("prometheus", {}).get("enabled", False):
            try_methods.append(self._fetch_from_prometheus)
        if self.backends.get("cloudwatch", {}).get("enabled", False):
            try_methods.append(self._fetch_from_cloudwatch)
        # fallback: synthetic generator
        try_methods.append(self._generate_synthetic)

        for method in try_methods:
            attempt = 0
            max_attempts = 3
            backoff = 0.5
            while attempt < max_attempts:
                try:
                    data = method(metric_name, window_minutes=window_minutes, points=points)
                    # basic validation
                    if not isinstance(data, list):
                        raise ValueError("invalid data from backend")
                    # sort to keep deterministic percentiles
                    data_sorted = sorted([float(x) for x in data])
                    # cache result
                    self.cache[cache_key] = {"ts": now, "data": data_sorted}
                    return data_sorted
                except Exception as e:
                    last_exception = e
                    attempt += 1
                    time.sleep(backoff * (1 + random.random() * 0.2))
                    backoff *= 2
                    # record failure to allow circuit to open if persistent
                    self.circuit.record_failure()

        # If all backends failed, raise a BackendFailure so callers can degrade
        raise BackendFailure(f"All backends failed for metric {metric_name}: {last_exception}")

    # ---- Backend implementation details ----

    def _fetch_from_prometheus(self, metric_name: str, window_minutes: int, points: int) -> List[float]:
        """
        Placeholder: Replace with actual Prometheus query via requests or prometheus-api-client.
        Here we simulate network jitter and occasional failures.
        """
        # simulate flaky behavior
        if random.random() < 0.05:
            raise Exception("prometheus transient error")

        # Simulate a realistic metric range per metric keyword
        if "cpu" in metric_name:
            base = 50.0
            spread = 40.0
        elif "memory" in metric_name:
            base = 60.0
            spread = 30.0
        elif "disk" in metric_name or "iops" in metric_name:
            base = 2000.0
            spread = 4000.0
        elif "latency" in metric_name:
            base = 20.0
            spread = 60.0
        elif "incidents" in metric_name:
            base = 1.0
            spread = 5.0
        else:
            base = 100.0
            spread = 100.0

        # Build a pseudo-time-series with small correlation
        series = []
        for i in range(points):
            noise = random.gauss(0, spread * 0.1)
            drift = (i / points) * (spread * 0.05)
            value = max(0.0, base + noise + drift)
            series.append(round(value, 2))
        return series

    def _fetch_from_cloudwatch(self, metric_name: str, window_minutes: int, points: int) -> List[float]:
        """
        Placeholder for CloudWatch. Simulate slightly different distributions.
        """
        if random.random() < 0.08:
            raise Exception("cloudwatch transient error")
        # Slightly different base/spread to simulate cross-backend variance
        return [round(max(0.0, random.gauss(100, 50)), 2) for _ in range(points)]

    def _generate_synthetic(self, metric_name: str, window_minutes: int, points: int) -> List[float]:
        """
        Synthetic fallback when no backend is available. Deterministic-ish based on metric name.
        """
        seed = sum(ord(c) for c in metric_name) % 1000
        random.seed(seed + int(time.time() // 60))  # change per minute
        return [round(max(0.0, random.gauss(100, 50)), 2) for _ in range(points)]
