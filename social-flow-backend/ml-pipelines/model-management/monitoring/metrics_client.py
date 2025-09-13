# Client library for pushing/pulling metrics
"""
metrics_client.py
A small wrapper to standardize instrumentation across services.
Usage:
    from metrics_client import MetricsClient
    mc = MetricsClient("fraud-detector")
    with mc.timer("infer"):
        ... call model ...
    mc.increment("requests", labels={"status":"success"})
"""

import time
import logging
from typing import Dict, Optional
from contextlib import contextmanager
from prometheus_client import Counter, Summary, Gauge, CollectorRegistry

from utils import setup_logger

logger = setup_logger("MetricsClient")


class MetricsClient:
    def __init__(self, model_name: str, registry: Optional[CollectorRegistry] = None):
        self.model = model_name
        self.registry = registry
        # standardized metric names
        self._counters: Dict[str, Counter] = {}
        self._summaries: Dict[str, Summary] = {}
        self._gauges: Dict[str, Gauge] = {}

    def _counter(self, name: str, documentation: str, labels: Optional[list] = None):
        if name not in self._counters:
            self._counters[name] = Counter(name, documentation, labels or [], registry=self.registry)
        return self._counters[name]

    def _summary(self, name: str, documentation: str, labels: Optional[list] = None):
        if name not in self._summaries:
            self._summaries[name] = Summary(name, documentation, labels or [], registry=self.registry)
        return self._summaries[name]

    def _gauge(self, name: str, documentation: str, labels: Optional[list] = None):
        if name not in self._gauges:
            self._gauges[name] = Gauge(name, documentation, labels or [], registry=self.registry)
        return self._gauges[name]

    def increment(self, metric: str, amount: float = 1.0, labels: Optional[Dict[str, str]] = None):
        c = self._counter(metric, f"{metric} counter", labels and list(labels.keys()) or [])
        if labels:
            c.labels(**labels).inc(amount)
        else:
            c.inc(amount)

    def observe(self, metric: str, value: float, labels: Optional[Dict[str, str]] = None):
        s = self._summary(metric, f"{metric} summary", labels and list(labels.keys()) or [])
        if labels:
            s.labels(**labels).observe(value)
        else:
            s.observe(value)

    def set_gauge(self, metric: str, value: float, labels: Optional[Dict[str, str]] = None):
        g = self._gauge(metric, f"{metric} gauge", labels and list(labels.keys()) or [])
        if labels:
            g.labels(**labels).set(value)
        else:
            g.set(value)

    @contextmanager
    def timer(self, metric: str, labels: Optional[Dict[str, str]] = None):
        start = time.time()
        try:
            yield
            duration = time.time() - start
            self.observe(metric, duration, labels)
        except Exception:
            duration = time.time() - start
            # record with failure label if provided
            labels2 = dict(labels or {})
            labels2.update({"status": "failure"})
            self.observe(metric, duration, labels2)
            raise
