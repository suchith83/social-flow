# scripts/monitoring/prometheus_exporter.py
import logging
import threading
from typing import Dict, Any

from prometheus_client import start_http_server, Gauge, Counter, Summary

logger = logging.getLogger("monitoring.prometheus")


class PrometheusExporter:
    """
    Thin wrapper over prometheus_client to register/export metrics.
    Start HTTP endpoint provided in config.
    """

    def __init__(self, config: Dict[str, Any]):
        p_cfg = config.get("monitoring", {}).get("prometheus", {})
        self.enabled = bool(p_cfg.get("enabled", True))
        self.port = int(p_cfg.get("port", 9101))
        # sample metric set - consumer code should use these or create their own
        self._gauges = {}
        self._counters = {}
        self._summaries = {}

        # register some default metrics
        self.register_gauge("synthetic_check_latency_seconds", "Latency of synthetic checks in seconds", labels=["check"])
        self.register_counter("synthetic_check_failures_total", "Failures of synthetic checks", labels=["check"])

    def start(self):
        if not self.enabled:
            logger.info("Prometheus exporter disabled in config")
            return
        logger.info("Starting Prometheus HTTP server on port %d", self.port)
        # run in a background thread safe for servers that use event loops
        t = threading.Thread(target=start_http_server, args=(self.port,), daemon=True)
        t.start()

    def register_gauge(self, name: str, doc: str, labels=None):
        labels = labels or []
        if name not in self._gauges:
            if labels:
                self._gauges[name] = Gauge(name, doc, labels)
            else:
                self._gauges[name] = Gauge(name, doc)
        return self._gauges[name]

    def set_gauge(self, name: str, value, labels=None):
        g = self._gauges.get(name)
        if not g:
            g = self.register_gauge(name, name, labels or [])
        if labels:
            g.labels(*labels).set(value)
        else:
            g.set(value)

    def register_counter(self, name: str, doc: str, labels=None):
        labels = labels or []
        if name not in self._counters:
            if labels:
                self._counters[name] = Counter(name, doc, labels)
            else:
                self._counters[name] = Counter(name, doc)
        return self._counters[name]

    def inc_counter(self, name: str, amount=1, labels=None):
        c = self._counters.get(name)
        if not c:
            c = self.register_counter(name, name, labels or [])
        if labels:
            c.labels(*labels).inc(amount)
        else:
            c.inc(amount)

    def register_summary(self, name: str, doc: str, labels=None):
        labels = labels or []
        if name not in self._summaries:
            if labels:
                self._summaries[name] = Summary(name, doc, labels)
            else:
                self._summaries[name] = Summary(name, doc)
        return self._summaries[name]
