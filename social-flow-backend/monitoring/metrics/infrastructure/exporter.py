# Exports collected infrastructure data to dashboards or external systems
"""
Prometheus exporter for infrastructure metrics and optional cloud adapters.

- start_prometheus_http exposes /metrics via prometheus_client.start_http_server
- cloud exporter hooks are intentionally pluggable (to support CloudWatch, Stackdriver, etc.)
"""

import logging
from prometheus_client import start_http_server, REGISTRY
from .config import InfraMetricsConfig

logger = logging.getLogger("infra_metrics.exporter")


class InfraMetricsExporter:
    """Simple exporter manager."""

    def __init__(self, port: int = InfraMetricsConfig.EXPORTER_PORT, backend: str = None):
        self.port = port
        self.backend = backend or InfraMetricsConfig.EXPORTER_BACKEND
        self._started = False

    def start_prometheus_http(self):
        """Start Prometheus HTTP endpoint in the background (blocking thread inside prometheus_client)."""
        if self._started:
            logger.warning("Exporter already started")
            return
        logger.info("Starting Prometheus metrics HTTP server on port %s", self.port)
        # start_http_server spawns a background thread exposing metrics from global registry
        start_http_server(self.port)
        self._started = True

    def register_collector(self, collector):
        """Optional: register custom collectors to the global REGISTRY."""
        try:
            REGISTRY.register(collector)
        except Exception:
            logger.debug("Collector could not be registered or already registered")

    def start(self):
        if self.backend == "prometheus":
            self.start_prometheus_http()
        else:
            raise NotImplementedError(f"Exporter backend {self.backend} not implemented")

    # Placeholder for cloud export push (e.g., CloudWatch PutMetricData)
    def push_to_cloud(self, push_callable):
        if not InfraMetricsConfig.ENABLE_CLOUD_EXPORT:
            logger.debug("Cloud export not enabled")
            return
        try:
            push_callable()
        except Exception:
            logger.exception("Cloud push failed")
