# Exports collected metrics to external systems (e.g., Prometheus, Grafana)
"""
Exports metrics to external systems (Prometheus / OpenTelemetry).
"""

from prometheus_client import start_http_server
import logging
from .config import MetricsConfig


class MetricsExporter:
    """Metrics exporter service."""

    def __init__(self, port: int = MetricsConfig.EXPORTER_PORT):
        self.port = port
        self.backend = MetricsConfig.EXPORTER_BACKEND

    def start(self):
        """Start the exporter server."""
        if self.backend == "prometheus":
            logging.info(f"[MetricsExporter] Starting Prometheus exporter on port {self.port}")
            start_http_server(self.port)
        else:
            raise NotImplementedError(f"Exporter backend '{self.backend}' not yet supported.")
