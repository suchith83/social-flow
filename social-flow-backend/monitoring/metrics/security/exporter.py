# Exports security findings to dashboards, SIEM, or external APIs
"""
Exporter for security metrics.

Starts a Prometheus HTTP endpoint to expose the metrics collected above.
Also contains a placeholder to push structured events to SIEM (if configured).
"""

import logging
from prometheus_client import start_http_server
from .config import SecurityMetricsConfig

logger = logging.getLogger("security_metrics.exporter")

class SecurityExporter:
    def __init__(self, port: int = None):
        self.port = port or SecurityMetricsConfig.EXPORTER_PORT
        self.backend = SecurityMetricsConfig.EXPORTER_BACKEND

    def start(self):
        if self.backend != "prometheus":
            raise NotImplementedError(f"Exporter backend '{self.backend}' not supported for security module.")
        logger.info("Starting Prometheus exporter for security metrics on port %s", self.port)
        # prometheus_client spawns a background HTTP server thread
        start_http_server(self.port)
