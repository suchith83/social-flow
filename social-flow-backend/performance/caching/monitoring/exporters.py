import json
import logging
from typing import Dict, Any
import requests

logger = logging.getLogger(__name__)

class PrometheusExporter:
    """Exports cache metrics to a Prometheus Pushgateway."""

    def __init__(self, pushgateway_url: str):
        self.url = pushgateway_url.rstrip("/")

    def export(self, metrics: Dict[str, Any]):
        """Push metrics to Prometheus Pushgateway."""
        try:
            lines = []
            for key, value in metrics.items():
                lines.append(f"{key} {value}")
            payload = "\n".join(lines)

            response = requests.post(
                f"{self.url}/metrics/job/cache-monitor",
                data=payload,
                headers={"Content-Type": "text/plain"},
                timeout=5,
            )
            response.raise_for_status()
            logger.info("Exported metrics to Prometheus successfully.")
        except Exception as e:
            logger.error(f"Failed to export metrics to Prometheus: {e}")


class OpenTelemetryExporter:
    """Exports cache metrics to an OpenTelemetry collector."""

    def __init__(self, collector_endpoint: str):
        self.collector_endpoint = collector_endpoint

    def export(self, metrics: Dict[str, Any]):
        """Send metrics as JSON to OpenTelemetry endpoint."""
        try:
            response = requests.post(
                self.collector_endpoint,
                json=metrics,
                timeout=5,
            )
            response.raise_for_status()
            logger.info("Exported metrics to OpenTelemetry successfully.")
        except Exception as e:
            logger.error(f"Failed to export metrics to OpenTelemetry: {e}")
