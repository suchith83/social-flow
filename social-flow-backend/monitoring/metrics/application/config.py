# Configuration for metrics, exporters, and alerts
"""
Central configuration for metrics monitoring.
Supports dynamic updates via environment variables or config files.
"""

import os
from typing import List


class MetricsConfig:
    # Exporter settings
    EXPORTER_BACKEND: str = os.getenv("METRICS_EXPORTER", "prometheus")  # prometheus | otlp | statsd
    EXPORTER_PORT: int = int(os.getenv("METRICS_EXPORTER_PORT", "9090"))

    # Collection settings
    SCRAPE_INTERVAL: int = int(os.getenv("METRICS_SCRAPE_INTERVAL", "15"))  # seconds
    ENABLE_INSTRUMENTATION: bool = os.getenv("ENABLE_INSTRUMENTATION", "true").lower() == "true"

    # Anomaly detection
    ANOMALY_DETECTION_METHOD: str = os.getenv("ANOMALY_METHOD", "zscore")  # zscore | ewma | isolation_forest
    ANOMALY_SENSITIVITY: float = float(os.getenv("ANOMALY_SENSITIVITY", "2.5"))

    # Alerting
    ALERT_CHANNELS: List[str] = os.getenv("ALERT_CHANNELS", "email,slack").split(",")

    @classmethod
    def summary(cls) -> dict:
        """Return configuration summary."""
        return {k: getattr(cls, k) for k in dir(cls) if k.isupper()}
