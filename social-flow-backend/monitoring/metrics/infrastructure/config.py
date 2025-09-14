# Configuration for infrastructure metrics, exporters, and alerts
"""
Configuration for infrastructure metrics.

This centralizes environment-driven settings which are safe for production deployment.
"""

import os
from typing import List


class InfraMetricsConfig:
    # Prometheus exporter
    EXPORTER_BACKEND: str = os.getenv("INFRA_METRICS_EXPORTER", "prometheus")
    EXPORTER_PORT: int = int(os.getenv("INFRA_METRICS_PORT", "9187"))  # default infra port

    # Sampling
    SCRAPE_INTERVAL_SECONDS: int = int(os.getenv("INFRA_SCRAPE_INTERVAL", "10"))
    HIGH_RES_INTERVAL_SECONDS: int = int(os.getenv("INFRA_HIGH_RES_INTERVAL", "1"))  # for cpu percent bursts

    # Anomaly detection
    ANOMALY_METHOD: str = os.getenv("INFRA_ANOMALY_METHOD", "ewma")  # ewma | zscore | isolation_forest
    ANOMALY_SENSITIVITY: float = float(os.getenv("INFRA_ANOMALY_SENSITIVITY", "3.0"))

    # Cloud integrations
    ENABLE_CLOUD_EXPORT: bool = os.getenv("INFRA_ENABLE_CLOUD_EXPORT", "false").lower() == "true"
    CLOUD_BACKEND: str = os.getenv("INFRA_CLOUD_BACKEND", "cloudwatch")  # cloudwatch | stackdriver

    # Kubernetes
    ENABLE_K8S_COLLECTION: bool = os.getenv("INFRA_ENABLE_K8S_COLLECTION", "false").lower() == "true"

    # Alerting channels (csv)
    ALERT_CHANNELS: List[str] = os.getenv("INFRA_ALERT_CHANNELS", "email,slack").split(",")

    @classmethod
    def summary(cls) -> dict:
        return {k: getattr(cls, k) for k in dir(cls) if k.isupper()}
