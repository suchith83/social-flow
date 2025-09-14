# Configuration for KPIs, alerts, and business dashboards
"""
Configuration for business metrics monitoring.
"""

import os


class BusinessMetricsConfig:
    EXPORTER_BACKEND: str = os.getenv("BUSINESS_METRICS_EXPORTER", "prometheus")
    EXPORTER_PORT: int = int(os.getenv("BUSINESS_METRICS_PORT", "9200"))

    ANOMALY_METHOD: str = os.getenv("BUSINESS_ANOMALY_METHOD", "zscore")
    ANOMALY_SENSITIVITY: float = float(os.getenv("BUSINESS_ANOMALY_SENSITIVITY", "2.0"))

    ALERT_CHANNELS = os.getenv("BUSINESS_ALERT_CHANNELS", "email,slack").split(",")

    @classmethod
    def summary(cls) -> dict:
        return {k: getattr(cls, k) for k in dir(cls) if k.isupper()}
