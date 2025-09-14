# Configuration for security rules, alert thresholds, and integrations
"""
Configuration for security metrics and monitoring.
All sensible defaults via environment variables so infra can tweak in deployments.
"""

import os
from typing import List

class SecurityMetricsConfig:
    # Exporter / scraping
    EXPORTER_BACKEND: str = os.getenv("SEC_METRICS_EXPORTER", "prometheus")
    EXPORTER_PORT: int = int(os.getenv("SEC_METRICS_PORT", "9400"))

    # Collection choices
    ENABLE_SIEM_PUSH: bool = os.getenv("SEC_ENABLE_SIEM_PUSH", "false").lower() == "true"
    SIEM_ENDPOINT: str = os.getenv("SEC_SIEM_ENDPOINT", "")

    # Anomaly detection defaults for security (conservative)
    ANOMALY_METHOD: str = os.getenv("SEC_ANOMALY_METHOD", "isolation_forest")  # zscore | ewma | isolation_forest
    ANOMALY_SENSITIVITY: float = float(os.getenv("SEC_ANOMALY_SENSITIVITY", "0.05"))  # contamination for iforest

    # Threat intel
    THREAT_INTEL_PROVIDERS: List[str] = os.getenv("SEC_THREAT_INTEL", "abuseipdb,virustotal").split(",")

    # Alerting
    ALERT_CHANNELS: List[str] = os.getenv("SEC_ALERT_CHANNELS", "pagerduty,slack,email").split(",")
    ALERT_DEDUPE_WINDOW_SECONDS: int = int(os.getenv("SEC_ALERT_DEDUPE_WINDOW", "600"))  # 10 minutes dedupe

    # Rate limits for event ingestion to avoid overload
    INGEST_RATE_PER_SEC: int = int(os.getenv("SEC_INGEST_RATE_PER_SEC", "500"))

    @classmethod
    def summary(cls) -> dict:
        return {k: getattr(cls, k) for k in dir(cls) if k.isupper()}
