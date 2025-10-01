# cache_monitoring_config.py
# Created by Create-Configurations.ps1
"""
cache_monitoring_config.py
--------------------------
Defines observability and monitoring configurations for caching.
Includes metrics exporters, tracing flags, and alert thresholds.
"""

import os


class CacheMonitoringConfig:
    """Monitoring configuration for caches."""

    ENABLE_METRICS = os.getenv("CACHE_METRICS", "true").lower() == "true"
    ENABLE_TRACING = os.getenv("CACHE_TRACING", "false").lower() == "true"
    ALERT_THRESHOLD_HIT_RATIO = float(os.getenv("CACHE_HIT_ALERT", "0.85"))
    ALERT_THRESHOLD_LATENCY_MS = int(os.getenv("CACHE_LAT_ALERT", "200"))

    EXPORTER = os.getenv("CACHE_METRICS_EXPORTER", "prometheus")  # prometheus | otlp | datadog

    @classmethod
    def summary(cls):
        return {
            "metrics_enabled": cls.ENABLE_METRICS,
            "tracing_enabled": cls.ENABLE_TRACING,
            "hit_ratio_threshold": cls.ALERT_THRESHOLD_HIT_RATIO,
            "latency_threshold_ms": cls.ALERT_THRESHOLD_LATENCY_MS,
            "exporter": cls.EXPORTER,
        }


if __name__ == "__main__":
    print("?? Cache Monitoring Config:", CacheMonitoringConfig.summary())
