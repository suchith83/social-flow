# Package initializer for application monitoring
"""
Application Metrics Monitoring Package.

This package provides:
- Metrics collection for application-level KPIs
- Exporters for Prometheus / OpenTelemetry
- Automatic instrumentation (middleware, decorators)
- Anomaly detection using ML/statistical models
- Alerting logic for proactive monitoring
"""

from .metrics_collector import MetricsCollector
from .exporter import MetricsExporter
from .instrumentation import instrument_function, RequestMetricsMiddleware
from .anomaly_detection import AnomalyDetector
from .alerts import AlertManager
from .config import MetricsConfig

__all__ = [
    "MetricsCollector",
    "MetricsExporter",
    "instrument_function",
    "RequestMetricsMiddleware",
    "AnomalyDetector",
    "AlertManager",
    "MetricsConfig",
]
