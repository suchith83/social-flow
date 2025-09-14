# Package initializer for infrastructure monitoring
"""
Infrastructure Metrics Monitoring package.

Provides:
- InfraCollector: collects host/container/k8s-level metrics (CPU, mem, disk, net, processes)
- MetricsExporter: exposes Prometheus endpoint (and placeholder for cloud exporters)
- K8sMetrics: optional collection for Kubernetes objects if running inside cluster
- Anomaly detection and alerting
"""

from .config import InfraMetricsConfig
from .infra_collector import InfraCollector
from .exporter import InfraMetricsExporter
from .k8s_metrics import K8sMetricsCollector
from .anomaly_detection import InfraAnomalyDetector
from .alerts import InfraAlertManager

__all__ = [
    "InfraMetricsConfig",
    "InfraCollector",
    "InfraMetricsExporter",
    "K8sMetricsCollector",
    "InfraAnomalyDetector",
    "InfraAlertManager",
]
