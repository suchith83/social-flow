# Package initializer for security monitoring and tooling
"""
Security Metrics Monitoring package.

Exports:
 - SecurityMetricsConfig: configuration
 - SecurityCollector: collects security telemetry and updates metrics
 - SecurityExporter: starts Prometheus exporter (and can push to SIEM)
 - SecurityAnomalyDetector: specialized anomaly detection for security signals
 - SecurityAlertManager: alert routing with dedupe & severity escalation
 - ThreatIntelClient: optional threat intelligence lookups
"""

from .config import SecurityMetricsConfig
from .security_collector import SecurityCollector
from .exporter import SecurityExporter
from .anomaly_detection import SecurityAnomalyDetector
from .alerts import SecurityAlertManager
from .threat_intel import ThreatIntelClient
from .utils import parse_ip, hash_event_id

__all__ = [
    "SecurityMetricsConfig",
    "SecurityCollector",
    "SecurityExporter",
    "SecurityAnomalyDetector",
    "SecurityAlertManager",
    "ThreatIntelClient",
    "parse_ip",
    "hash_event_id",
]
