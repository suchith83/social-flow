# __init__.py
"""
Monitoring Library
==================
A production-grade monitoring library for distributed systems.

Features:
- Structured logging (JSON, correlation IDs)
- Metrics collection (counters, gauges, histograms)
- Distributed tracing (OpenTelemetry compatible)
- Alerts integration (Slack, Email, PagerDuty)
- Health & readiness probes
- Performance profiling
- Exporters for Prometheus & OpenTelemetry
"""

from .logger import get_logger
from .metrics import MetricsCollector
from .tracer import Tracer
from .alerts import AlertManager
from .healthcheck import HealthCheck
from .profiler import Profiler
from .exporter import Exporter

__all__ = [
    "get_logger",
    "MetricsCollector",
    "Tracer",
    "AlertManager",
    "HealthCheck",
    "Profiler",
    "Exporter"
]
