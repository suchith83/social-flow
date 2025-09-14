# scripts/monitoring/__init__.py
"""
Monitoring package for Social Flow

Provides:
 - Config-driven metric & log collection
 - Prometheus exporter integration
 - Synthetic checks (HTTP, gRPC, DB ping)
 - Alert manager (Slack / PagerDuty / Email hooks)
 - Basic dashboard generator for Grafana-friendly JSON
 - Runner to orchestrate collectors and scheduled checks

Design goals:
 - Safe defaults
 - Lightweight dependencies (requests, prometheus_client, watchdog optional)
 - Extensible and testable
"""
__version__ = "1.0.0"
__author__ = "Social Flow Observability Team"
