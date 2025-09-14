# Integrates with monitoring/alerting systems
# performance/cdn/edge-locations/monitor_integration.py
"""
Monitoring and observability integration helpers.

- Emit metrics (prometheus-style) via a simple in-memory collector
- Push alerts/events to external systems (placeholder adapters)
- Provide hooks for publishing health, capacity, and routing events
"""

from typing import Dict, Any, Optional
from collections import defaultdict
import time
from .utils import logger

class InMemoryMetrics:
    """Simple metrics aggregator for counters and gauges (not thread-safe; intended for single-process demo)."""
    def __init__(self):
        self.counters = defaultdict(float)
        self.gauges = {}
        self.histograms = defaultdict(list)

    def inc(self, name: str, val: float = 1.0):
        self.counters[name] += val

    def set_gauge(self, name: str, val: float):
        self.gauges[name] = val

    def observe(self, name: str, val: float):
        self.histograms[name].append(val)

    def snapshot(self) -> Dict[str, Any]:
        return {
            "counters": dict(self.counters),
            "gauges": dict(self.gauges),
            "histograms": {k: list(v) for k, v in self.histograms.items()},
            "ts": time.time(),
        }

metrics = InMemoryMetrics()

# Example adapter interface for pushing alerts (can be implemented for PagerDuty/Slack)
class AlertAdapter:
    async def send_alert(self, title: str, body: str, severity: str = "warning"):
        """Implementers should send to an external service."""
        raise NotImplementedError()

class LoggingAlertAdapter(AlertAdapter):
    async def send_alert(self, title: str, body: str, severity: str = "warning"):
        logger.warning(f"ALERT [{severity}] {title} - {body}")

async def publish_health_event(node_id: str, healthy: bool, adapter: Optional[AlertAdapter] = None):
    metrics.set_gauge(f"node.{node_id}.healthy", 1.0 if healthy else 0.0)
    metrics.inc("health_check.runs", 1)
    if not healthy and adapter:
        await adapter.send_alert("Edge node unhealthy", f"Node {node_id} marked unhealthy", severity="high")
