# exporter.py
from typing import Dict, Any


class Exporter:
    """
    Generic exporter interface for metrics/logs/traces.
    Can be extended for Prometheus, OpenTelemetry, etc.
    """

    def __init__(self, metrics_collector, tracer):
        self.metrics = metrics_collector
        self.tracer = tracer

    def export_prometheus(self) -> str:
        """Export metrics in Prometheus text format."""
        snapshot = self.metrics.export()
        lines = []
        for name, value in snapshot["counters"].items():
            lines.append(f"{name}_total {value}")
        for name, value in snapshot["gauges"].items():
            lines.append(f"{name} {value}")
        for name, values in snapshot["histograms"].items():
            for v in values:
                lines.append(f"{name}_bucket {v}")
        return "\n".join(lines)

    def export_traces(self) -> Dict[str, Any]:
        """Export traces as JSON."""
        return self.tracer.export()
