# metrics.py
# Created by Create-DistributedFiles.ps1
"""
Lightweight tracing-related metrics to observe tracing health (span creation, exporter errors).

These are Prometheus metrics and are intentionally low-cardinality. The module is defensive if prometheus_client is missing.
"""

import logging

logger = logging.getLogger("tracing.distributed.metrics")

try:
    from prometheus_client import Counter, Gauge
    PROM_AVAILABLE = True
except Exception:
    PROM_AVAILABLE = False
    logger.debug("prometheus_client not installed; tracing metrics disabled.")


class TraceMetrics:
    def __init__(self):
        if not PROM_AVAILABLE:
            # no-op placeholders
            self.spans_created = None
            self.exporter_errors = None
            self.spans_sampled = None
            return

        # low-cardinality metrics
        self.spans_created = Counter("tracing_spans_created_total", "Total spans created by process")
        self.spans_sampled = Counter("tracing_spans_sampled_total", "Total spans that were sampled and exported")
        self.exporter_errors = Counter("tracing_exporter_errors_total", "Errors observed when exporting spans")
        self.active_traces = Gauge("tracing_active_traces", "Number of traces currently active (approx)")

    def inc_span(self, sampled: bool = False):
        try:
            if self.spans_created:
                self.spans_created.inc()
            if sampled and self.spans_sampled:
                self.spans_sampled.inc()
        except Exception:
            logger.exception("inc_span failed")

    def inc_export_error(self):
        try:
            if self.exporter_errors:
                self.exporter_errors.inc()
        except Exception:
            logger.exception("inc_export_error failed")

    def set_active_traces(self, value: int):
        try:
            if self.active_traces:
                self.active_traces.set(value)
        except Exception:
            logger.exception("set_active_traces failed")
