# exporters.py
# Created automatically by Create-CorrelationFiles.ps1
"""
Exporter initialization helpers.

- init_otlp_exporter: sets up OTLP exporter to collector (gRPC or HTTP) and returns a span processor
- init_console_exporter: convenience for debugging
These are kept minimal and pluggable (no side effects on import).
"""

import logging
from typing import Optional

from .config import CorrelationConfig

logger = logging.getLogger("tracing.correlation.exporters")

try:
    # OTLP exporter dependencies
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter as OTLPGrpcSpanExporter
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    OTLP_AVAILABLE = True
except Exception:
    OTLP_AVAILABLE = False
    logger.info("OTLP exporter not available; OTLP export will be skipped.")


def init_otlp_exporter(endpoint: Optional[str] = None):
    """
    Initialize and return a BatchSpanProcessor configured with OTLP exporter.
    If OTLP exporter is unavailable or endpoint not provided, returns None.
    """
    if not OTLP_AVAILABLE:
        logger.warning("OTLP exporter not installed.")
        return None
    endpoint = endpoint or CorrelationConfig.OTLP_ENDPOINT
    if not endpoint:
        logger.warning("OTLP endpoint not configured; skipping OTLP exporter initialization.")
        return None
    try:
        exporter = OTLPGrpcSpanExporter(endpoint=endpoint)
        processor = BatchSpanProcessor(exporter)
        logger.info("Initialized OTLP exporter pointing to %s", endpoint)
        return processor
    except Exception:
        logger.exception("Failed to initialize OTLP exporter")
        return None


def init_console_exporter():
    """Console exporter is used by tracer.init_tracer via SDK's ConsoleSpanExporter; helper kept for completeness."""
    # no-op here because tracer.init_tracer uses SDK's ConsoleSpanExporter directly
    logger.debug("Console exporter helper invoked (no-op).")
    return None
