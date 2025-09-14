# utils.py
# Created automatically by Create-CorrelationFiles.ps1
"""
Utility helpers for trace correlation and logging integration.

Includes:
 - helpers to build traceparent header values for custom instrumentation (best-effort)
 - log record processor example for injecting trace ids into Python logging records (for logging frameworks without OTel integration)
"""

import logging
from typing import Optional

try:
    from opentelemetry.trace import get_current_span, SpanContext, INVALID_SPAN_CONTEXT
    OTel_AVAILABLE = True
except Exception:
    OTel_AVAILABLE = False

logger = logging.getLogger("tracing.correlation.utils")


def current_trace_id_hex() -> Optional[str]:
    """
    Return the current trace id as hex string (32 chars) or None.
    """
    if not OTel_AVAILABLE:
        return None
    try:
        span = get_current_span()
        if span is None:
            return None
        ctx = span.get_span_context()
        if not ctx or not ctx.is_valid:
            return None
        return format(ctx.trace_id, "032x")
    except Exception:
        return None


def current_span_id_hex() -> Optional[str]:
    """Return current span id as hex (16 chars) or None."""
    if not OTel_AVAILABLE:
        return None
    try:
        span = get_current_span()
        if span is None:
            return None
        ctx = span.get_span_context()
        if not ctx or not ctx.is_valid:
            return None
        return format(ctx.span_id, "016x")
    except Exception:
        return None


class TraceIdLogFilter(logging.Filter):
    """
    Logging filter that injects trace_id and span_id into LogRecord.extra fields if available.
    Add to logging config: logger.addFilter(TraceIdLogFilter())
    """
    def filter(self, record):
        try:
            tid = current_trace_id_hex()
            sid = current_span_id_hex()
            if tid:
                record.trace_id = tid
            else:
                record.trace_id = None
            if sid:
                record.span_id = sid
            else:
                record.span_id = None
        except Exception:
            record.trace_id = None
            record.span_id = None
        return True
