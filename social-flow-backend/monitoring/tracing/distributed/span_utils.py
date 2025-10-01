# span_utils.py
# Created by Create-DistributedFiles.ps1
"""
Utilities for advanced span operations:
 - link_spans: create links between spans when events connect different traces
 - retry_span: create a span that wraps retry logic and records attempts
 - tag_error: tag current span with structured error info
 - safe_set_attribute: set attribute on current span with error handling
"""

import logging
import time
from typing import Optional, Dict, Any

logger = logging.getLogger("tracing.distributed.span_utils")

try:
    from opentelemetry.trace import get_current_span, Link
    OTel_SPAN_AVAILABLE = True
except Exception:
    OTel_SPAN_AVAILABLE = False


def link_spans(span_contexts):
    """
    Return list of Link objects for provided span_contexts.
    span_contexts: iterable of objects with trace_id and span_id attributes (or mappings)
    """
    if not OTel_SPAN_AVAILABLE:
        return []
    links = []
    for ctx in span_contexts:
        try:
            if hasattr(ctx, "trace_id") and hasattr(ctx, "span_id"):
                links.append(Link(ctx))
            elif isinstance(ctx, dict):
                # support dicts: {'trace_id': int, 'span_id': int}
                from opentelemetry.trace import SpanContext
                sc = SpanContext(trace_id=int(ctx["trace_id"]), span_id=int(ctx["span_id"]), is_remote=True)
                links.append(Link(sc))
        except Exception:
            logger.debug("link_spans: invalid ctx %s", ctx)
    return links


class retry_span:
    """
    Context manager for retry-aware spans.

    Example:
      with retry_span(tracer, "send_request", max_attempts=3) as rs:
          do network call
          if failed and will_retry:
              rs.attempt_failed(error=exc)
    The span will record attributes: retry.attempts, retry.succeeded, retry.duration_ms
    """
    def __init__(self, tracer, name: str, max_attempts: int = 3):
        self.tracer = tracer
        self.name = name
        self.max_attempts = max_attempts
        self.attempts = 0
        self.start_ts = None
        self._span = None

    def __enter__(self):
        self.start_ts = time.time()
        self._span = self.tracer.start_as_current_span(self.name).__enter__()
        try:
            self._span.set_attribute("retry.max_attempts", self.max_attempts)
        except Exception:
            pass
        return self

    def attempt_failed(self, error: Optional[Exception] = None):
        self.attempts += 1
        try:
            self._span.set_attribute(f"retry.attempt_{self.attempts}_failed", True)
            if error is not None:
                self._span.record_exception(error)
        except Exception:
            pass

    def __exit__(self, exc_type, exc, tb):
        duration_ms = int((time.time() - self.start_ts) * 1000) if self.start_ts else 0
        try:
            self._span.set_attribute("retry.attempts", self.attempts)
            self._span.set_attribute("retry.duration_ms", duration_ms)
            self._span.set_attribute("retry.succeeded", exc_type is None)
        except Exception:
            pass
        try:
            self.tracer.start_as_current_span(self.name).__exit__(exc_type, exc_type and exc_type(), tb)
        except Exception:
            pass


def tag_error(err: Exception, attrs: Optional[Dict[str, Any]] = None):
    """Attach structured error information to the current span."""
    if not OTel_SPAN_AVAILABLE:
        return
    span = get_current_span()
    if span is None:
        return
    try:
        span.record_exception(err)
        span.set_attribute("error.message", str(err))
        span.set_attribute("error.type", type(err).__name__)
        if attrs:
            for k, v in attrs.items():
                try:
                    span.set_attribute(k, v)
                except Exception:
                    pass
    except Exception:
        logger.exception("tag_error failed")


def safe_set_attribute(key: str, value):
    """Set attribute on current span safely (no exceptions to caller)."""
    if not OTel_SPAN_AVAILABLE:
        return
    span = get_current_span()
    if span is None:
        return
    try:
        span.set_attribute(key, value)
    except Exception:
        logger.debug("safe_set_attribute failed for %s", key)
