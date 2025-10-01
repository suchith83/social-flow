# span_helpers.py
# Created automatically by Create-CorrelationFiles.ps1
"""
Helpers for creating spans and decorating functions to be traced.

Includes:
 - traced decorator (sync + async aware)
 - start_span contextmanager wrapper
 - set_span_attributes utility
"""

import functools
import inspect
import logging
from contextlib import contextmanager
from typing import Optional, Dict, Any

from .tracer import get_tracer
from .config import CorrelationConfig

logger = logging.getLogger("tracing.correlation.span_helpers")


def _is_coroutine_function(fn):
    return inspect.iscoroutinefunction(fn) or inspect.isawaitable(fn)


def traced(span_name: Optional[str] = None, attributes: Optional[Dict[str, Any]] = None):
    """
    Decorator to trace a function. Works for sync and async functions.
    Usage:
      @traced("my_operation")
      async def do_work(...):
          ...
    """

    def decorator(func):
        tracer = get_tracer(func.__module__)

        if inspect.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                name = span_name or f"{func.__module__}.{func.__name__}"
                with tracer.start_as_current_span(name) as span:
                    try:
                        if attributes:
                            for k, v in attributes.items():
                                try:
                                    span.set_attribute(k, v)
                                except Exception:
                                    pass
                        return await func(*args, **kwargs)
                    except Exception as exc:
                        try:
                            span.record_exception(exc)
                            span.set_status(Exception)
                        except Exception:
                            pass
                        raise
            return async_wrapper

        else:
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                name = span_name or f"{func.__module__}.{func.__name__}"
                with tracer.start_as_current_span(name) as span:
                    try:
                        if attributes:
                            for k, v in attributes.items():
                                try:
                                    span.set_attribute(k, v)
                                except Exception:
                                    pass
                        return func(*args, **kwargs)
                    except Exception as exc:
                        try:
                            span.record_exception(exc)
                        except Exception:
                            pass
                        raise
            return sync_wrapper

    return decorator


@contextmanager
def start_span(name: str, attributes: Optional[Dict[str, Any]] = None):
    """
    Context manager to start a span; usage:
       with start_span("heavy_work"):
           do_work()
    Works as a no-op if tracing disabled / no-op tracer.
    """
    tracer = get_tracer()
    try:
        ctx = tracer.start_as_current_span(name)
        # enter context
        ctx.__enter__()
        span = None
        try:
            # if tracer provided a span object via context, attempt to set attributes
            span = hasattr(ctx, "__enter__") and None  # we don't have direct span object with noop
            if attributes:
                # best-effort set attributes on current span via API
                try:
                    from opentelemetry.trace import get_current_span
                    cs = get_current_span()
                    for k, v in attributes.items():
                        try:
                            cs.set_attribute(k, v)
                        except Exception:
                            pass
                except Exception:
                    pass
            yield
        finally:
            ctx.__exit__(None, None, None)
    except Exception:
        # If tracer is no-op, just yield
        yield


def set_span_attributes(attrs: Dict[str, Any]):
    """Set attributes on the current span (best-effort)."""
    try:
        from opentelemetry.trace import get_current_span
        span = get_current_span()
        if span is None:
            return
        for k, v in attrs.items():
            try:
                span.set_attribute(k, v)
            except Exception:
                pass
    except Exception:
        pass
