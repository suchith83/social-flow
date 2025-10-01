# http_instrumentation.py
# Created by Create-DistributedFiles.ps1
"""
HTTP instrumentation helpers for distributed tracing.

Contains:
 - client_inject_headers: convenience for injecting trace context into outgoing HTTP requests (requests, httpx)
 - ASGI middleware for server-side context extraction + automatic span creation for incoming requests
 - Optional small wrappers for httpx request sending to automatically create child spans
"""

import logging
import typing
from typing import Callable, Dict

from .distributed_tracer import ensure_tracer
from .config import DistributedConfig

logger = logging.getLogger("tracing.distributed.http_instrumentation")

try:
    from opentelemetry.propagate import inject, extract
    from opentelemetry.trace import get_current_span, set_span_in_context
    PROP_AVAILABLE = True
except Exception:
    PROP_AVAILABLE = False
    logger.debug("OTel propagation not available — http instrumentation will be limited.")


def client_inject_headers(headers: Dict[str, str] = None):
    """
    Inject current trace context into headers dict (mutates and returns it).
    Use this before performing an outbound HTTP call.
    """
    headers = headers or {}
    if not PROP_AVAILABLE:
        return headers
    try:
        inject(headers)
    except Exception:
        logger.exception("Failed to inject trace headers")
    return headers


class ASGITraceMiddleware:
    """
    ASGI middleware that extracts incoming trace context (if present) and creates a server span.
    Compatible with FastAPI/Starlette/Quart-like ASGI apps.

    Usage:
      app.add_middleware(ASGITraceMiddleware)
    """
    def __init__(self, app, service_name: str = "service"):
        self.app = app
        ensure_tracer(service_name)

    async def __call__(self, scope, receive, send):
        if scope.get("type") != "http":
            return await self.app(scope, receive, send)

        headers = {}
        for k, v in scope.get("headers", []):
            try:
                headers[k.decode("utf-8")] = v.decode("utf-8")
            except Exception:
                continue

        tracer = ensure_tracer().get_tracer("http.server")
        if PROP_AVAILABLE:
            ctx = extract(headers)
        else:
            ctx = None

        path = scope.get("path", "/")
        method = scope.get("method", "GET")
        span_name = f"{method} {path}"

        # start span with extracted context if available
        try:
            with tracer.start_as_current_span(span_name, context=ctx if ctx is not None else None) as span:
                # set common attributes
                try:
                    span.set_attribute("http.method", method)
                    span.set_attribute("http.target", path)
                    span.set_attribute("http.scheme", scope.get("scheme", "http"))
                except Exception:
                    pass
                await self.app(scope, receive, send)
        except Exception as exc:
            # record exception in current span if possible, then re-raise
            try:
                cur = get_current_span()
                if cur is not None:
                    cur.record_exception(exc)
            except Exception:
                pass
            raise


# Optional helper for httpx to auto-create spans around requests (sync/async)
def httpx_request_span_wrapper(client, tracer_name="http.client"):
    """
    Wrap an httpx.Client instance to create a span for each request (works with sync clients).
    Example:
      client = httpx.Client()
      wrapped = httpx_request_span_wrapper(client)
      resp = wrapped.get("https://example.com")
    This is a minimal wrapper — prefer OpenTelemetry httpx instrumentation in production.
    """
    try:
        import httpx
    except Exception:
        logger.debug("httpx not installed; wrapper not available")
        return client

    tracer = ensure_tracer().get_tracer(tracer_name)

    class ClientWrapper:
        def __init__(self, inner):
            self._inner = inner

        def request(self, method, url, *args, **kwargs):
            headers = kwargs.get("headers", {}) or {}
            # inject trace context
            client_inject_headers(headers)
            kwargs["headers"] = headers
            span_name = f"{method.upper()} {url}"
            with tracer.start_as_current_span(span_name) as span:
                try:
                    span.set_attribute("http.method", method.upper())
                    span.set_attribute("http.url", url)
                except Exception:
                    pass
                return self._inner.request(method, url, *args, **kwargs)

        # delegate other attributes
        def __getattr__(self, item):
            return getattr(self._inner, item)

    return ClientWrapper(client)
