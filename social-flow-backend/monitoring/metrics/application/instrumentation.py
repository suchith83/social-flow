# Instrumentation hooks for tracing and metrics
"""
Automatic instrumentation for functions and web frameworks.
"""

import time
from functools import wraps
from .metrics_collector import MetricsCollector

collector = MetricsCollector()


def instrument_function(endpoint: str, method: str = "custom"):
    """
    Decorator to measure latency, request count, and errors for any function.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                duration = time.perf_counter() - start
                collector.observe_request(endpoint, method, "200", duration)
                return result
            except Exception as e:
                duration = time.perf_counter() - start
                collector.observe_request(endpoint, method, "500", duration)
                collector.record_error(type(e).__name__)
                raise
        return wrapper
    return decorator


class RequestMetricsMiddleware:
    """
    Example middleware for ASGI apps (FastAPI, Starlette, etc.).
    Tracks request latency, status codes, and errors.
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)

        start = time.perf_counter()
        method = scope["method"]
        path = scope["path"]

        async def send_wrapper(response):
            status = str(response["status"])
            duration = time.perf_counter() - start
            collector.observe_request(path, method, status, duration)
            await send(response)

        try:
            await self.app(scope, receive, send_wrapper)
        except Exception as e:
            duration = time.perf_counter() - start
            collector.observe_request(path, method, "500", duration)
            collector.record_error(type(e).__name__)
            raise
