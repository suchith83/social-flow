# Custom middleware for API processing
import time
from .monitoring import APIMetricsCollector


class OptimizationMiddleware:
    """
    Middleware wrapper for API frameworks (e.g., FastAPI).
    - Collects metrics
    - Applies compression
    - Rate limiting & throttling hooks
    """

    def __init__(self, app, metrics: APIMetricsCollector):
        self.app = app
        self.metrics = metrics

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        route = scope.get("path", "unknown")
        start = time.perf_counter()
        success = True

        async def wrapped_send(message):
            nonlocal success
            if message["type"] == "http.response.start":
                if message["status"] >= 400:
                    success = False
            await send(message)

        await self.app(scope, receive, wrapped_send)

        latency = time.perf_counter() - start
        self.metrics.record_request(route, latency, success)
