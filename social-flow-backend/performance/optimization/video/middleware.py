# Custom middleware for video processing
"""
middleware.py

ASGI middleware that can:
- Record metrics for segment responses
- Apply simple caching lookups before handing off to the application
- Add headers used by downstream CDNs (cache-control, x-variant, etc)

This is intentionally lightweight and compatible with FastAPI/Starlette/Django-ASGI apps.
"""

from typing import Callable
from .monitoring import VideoMetricsCollector
from .caching import SegmentCache
import time
import asyncio


class VideoOptimizationMiddleware:
    """
    ASGI middleware for video optimizations.

    Example usage:
        app.add_middleware(VideoOptimizationMiddleware, cache=segment_cache, metrics=metrics)
    """

    def __init__(self, app, cache: SegmentCache = None, metrics: VideoMetricsCollector = None):
        self.app = app
        self.cache = cache
        self.metrics = metrics

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        path = scope.get("path", "")
        # Only short-circuit for typical segment requests (.ts, .m4s, .mp4)
        if self.cache and any(path.endswith(ext) for ext in (".ts", ".m4s", ".mp4", ".frag")):
            cached = await self.cache.get(path)
            if cached is not None:
                # Serve cached bytes directly via ASGI send events
                if self.metrics:
                    self.metrics.record_cache_hit()
                start = time.time()
                await send({
                    "type": "http.response.start",
                    "status": 200,
                    "headers": [
                        (b"content-type", b"video/MP2T" if path.endswith(".ts") else b"video/mp4"),
                        (b"cache-control", b"public, max-age=300"),
                    ]
                })
                await send({
                    "type": "http.response.body",
                    "body": cached,
                    "more_body": False
                })
                if self.metrics:
                    self.metrics.record_segment_latency(time.time() - start)
                return
            else:
                if self.metrics:
                    self.metrics.record_cache_miss()

        # Otherwise, instrument downstream app
        start = time.time()
        sent_first = False

        async def wrapped_send(message):
            nonlocal sent_first
            if message["type"] == "http.response.start":
                sent_first = True
            if message["type"] == "http.response.body" and not message.get("more_body", False):
                # last chunk; record total duration
                if self.metrics:
                    self.metrics.record_segment_latency(time.time() - start)
            await send(message)

        await self.app(scope, receive, wrapped_send)
