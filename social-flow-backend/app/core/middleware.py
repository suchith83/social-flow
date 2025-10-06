"""Custom FastAPI middlewares (request context, logging enrichment)."""
from __future__ import annotations

import time
import contextvars

from app.application.container import get_container

_request_id_ctx: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="")


def get_request_id() -> str:
    return _request_id_ctx.get()


class RequestContextMiddleware:
    """Inject a request id + measure latency; attach to app.state for downstream use."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):  # ASGI signature
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        container = get_container()
        rid = container.request_id()
        token = _request_id_ctx.set(rid)
        start = time.perf_counter()
        try:
            async def send_wrapper(message):
                if message.get("type") == "http.response.start":
                    headers = list(message.get("headers", []))
                    headers.append((b"x-request-id", rid.encode()))
                    message["headers"] = headers
                await send(message)
            await self.app(scope, receive, send_wrapper)
        finally:
            duration_ms = (time.perf_counter() - start) * 1000.0
            # Lightweight logging via stdlib to avoid recursion
            import logging
            logging.getLogger("app.request").info(
                "request_complete", extra={"request_id": rid, "path": scope.get("path"), "duration_ms": round(duration_ms, 2)}
            )
            _request_id_ctx.reset(token)


__all__ = ["RequestContextMiddleware", "get_request_id"]
