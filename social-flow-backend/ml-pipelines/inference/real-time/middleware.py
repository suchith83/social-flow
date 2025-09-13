# Logging, tracing, rate limiting
# ================================================================
# File: middleware.py
# Purpose: Logging, tracing, and rate limiting middleware
# ================================================================

import time
import logging
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger("Middleware")


class LoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        response = await call_next(request)
        process_time = time.time() - start_time
        logger.info(f"{request.method} {request.url.path} - {process_time:.4f}s")
        return response


def setup_middleware(app, config):
    app.add_middleware(LoggingMiddleware)
    # TODO: add rate limiting (Redis), tracing (OpenTelemetry)
