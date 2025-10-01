# config.py
# Created automatically by Create-CorrelationFiles.ps1
"""
Configuration for tracing/correlation.

Read-only dataclass style configuration driven by environment variables.
"""

import os
from typing import Optional


class CorrelationConfig:
    # General tracing enablement
    ENABLE_TRACING: bool = os.getenv("ENABLE_TRACING", "true").lower() == "true"

    # Service identity (used in resource/service.name)
    SERVICE_NAME: str = os.getenv("TRACING_SERVICE_NAME", "my-service")
    SERVICE_VERSION: Optional[str] = os.getenv("TRACING_SERVICE_VERSION", None)

    # Exporter backend: "otlp" | "console"
    EXPORTER_BACKEND: str = os.getenv("TRACING_EXPORTER_BACKEND", "console")
    OTLP_ENDPOINT: Optional[str] = os.getenv("OTLP_ENDPOINT", None)  # e.g., "http://otel-collector:4317"

    # Sampling
    SAMPLER: str = os.getenv("TRACING_SAMPLER", "probabilistic")  # "always_on" | "always_off" | "probabilistic"
    SAMPLER_PROBABILITY: float = float(os.getenv("TRACING_SAMPLER_PROBABILITY", "0.01"))

    # Context propagation preferences
    PROPAGATORS: str = os.getenv("TRACING_PROPAGATORS", "tracecontext,baggage")  # comma separated

    # Trace header names to capture for convenience (supports custom headers)
    TRACE_HEADER_NAMES = os.getenv("TRACING_HEADER_NAMES", "traceparent,tracestate,b3").split(",")

    # Logging integration
    INJECT_LOG_TRACE_IDS: bool = os.getenv("TRACING_INJECT_LOG_TRACE_IDS", "true").lower() == "true"

    @classmethod
    def summary(cls) -> dict:
        return {k: getattr(cls, k) for k in dir(cls) if k.isupper()}
