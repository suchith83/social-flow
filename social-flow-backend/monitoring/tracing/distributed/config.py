# config.py
# Created by Create-DistributedFiles.ps1
"""
Configuration for distributed tracing module.
Environment-driven with sensible production defaults.
"""

import os
from typing import Optional

class DistributedConfig:
    # enable/disable distributed tracing
    ENABLED: bool = os.getenv("DIST_TRACING_ENABLED", "true").lower() == "true"

    # exporter for distributed traces (otlp/grpc/http/console)
    EXPORTER_BACKEND: str = os.getenv("DIST_TRACING_EXPORTER", "otlp")  # otlp | console | none
    OTLP_ENDPOINT: Optional[str] = os.getenv("DIST_TRACING_OTLP_ENDPOINT", None)  # e.g., "http://otel-collector:4317"
    OTLP_INSECURE: bool = os.getenv("DIST_TRACING_OTLP_INSECURE", "true").lower() == "true"

    # sampling: choose conservative default
    SAMPLER: str = os.getenv("DIST_TRACING_SAMPLER", "probabilistic")  # always_on | always_off | probabilistic
    SAMPLER_PROBABILITY: float = float(os.getenv("DIST_TRACING_SAMPLER_PROBABILITY", "0.01"))

    # headers for propagation (defaults include W3C + B3 for compatibility)
    PROPAGATION_HEADERS: str = os.getenv("DIST_TRACING_PROP_HEADERS", "traceparent,tracestate,b3")

    # instrumentations: auto-instrument common libs if desired
    AUTO_INSTRUMENT: bool = os.getenv("DIST_TRACING_AUTO_INSTRUMENT", "false").lower() == "true"

    # maximum linkable spans kept in memory for batch linking
    MAX_LINK_HISTORY: int = int(os.getenv("DIST_TRACING_MAX_LINK_HISTORY", "500"))

    # metrics: sample percent of spans to emit metrics for to reduce cardinality
    METRICS_SAMPLE_RATE: float = float(os.getenv("DIST_TRACING_METRICS_RATE", "0.05"))

    @classmethod
    def summary(cls):
        return {k: getattr(cls, k) for k in dir(cls) if k.isupper()}
