# __init__.py
# Created automatically by Create-CorrelationFiles.ps1
"""
Tracing correlation utilities.

Exports:
 - CorrelationConfig: configuration
 - get_tracer: factory to get an instrumented OpenTelemetry tracer
 - ContextPropagator: HTTP header propagation helpers (W3C TraceContext, B3)
 - middleware helpers for ASGI and WSGI
 - span helpers (decorator/contextmanager)
 - Samplers and Exporter helpers
"""

from .config import CorrelationConfig
from .tracer import get_tracer, init_tracer, shutdown_tracer
from .context_propagation import ContextPropagator, extract_context_from_headers, inject_context_to_headers
from .span_helpers import traced, start_span, set_span_attributes
from .sampler import PrioritySampler, ProbabilisticSampler
from .exporters import init_otlp_exporter, init_console_exporter

__all__ = [
    "CorrelationConfig",
    "get_tracer",
    "init_tracer",
    "shutdown_tracer",
    "ContextPropagator",
    "extract_context_from_headers",
    "inject_context_to_headers",
    "traced",
    "start_span",
    "set_span_attributes",
    "PrioritySampler",
    "ProbabilisticSampler",
    "init_otlp_exporter",
    "init_console_exporter",
]
