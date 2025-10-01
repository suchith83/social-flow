# __init__.py
# Created by Create-DistributedFiles.ps1
"""
Distributed tracing helpers.

Exports:
 - DistributedConfig: tracing config
 - DistributedTracer: centralized tracer init/start/shutdown helpers
 - HTTPInstrumentor: request instrumentation helpers for clients & servers
 - GRPCInstrumentor: interceptors for grpc (client + server)
 - span utilities: link_spans, retry_span, tag_error, safe_set_attribute
 - baggage helpers: set/get baggage values safely across processes
 - metrics: lightweight counters for trace failures/sampled/spans_created
"""

from .config import DistributedConfig
from .distributed_tracer import DistributedTracer, ensure_tracer
from .http_instrumentation import HTTPInstrumentor
from .grpc_instrumentation import GRPCInstrumentor
from .span_utils import link_spans, retry_span, tag_error, safe_set_attribute
from .baggage_utils import set_baggage, get_baggage, propagate_baggage_headers
from .metrics import TraceMetrics

__all__ = [
    "DistributedConfig",
    "DistributedTracer",
    "ensure_tracer",
    "HTTPInstrumentor",
    "GRPCInstrumentor",
    "link_spans",
    "retry_span",
    "tag_error",
    "safe_set_attribute",
    "set_baggage",
    "get_baggage",
    "propagate_baggage_headers",
    "TraceMetrics",
]
