# grpc_instrumentation.py
# Created by Create-DistributedFiles.ps1
"""
gRPC interceptors for tracing.

Provides:
 - ServerInterceptor: extracts incoming context, starts server spans, records metadata
 - ClientInterceptor: injects trace context into outgoing RPC metadata and creates client spans

This code is defensive and will no-op if grpc / OpenTelemetry grpc instrumentation not available.
"""

import logging

logger = logging.getLogger("tracing.distributed.grpc_instrumentation")

try:
    import grpc
    GRPC_AVAILABLE = True
except Exception:
    grpc = None
    GRPC_AVAILABLE = False
    logger.debug("grpc not installed; grpc instrumentation disabled.")

try:
    from opentelemetry.propagate import inject, extract
    from opentelemetry.trace import get_current_span
    PROP_AVAILABLE = True
except Exception:
    PROP_AVAILABLE = False

from .distributed_tracer import ensure_tracer

# Server interceptor
if GRPC_AVAILABLE:
    class OpenTelemetryServerInterceptor(grpc.ServerInterceptor):
        def __init__(self, service_name: str = "service"):
            ensure_tracer(service_name)
            self._tracer = ensure_tracer().get_tracer("grpc.server")

        def intercept_service(self, continuation, handler_call_details):
            # extract metadata to dict
            metadata = {}
            if handler_call_details.invocation_metadata:
                for md in handler_call_details.invocation_metadata:
                    metadata[md.key] = md.value

            if PROP_AVAILABLE:
                ctx = extract(metadata)
            else:
                ctx = None

            def new_behavior(request, context):
                span_name = f"gRPC {handler_call_details.method}"
                with self._tracer.start_as_current_span(span_name, context=ctx) as span:
                    try:
                        span.set_attribute("rpc.system", "grpc")
                        span.set_attribute("rpc.service", handler_call_details.method)
                    except Exception:
                        pass
                    return continuation(handler_call_details).unary_unary(request, context)

            # Return wrapped handler; this is simplified for unary-unary handlers.
            return grpc.unary_unary_rpc_method_handler(new_behavior)
else:
    OpenTelemetryServerInterceptor = None

# Client interceptor
if GRPC_AVAILABLE:
    class OpenTelemetryClientInterceptor(grpc.UnaryUnaryClientInterceptor,
                                         grpc.UnaryStreamClientInterceptor,
                                         grpc.StreamUnaryClientInterceptor,
                                         grpc.StreamStreamClientInterceptor):
        def __init__(self, service_name: str = "service"):
            ensure_tracer(service_name)
            self._tracer = ensure_tracer().get_tracer("grpc.client")

        def _inject_metadata(self, metadata):
            # metadata: list of (key, value)
            md_dict = {k: v for k, v in metadata} if metadata else {}
            if PROP_AVAILABLE:
                inject(md_dict)
            # convert back to list
            return list(md_dict.items())

        def intercept_unary_unary(self, continuation, client_call_details, request):
            metadata = client_call_details.metadata or []
            metadata = self._inject_metadata(metadata)
            new_details = _replace_client_call_details(client_call_details, metadata)
            span_name = f"gRPC {client_call_details.method}"
            with self._tracer.start_as_current_span(span_name) as span:
                try:
                    span.set_attribute("rpc.system", "grpc")
                except Exception:
                    pass
                return continuation(new_details, request)

        # For brevity, other intercept methods can delegate to the unary_unary implementation or be implemented similarly.
else:
    OpenTelemetryClientInterceptor = None

# helper to replace client_call_details (gRPC internals)
def _replace_client_call_details(old, metadata):
    # gRPC client_call_details is a namedtuple-like; create a new one preserving attributes but replacing metadata
    try:
        return type(old)(old.method, old.timeout, metadata, old.credentials, old.wait_for_ready, old.compression)
    except Exception:
        # fallback: return old
        return old


class GRPCInstrumentor:
    """Facade exposing interceptors (server + client) if available."""
    def __init__(self, service_name: str = "service"):
        self.service_name = service_name
        self.server_interceptor = OpenTelemetryServerInterceptor(service_name) if GRPC_AVAILABLE and OpenTelemetryServerInterceptor else None
        self.client_interceptor = OpenTelemetryClientInterceptor(service_name) if GRPC_AVAILABLE and OpenTelemetryClientInterceptor else None
