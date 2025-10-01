# distributed_tracer.py
# Created by Create-DistributedFiles.ps1
"""
Distributed tracer lifecycle and helpers.

Provides:
 - DistributedTracer: initialize OTLP exporter, configure sampler, add processors
 - ensure_tracer: decorator/helper to get a tracer or noop tracer
 - defensive behavior if OpenTelemetry is not installed
"""

import logging
from typing import Optional

from .config import DistributedConfig

logger = logging.getLogger("tracing.distributed.distributed_tracer")

try:
    from opentelemetry import trace
    from opentelemetry.sdk.resources import Resource, SERVICE_NAME
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
    from opentelemetry.sdk.trace.sampling import TraceIdRatioBased, ALWAYS_ON, ALWAYS_OFF
    # OTLP exporter (grpc)
    try:
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
        OTLP_AVAILABLE = True
    except Exception:
        OTLP_AVAILABLE = False
    OTel_AVAILABLE = True
except Exception:
    # Provide no-op fallbacks
    trace = None
    TracerProvider = None
    OTLP_AVAILABLE = False
    OTel_AVAILABLE = False
    logger.warning("OpenTelemetry SDK not available — tracing will be no-op.")


_tracer_provider = None
_span_processor = None


def _build_resource(service_name: str = "service"):
    if not OTel_AVAILABLE:
        return None
    try:
        return Resource.create({SERVICE_NAME: service_name})
    except Exception:
        return None


class DistributedTracer:
    """
    Centralized tracer initializer for distributed tracing.

    Usage:
      dt = DistributedTracer(service_name="payments", exporter_endpoint="http://otel-collector:4317")
      dt.start()
      tracer = dt.get_tracer("module")
      # at shutdown: dt.shutdown()
    """

    def __init__(self,
                 service_name: str,
                 exporter_backend: Optional[str] = None,
                 otlp_endpoint: Optional[str] = None,
                 sampler: Optional[object] = None):
        self.service_name = service_name
        self.exporter_backend = exporter_backend or DistributedConfig.EXPORTER_BACKEND
        self.otlp_endpoint = otlp_endpoint or DistributedConfig.OTLP_ENDPOINT
        self.sampler = sampler
        self._started = False

    def start(self):
        global _tracer_provider, _span_processor
        if not DistributedConfig.ENABLED:
            logger.info("Distributed tracing disabled by config")
            return

        if not OTel_AVAILABLE:
            logger.warning("OpenTelemetry not installed; start is noop")
            return

        if _tracer_provider is not None:
            logger.debug("Tracer provider already started")
            return

        # choose sampler
        sampler_obj = None
        if self.sampler is not None:
            sampler_obj = self.sampler
        else:
            s = DistributedConfig.SAMPLER
            if s == "always_on":
                sampler_obj = ALWAYS_ON
            elif s == "always_off":
                sampler_obj = ALWAYS_OFF
            else:
                sampler_obj = TraceIdRatioBased(DistributedConfig.SAMPLER_PROBABILITY)

        # create provider with resource
        resource = _build_resource(self.service_name)
        _tracer_provider = TracerProvider(resource=resource, sampler=sampler_obj)
        trace.set_tracer_provider(_tracer_provider)

        # set exporter
        if self.exporter_backend == "console":
            exporter = ConsoleSpanExporter()
            _span_processor = BatchSpanProcessor(exporter)
        elif self.exporter_backend == "otlp" and OTLP_AVAILABLE and self.otlp_endpoint:
            exporter = OTLPSpanExporter(endpoint=self.otlp_endpoint, insecure=DistributedConfig.OTLP_INSECURE)
            _span_processor = BatchSpanProcessor(exporter)
        elif self.exporter_backend == "otlp" and not OTLP_AVAILABLE:
            logger.warning("OTLP exporter not available (package not installed) — falling back to console")
            exporter = ConsoleSpanExporter()
            _span_processor = BatchSpanProcessor(exporter)
        else:
            # default: console
            exporter = ConsoleSpanExporter()
            _span_processor = BatchSpanProcessor(exporter)

        _tracer_provider.add_span_processor(_span_processor)
        self._started = True
        logger.info("DistributedTracer started (service=%s backend=%s)", self.service_name, self.exporter_backend)

    def get_tracer(self, name: str):
        if not OTel_AVAILABLE or not DistributedConfig.ENABLED:
            # return noop tracer
            class Noop:
                def start_as_current_span(self, *args, **kwargs):
                    class Ctx:
                        def __enter__(self_self): return None
                        def __exit__(self_self, *a): pass
                    return Ctx()
            return Noop()
        return trace.get_tracer(name)

    def shutdown(self):
        global _tracer_provider, _span_processor
        if not OTel_AVAILABLE:
            return
        if _span_processor:
            try:
                _span_processor.shutdown()
            except Exception:
                logger.exception("Error shutting span processor")
        _tracer_provider = None
        _span_processor = None
        logger.info("DistributedTracer shutdown complete")


# Convenience: ensure a tracer exists
_default_tracer: Optional[DistributedTracer] = None

def ensure_tracer(service_name: str = "service", exporter_endpoint: Optional[str] = None):
    global _default_tracer
    if _default_tracer is None:
        _default_tracer = DistributedTracer(service_name=service_name, otlp_endpoint=exporter_endpoint)
        _default_tracer.start()
    return _default_tracer
