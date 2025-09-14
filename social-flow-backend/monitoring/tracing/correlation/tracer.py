# tracer.py
# Created automatically by Create-CorrelationFiles.ps1
"""
Tracer setup and lifecycle helpers.

Uses OpenTelemetry SDK (if available). This file creates the tracer provider,
applies resource (service.name), configures sampler and exporters, and exposes
get_tracer to get named tracers consistently.

Note: This module is defensive — if OpenTelemetry dependencies are not present
it will provide no-op fallbacks so importing modules doesn't crash tests.
"""

import logging
from typing import Optional

from .config import CorrelationConfig

logger = logging.getLogger("tracing.correlation.tracer")

try:
    from opentelemetry import trace
    from opentelemetry.sdk.resources import Resource, SERVICE_NAME, SERVICE_VERSION
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
    from opentelemetry.propagate import set_global_textmap, get_global_textmap
    from opentelemetry.instrumentation.logging import LoggingInstrumentor
    OTel_AVAILABLE = True
except Exception:
    OTel_AVAILABLE = False
    logger.warning("OpenTelemetry SDK not available; tracing will be noop.")


# Keep module-level provider references for shutdown
_tracer_provider = None
_span_processor = None


def _build_resource():
    """Construct OpenTelemetry Resource with service attributes."""
    if not OTel_AVAILABLE:
        return None
    attrs = {SERVICE_NAME: CorrelationConfig.SERVICE_NAME}
    if CorrelationConfig.SERVICE_VERSION:
        attrs[SERVICE_VERSION] = CorrelationConfig.SERVICE_VERSION
    return Resource(attributes=attrs)


def get_tracer(name: str = __name__):
    """
    Return a tracer instance. If OTel not present, returns a no-op tracer proxy.
    """
    if not CorrelationConfig.ENABLE_TRACING:
        logger.debug("Tracing disabled by configuration.")
        if OTel_AVAILABLE:
            return trace.get_tracer_provider().get_tracer(name)
        else:
            class NoopTracer:
                def start_as_current_span(self, *a, **k):
                    class NoopCtx:
                        def __enter__(self_self): return None
                        def __exit__(self_self, exc_type, exc, tb): pass
                    return NoopCtx()
                def start_span(self, *a, **k):
                    return None
            return NoopTracer()
    if not OTel_AVAILABLE:
        logger.debug("OpenTelemetry not installed; returning noop tracer")
        class NoopTracer:
            def start_as_current_span(self, *a, **k):
                class NoopCtx:
                    def __enter__(self_self): return None
                    def __exit__(self_self, exc_type, exc, tb): pass
                return NoopCtx()
        return NoopTracer()
    return trace.get_tracer(__name__ if name is None else name)


def init_tracer(exporter_backend: Optional[str] = None, sampler=None):
    """
    Initialize a global tracer provider with resource, sampler and exporter(s).
    exporter_backend: overrides CorrelationConfig.EXPORTER_BACKEND (otlp/console)
    sampler: optional sampler instance from opentelemetry.sdk.trace.sampling
    """
    global _tracer_provider, _span_processor
    if not OTel_AVAILABLE:
        logger.debug("OpenTelemetry not available; skipping tracer initialization")
        return

    if _tracer_provider:
        logger.debug("Tracer provider already initialized")
        return

    exporter_backend = exporter_backend or CorrelationConfig.EXPORTER_BACKEND
    resource = _build_resource()

    # create tracer provider with sampler
    tp_kwargs = {"resource": resource} if resource is not None else {}
    if sampler is not None:
        tp_kwargs["sampler"] = sampler

    _tracer_provider = TracerProvider(**tp_kwargs)
    trace.set_tracer_provider(_tracer_provider)

    # set up exporter and batch span processor
    if exporter_backend == "console":
        exporter = ConsoleSpanExporter()
        _span_processor = BatchSpanProcessor(exporter)
        _tracer_provider.add_span_processor(_span_processor)
        logger.info("Initialized ConsoleSpanExporter for tracing")
    elif exporter_backend == "otlp":
        # otlp exporter is created in exporters.init_otlp_exporter()
        try:
            from .exporters import init_otlp_exporter
            processor = init_otlp_exporter()
            if processor:
                _span_processor = processor
                _tracer_provider.add_span_processor(_span_processor)
                logger.info("Initialized OTLP exporter for tracing")
        except Exception:
            logger.exception("Failed to initialize OTLP exporter; falling back to console")
            exporter = ConsoleSpanExporter()
            _span_processor = BatchSpanProcessor(exporter)
            _tracer_provider.add_span_processor(_span_processor)
    else:
        logger.warning("Unknown exporter backend '%s', defaulting to console", exporter_backend)
        exporter = ConsoleSpanExporter()
        _span_processor = BatchSpanProcessor(exporter)
        _tracer_provider.add_span_processor(_span_processor)

    # Optional: integrate trace ids into structured logs
    if CorrelationConfig.INJECT_LOG_TRACE_IDS:
        try:
            LoggingInstrumentor().instrument(set_logging_format=True)
            logger.info("LoggingInstrumentor enabled: trace ids will be injected into logs")
        except Exception:
            logger.exception("Failed to enable LoggingInstrumentor")

    # Setup default propagation (done in context_propagation usually)
    logger.info("Tracer initialized (service=%s)", CorrelationConfig.SERVICE_NAME)


def shutdown_tracer():
    """Shutdown span processor(s) and flush exporters."""
    global _tracer_provider, _span_processor
    if not OTel_AVAILABLE or not _tracer_provider:
        return
    try:
        if _span_processor:
            _span_processor.shutdown()
        # no explicit shutdown for TracerProvider required in SDK currently
        _tracer_provider = None
        _span_processor = None
        logger.info("Tracer provider shutdown complete")
    except Exception:
        logger.exception("Error shutting down tracer")
