# otel_wrappers.py
# Created by Create-SamplingFiles.ps1
"""
Lightweight wrappers to create OpenTelemetry-compatible samplers.
These return sampler objects when the OpenTelemetry SDK is available,
otherwise return None (calling code should handle None).
"""

import logging
from .config import SamplingConfig

logger = logging.getLogger("tracing.sampling.otel_wrappers")

try:
    from opentelemetry.sdk.trace.sampling import TraceIdRatioBased, ALWAYS_ON, ALWAYS_OFF
    OTEL_SAMPLERS_AVAILABLE = True
except Exception:
    OTEL_SAMPLERS_AVAILABLE = False
    logger.debug("OpenTelemetry sampler classes not available; wrapper will return None.")


def ProbabilisticSamplerWrapper(prob: float = None):
    """
    Return an OpenTelemetry TraceIdRatioBased sampler instance configured with 'prob'.
    If OTel SDK not present, returns None.
    """
    prob = SamplingConfig.DEFAULT_PROBABILITY if prob is None else prob
    if not OTEL_SAMPLERS_AVAILABLE:
        logger.debug("OTel samplers not available; ProbabilisticSamplerWrapper returning None")
        return None
    try:
        return TraceIdRatioBased(float(prob))
    except Exception:
        logger.exception("Failed to create TraceIdRatioBased sampler")
        return None


def AlwaysOnSamplerWrapper():
    if not OTEL_SAMPLERS_AVAILABLE:
        return None
    return ALWAYS_ON


def AlwaysOffSamplerWrapper():
    if not OTEL_SAMPLERS_AVAILABLE:
        return None
    return ALWAYS_OFF
