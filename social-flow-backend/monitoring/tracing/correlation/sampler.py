# sampler.py
# Created automatically by Create-CorrelationFiles.ps1
"""
Custom sampler helpers and simple wrappers for OpenTelemetry samplers.

Provides:
 - ProbabilisticSampler: wrapper around OTel TraceIdRatioBased sampler
 - PrioritySampler: simple priority sampling to force sampling for specific traces (e.g., error)
Note: OpenTelemetry SDK provides robust sampling; these wrappers help integrate env-configured choices.
"""

import logging
from .config import CorrelationConfig

logger = logging.getLogger("tracing.correlation.sampler")

try:
    from opentelemetry.sdk.trace.sampling import TraceIdRatioBased, ALWAYS_ON, ALWAYS_OFF
    OTel_SAMPLING_AVAILABLE = True
except Exception:
    OTel_SAMPLING_AVAILABLE = False
    logger.warning("OpenTelemetry samplers not available; sampling will default to SDK behavior.")


def ProbabilisticSampler(prob: float = None):
    """Return a probabilistic sampler for SDK initialization."""
    prob = CorrelationConfig.SAMPLER_PROBABILITY if prob is None else prob
    if not OTel_SAMPLING_AVAILABLE:
        logger.debug("Sampler not available, returning None")
        return None
    return TraceIdRatioBased(prob)


def PrioritySampler(kind: str = None):
    """Map config string to sampler objects (ALWAYS_ON / OFF / Probabilistic)"""
    kind = kind or CorrelationConfig.SAMPLER
    if not OTel_SAMPLING_AVAILABLE:
        return None
    if kind == "always_on":
        return ALWAYS_ON
    if kind == "always_off":
        return ALWAYS_OFF
    if kind == "probabilistic":
        return ProbabilisticSampler()
    # fallback
    return ProbabilisticSampler()
