# __init__.py
# Created by Create-SamplingFiles.ps1
"""
Tracing sampling strategies package.

Exports:
 - SamplingConfig: env-configured sampling defaults
 - OTEL sampler wrappers (ProbabilisticSamplerWrapper)
 - ReservoirSampler: bounded reservoir sampling for traces
 - AdaptiveSampler: feedback-driven adaptive sampling to maintain target throughput
 - PrioritySampler: force-sampling based on span attributes (errors, debug flags)
 - utilities for observing sampler metrics
"""

from .config import SamplingConfig
from .otel_wrappers import ProbabilisticSamplerWrapper, AlwaysOnSamplerWrapper, AlwaysOffSamplerWrapper
from .reservoir_sampler import ReservoirSampler
from .adaptive_sampler import AdaptiveSampler
from .priority_sampler import PrioritySampler
from .utils import SamplingMetrics

__all__ = [
    "SamplingConfig",
    "ProbabilisticSamplerWrapper",
    "AlwaysOnSamplerWrapper",
    "AlwaysOffSamplerWrapper",
    "ReservoirSampler",
    "AdaptiveSampler",
    "PrioritySampler",
    "SamplingMetrics",
]
