# config.py
# Created by Create-SamplingFiles.ps1
"""
Configuration for tracing sampling.

Drive behavior with environment variables so the sampling strategy can be tuned at deploy time.
"""

import os


class SamplingConfig:
    # Default probabilistic sampling probability (0.0 - 1.0)
    DEFAULT_PROBABILITY: float = float(os.getenv("SAMPLING_PROBABILITY", "0.01"))

    # Reservoir sampling configs
    RESERVOIR_SIZE: int = int(os.getenv("RESERVOIR_SIZE", "1000"))  # how many traces to keep sampled in reservoir
    RESERVOIR_WINDOW_SECONDS: int = int(os.getenv("RESERVOIR_WINDOW_SECONDS", "60"))  # sliding window length

    # Adaptive sampler target traces per second to export
    ADAPTIVE_TARGET_TPS: float = float(os.getenv("ADAPTIVE_TARGET_TPS", "5.0"))
    ADAPTIVE_MIN_PROB: float = float(os.getenv("ADAPTIVE_MIN_PROB", "0.0001"))
    ADAPTIVE_MAX_PROB: float = float(os.getenv("ADAPTIVE_MAX_PROB", "1.0"))
    ADAPTIVE_ADJUST_STEP: float = float(os.getenv("ADAPTIVE_ADJUST_STEP", "0.1"))  # relative step for adjustments

    # Priority sampler
    PRIORITY_ATTRIBUTES = os.getenv("PRIORITY_ATTRIBUTES", "error,debug,force_sample").split(",")

    # Metrics / instrumentation
    EXPORT_PROM_METRICS: bool = os.getenv("SAMPLER_EXPORT_PROM_METRICS", "true").lower() == "true"
