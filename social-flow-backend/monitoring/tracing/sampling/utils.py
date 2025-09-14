# utils.py
# Created by Create-SamplingFiles.ps1
"""
Utilities and lightweight metrics for samplers.

Exports SamplingMetrics: a thin wrapper around prometheus_client if present,
otherwise fallback to in-memory counters for observability in tests.
"""

import logging
import time

logger = logging.getLogger("tracing.sampling.utils")

try:
    from prometheus_client import Counter, Gauge
    PROM_AVAILABLE = True
except Exception:
    PROM_AVAILABLE = False
    logger.debug("prometheus_client not installed; sampler metrics will be in-memory only.")


class SamplingMetrics:
    def __init__(self):
        if PROM_AVAILABLE:
            self.sampled = Counter("tracing_sampler_sampled_total", "Traces chosen for sampling/export")
            self.rejected = Counter("tracing_sampler_rejected_total", "Traces rejected by sampler")
            self.forced = Counter("tracing_sampler_forced_total", "Traces force-sampled due to priority attributes")
            self.window_seen = Counter("tracing_sampler_window_seen_total", "Traces observed in current window (approx)")
            self.adjustments = Counter("tracing_sampler_adjustments_total", "Number of adaptive adjustments made")
            self.current_window_ts = time.time()
        else:
            self.sampled_v = 0
            self.rejected_v = 0
            self.forced_v = 0
            self.window_seen_v = 0
            self.adjustments_v = 0
            self.current_window_ts = time.time()

    # metric-friendly methods
    def inc_sampled(self, n: int = 1):
        if PROM_AVAILABLE:
            self.sampled.inc(n)
        else:
            self.sampled_v += n

    def inc_rejected(self, n: int = 1):
        if PROM_AVAILABLE:
            self.rejected.inc(n)
        else:
            self.rejected_v += n

    def inc_forced(self, n: int = 1):
        if PROM_AVAILABLE:
            self.forced.inc(n)
        else:
            self.forced_v += n

    def increment_seen(self) -> int:
        """
        Increment seen counter and return new seen value for the current window.
        For simple implemention, we don't expire seen counter here; higher-level samplers reset window state.
        """
        if PROM_AVAILABLE:
            self.window_seen.inc()
            # no direct way to return value from Prom Counter; so return 1 as minimal contract
            return 1
        else:
            self.window_seen_v += 1
            return self.window_seen_v

    def record_adjustment(self, old_prob: float, new_prob: float, observed_tps: float):
        if PROM_AVAILABLE:
            self.adjustments.inc()
        else:
            self.adjustments_v += 1
        logger.info("AdaptiveSampler adjustment: %s -> %s observed_tps=%s", old_prob, new_prob, observed_tps)

    def reset_window(self):
        # For in-memory counters we reset window_seen_v; for Prometheus Counters we can't decrement counters,
        # so rely on gauge/other observation for windowed observation if needed.
        if not PROM_AVAILABLE:
            self.window_seen_v = 0
