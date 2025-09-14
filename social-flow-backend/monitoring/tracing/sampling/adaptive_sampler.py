# adaptive_sampler.py
# Created by Create-SamplingFiles.ps1
"""
AdaptiveSampler: adjusts sampling probability dynamically based on observed export throughput.

Goal:
 - Maintain roughly `target_tps` exported traces per second (across the process)
 - Increase sampling probability when exported TPS below target, decrease when above
 - Respect min/max probability bounds to avoid over/under sampling
 - Expose should_sample(trace_id, attributes) method
"""

import threading
import time
import random
from typing import Optional, Dict, Any
from .config import SamplingConfig
from .utils import SamplingMetrics

class AdaptiveSampler:
    def __init__(self,
                 target_tps: float = None,
                 min_prob: float = None,
                 max_prob: float = None,
                 adjust_step: float = None,
                 metrics: Optional[SamplingMetrics] = None):
        self.target_tps = SamplingConfig.ADAPTIVE_TARGET_TPS if target_tps is None else float(target_tps)
        self.min_prob = SamplingConfig.ADAPTIVE_MIN_PROB if min_prob is None else float(min_prob)
        self.max_prob = SamplingConfig.ADAPTIVE_MAX_PROB if max_prob is None else float(max_prob)
        self.adjust_step = SamplingConfig.ADAPTIVE_ADJUST_STEP if adjust_step is None else float(adjust_step)

        self._prob = max(self.min_prob, min(self.max_prob, SamplingConfig.DEFAULT_PROBABILITY))
        self._lock = threading.Lock()
        self._exported_in_window = 0
        self._window_start = time.time()
        self._window_seconds = 1.0  # measure exported TPS per second
        self.metrics = metrics or SamplingMetrics()

    def _maybe_adjust(self):
        now = time.time()
        with self._lock:
            elapsed = now - self._window_start
            if elapsed >= self._window_seconds:
                exported = self._exported_in_window
                actual_tps = exported / elapsed if elapsed > 0 else 0.0
                # adjust probability: simple proportional step
                if actual_tps > self.target_tps:
                    # reduce probability
                    old = self._prob
                    self._prob = max(self.min_prob, self._prob * (1.0 - self.adjust_step))
                    self.metrics.record_adjustment(old, self._prob, actual_tps)
                elif actual_tps < self.target_tps:
                    old = self._prob
                    self._prob = min(self.max_prob, self._prob * (1.0 + self.adjust_step))
                    self.metrics.record_adjustment(old, self._prob, actual_tps)
                # reset window counters
                self._exported_in_window = 0
                self._window_start = now

    def should_sample(self, trace_id: Optional[str] = None, attributes: Optional[Dict[str, Any]] = None) -> bool:
        # adjust if window passed
        self._maybe_adjust()

        # priority attributes may override; caller should integrate with PrioritySampler wrapper
        p = None
        with self._lock:
            p = self._prob

        decide = random.random() < p
        if decide:
            # count as exported for window
            with self._lock:
                self._exported_in_window += 1
            self.metrics.inc_sampled()
        else:
            self.metrics.inc_rejected()
        return decide

    def current_probability(self) -> float:
        with self._lock:
            return float(self._prob)
