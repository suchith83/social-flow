# reservoir_sampler.py
# Created by Create-SamplingFiles.ps1
"""
ReservoirSampler: keeps a bounded reservoir of traces to sample uniformly from an incoming stream.

This sampler is suited for:
 - bursty traffic where you want to keep at most N sampled traces per window
 - reducing cardinality and cost by bounding exported traces

Design notes:
 - This implementation is NOT an OpenTelemetry SDK sampler (it does not implement the OTel sampler
   interface). Instead, it exposes a `should_sample(trace_id=None, attributes=None)` method that
   can be used by instrumentation wrappers to decide on sampling. You can adapt it into a full
   OTel sampler if desired.
 - The reservoir is refilled/restarted every `window_seconds` seconds.
"""

import threading
import time
import random
from typing import Optional, Any, Dict, List
from .config import SamplingConfig
from .utils import SamplingMetrics

class ReservoirSampler:
    def __init__(self,
                 size: int = None,
                 window_seconds: int = None,
                 metrics: Optional[SamplingMetrics] = None):
        self.size = SamplingConfig.RESERVOIR_SIZE if size is None else int(size)
        self.window_seconds = SamplingConfig.RESERVOIR_WINDOW_SECONDS if window_seconds is None else int(window_seconds)
        self._lock = threading.Lock()
        # reservoir holds chosen trace_ids in the current window (set for efficient membership test)
        self._reservoir = set()  # type: set
        self._window_start = time.time()
        self.metrics = metrics or SamplingMetrics()

    def _reset_window_if_needed(self):
        now = time.time()
        if now - self._window_start >= self.window_seconds:
            with self._lock:
                # double-check under lock
                if now - self._window_start >= self.window_seconds:
                    self._reservoir.clear()
                    self._window_start = now
                    self.metrics.reset_window()
                    return True
        return False

    def should_sample(self, trace_id: Optional[str] = None, attributes: Optional[Dict[str, Any]] = None) -> bool:
        """
        Decide whether to sample a trace:
          - If reservoir not full, accept (and add)
          - If reservoir full, accept with probability = size / seen_this_window (simple reservoir sampling)
        Returns True if chosen to sample/export.
        """
        self._reset_window_if_needed()
        with self._lock:
            seen = self.metrics.increment_seen()
            # if we have trace_id and it's already chosen, continue sampling it (for consistent sampling across process)
            if trace_id is not None and trace_id in self._reservoir:
                self.metrics.inc_sampled()
                return True

            if len(self._reservoir) < self.size:
                # accept and add
                if trace_id is not None:
                    self._reservoir.add(trace_id)
                else:
                    # store a placeholder id (random) if none provided
                    self._reservoir.add(str(random.getrandbits(64)))
                self.metrics.inc_sampled()
                return True

            # if reservoir is full, do reservoir sampling: accept with probability size / seen
            accept_prob = float(self.size) / float(max(1, seen))
            if random.random() < accept_prob:
                # evict random existing and add this trace_id if provided
                try:
                    # convert to list for pop
                    ev = next(iter(self._reservoir))
                    self._reservoir.remove(ev)
                except Exception:
                    pass
                if trace_id:
                    self._reservoir.add(trace_id)
                else:
                    self._reservoir.add(str(random.getrandbits(64)))
                self.metrics.inc_sampled()
                return True
            # otherwise reject
            return False

    def current_size(self) -> int:
        with self._lock:
            return len(self._reservoir)
