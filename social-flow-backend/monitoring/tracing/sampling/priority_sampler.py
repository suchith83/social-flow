# priority_sampler.py
# Created by Create-SamplingFiles.ps1
"""
PrioritySampler: a lightweight wrapper that force-samples spans when attributes indicate high priority.

Common use-cases:
 - Always sample spans with `error` attribute or status >= 500
 - Sample when explicit `force_sample` attribute or debug flag is present
 - Otherwise delegate to an underlying sampler (probabilistic/adaptive/reservoir)

This class expects instrumentations to call `should_sample(trace_id, attributes)` and pass span attributes.
"""

import logging
from typing import Optional, Dict, Any, Iterable
from .config import SamplingConfig
from .utils import SamplingMetrics

logger = logging.getLogger("tracing.sampling.priority_sampler")


class PrioritySampler:
    def __init__(self, underlying_sampler=None, priority_attributes: Optional[Iterable[str]] = None, metrics: Optional[SamplingMetrics] = None):
        self.underlying = underlying_sampler
        self.priority_attrs = list(priority_attributes or SamplingConfig.PRIORITY_ATTRIBUTES)
        self.metrics = metrics or SamplingMetrics()

    def _has_priority(self, attributes: Optional[Dict[str, Any]]) -> bool:
        if not attributes:
            return False
        for k in self.priority_attrs:
            if k in attributes:
                v = attributes.get(k)
                # treat truthy values (True or non-empty / non-zero) as priority
                if v is True:
                    return True
                if isinstance(v, (int, float)) and v != 0:
                    return True
                if isinstance(v, str) and v.strip().lower() not in ("", "false", "0", "none", "null"):
                    return True
        # also check for http.status_code >= 500
        sc = attributes.get("http.status_code") if attributes else None
        try:
            if sc is not None and int(sc) >= 500:
                return True
        except Exception:
            pass
        return False

    def should_sample(self, trace_id: Optional[str] = None, attributes: Optional[Dict[str, Any]] = None) -> bool:
        if self._has_priority(attributes):
            self.metrics.inc_forced()
            return True
        if self.underlying:
            decision = self.underlying.should_sample(trace_id=trace_id, attributes=attributes)
            # record metrics
            if decision:
                self.metrics.inc_sampled()
            else:
                self.metrics.inc_rejected()
            return decision
        # fallback: no underlying sampler => use default probability
        import random
        prob = SamplingConfig.DEFAULT_PROBABILITY
        if random.random() < prob:
            self.metrics.inc_sampled()
            return True
        self.metrics.inc_rejected()
        return False
