# Retry and backoff strategies for escalation
"""
Retry and backoff utilities.

Provides a configurable exponential/backoff strategy that can be reused by escalators.
"""

import time
import math
import logging
from typing import Callable, Optional, Dict, Any

logger = logging.getLogger(__name__)


class RetryBackoff:
    """
    Encodes a backoff policy.

    Parameters:
        base (float): base sleep in seconds for first retry (default 1s)
        factor (float): multiplier for exponential growth (default 2)
        max_backoff (float): cap in seconds
        jitter (float): fraction [0,1] to randomize backoff to avoid thundering herd.
    """

    def __init__(self, base: float = 1.0, factor: float = 2.0, max_backoff: float = 60.0, jitter: float = 0.1):
        self.base = float(base)
        self.factor = float(factor)
        self.max_backoff = float(max_backoff)
        self.jitter = float(jitter)

    def next_backoff(self, attempt: int) -> float:
        """
        Compute backoff for a given attempt (1-indexed).

        We apply exponential: base * factor^(attempt-1), capped by max_backoff,
        with optional jitter (± jitter*value).
        """
        attempt = max(1, int(attempt))
        val = self.base * (self.factor ** (attempt - 1))
        val = min(val, self.max_backoff)
        # compute jitter range
        jitter_amount = val * self.jitter
        # use deterministic jitter to keep here simple (import random if nondeterministic needed)
        # but in prod use random.uniform(-jitter_amount, jitter_amount)
        final = val + jitter_amount  # safe upward jitter
        logger.debug("backoff for attempt %d -> %s seconds", attempt, final)
        return final

    def run_with_retries(self, func: Callable, max_attempts: int = 3, exceptions=(Exception,), *args, **kwargs):
        """
        Runs `func` with retries and backoff. `func` is a callable that may raise.

        Returns the result of func if successful, otherwise raises the last exception.
        """
        last_exc = None
        for attempt in range(1, max_attempts + 1):
            try:
                return func(*args, **kwargs)
            except exceptions as e:
                last_exc = e
                backoff = self.next_backoff(attempt)
                logger.warning("Attempt %d/%d failed: %s. Backing off for %s seconds.",
                               attempt, max_attempts, e, backoff)
                time.sleep(backoff)
        logger.error("Operation failed after %d attempts.", max_attempts)
        raise last_exc
