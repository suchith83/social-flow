# invalidation_strategies.py
# Created by Create-Invalidation.ps1
"""
invalidation_strategies.py
---------------------------
Pluggable invalidation strategy abstractions.

Provides:
- ImmediateInvalidation: actively purge caches on write/delete.
- LazyInvalidation: mark keys as stale; invalidation happens on next read or via sweep.
- TimeBasedInvalidation: TTL / scheduled expiry management (cooperates with other layers).
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Iterable, Optional
import time
import logging

logger = logging.getLogger(__name__)


class InvalidationStrategy(ABC):
    """Abstract base class for invalidation strategies."""

    @abstractmethod
    def invalidate(self, keys: Iterable[str]) -> None:
        """Invalidate the given keys synchronously (or trigger async workflow)."""
        raise NotImplementedError

    @abstractmethod
    def schedule_invalidate(self, keys: Iterable[str], when: Optional[float] = None) -> None:
        """
        Schedule invalidation in the future.

        when: epoch seconds when invalidation should occur. If None, treat as immediate.
        """
        raise NotImplementedError


class ImmediateInvalidation(InvalidationStrategy):
    """Implements a direct invalidation call - callers expect immediate purge or propagation."""

    def __init__(self, invalidator):
        """
        invalidator: object exposing `invalidate(keys)` method (e.g., RedisInvalidator).
        """
        self.invalidator = invalidator

    def invalidate(self, keys: Iterable[str]) -> None:
        logger.debug("ImmediateInvalidation.invalidate called for %d keys", len(list(keys)))
        self.invalidator.invalidate(keys)

    def schedule_invalidate(self, keys: Iterable[str], when: Optional[float] = None) -> None:
        # for immediate strategy, scheduling means call at the requested time
        if when and when > time.time():
            # naive sleep-based scheduler for demonstration; in prod, enqueue to a scheduler/worker
            delay = when - time.time()
            logger.debug("ImmediateInvalidation scheduling in %s seconds", delay)
            time.sleep(delay)
        self.invalidate(keys)


class LazyInvalidation(InvalidationStrategy):
    """
    Marks keys stale and relies on reads to detect staleness or on background sweeps.

    This strategy expects a 'mark_stale' method on the invalidator.
    """

    def __init__(self, invalidator, stale_ttl: int = 60):
        self.invalidator = invalidator
        self.stale_ttl = stale_ttl

    def invalidate(self, keys: Iterable[str]) -> None:
        # mark keys as stale with a short TTL; a background sweep or cache-on-read will evict them
        try:
            self.invalidator.mark_stale(keys, ttl=self.stale_ttl)
        except AttributeError:
            # fallback to direct invalidation
            self.invalidator.invalidate(keys)

    def schedule_invalidate(self, keys: Iterable[str], when: Optional[float] = None) -> None:
        # schedule by setting a delayed stale marker
        if when and when > time.time():
            delay = when - time.time()
            time.sleep(delay)
        self.invalidate(keys)


class TimeBasedInvalidation(InvalidationStrategy):
    """
    Use TTLs and versioning to invalidate at specific times. Works well for caches that
    support per-key TTLs or versioned namespaces.
    """

    def __init__(self, invalidator, ttl: int = 3600):
        self.invalidator = invalidator
        self.ttl = ttl

    def invalidate(self, keys: Iterable[str]) -> None:
        # set short TTL instead of immediate delete
        try:
            self.invalidator.set_ttl(keys, ttl=0)  # immediate expire; many clients map 0 to immediate
        except AttributeError:
            self.invalidator.invalidate(keys)

    def schedule_invalidate(self, keys: Iterable[str], when: Optional[float] = None) -> None:
        # schedule by setting TTL to the delta between when and now
        if not when:
            self.invalidate(keys)
            return
        delta = max(0, int(when - time.time()))
        self.invalidator.set_ttl(keys, ttl=delta)
