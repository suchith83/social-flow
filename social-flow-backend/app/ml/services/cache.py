"""Lightweight caching adapter used by MLService.

Designed to be Redis-friendly: interface mirrors common get/set semantics
with optional TTL. Falls back to in-memory dict; production deployment can
swap this with a Redis implementation (see AI_ML_ARCHITECTURE.md Caching section).
"""
from __future__ import annotations
import time
import threading
from dataclasses import dataclass
from typing import Any, Optional, Dict


@dataclass
class CacheItem:
    value: Any
    expires_at: Optional[float]  # epoch seconds
    created_at: float
    hits: int = 0

    def is_expired(self) -> bool:
        return self.expires_at is not None and time.time() >= self.expires_at


class SimpleCache:
    """Thread-safe in-memory cache with TTL support.

    NOTE: Deterministic & lightweight for tests. Not intended for large scale.
    """
    def __init__(self):
        self._store: Dict[str, CacheItem] = {}
        self._lock = threading.Lock()

    def get(self, key: str) -> Any:
        with self._lock:
            item = self._store.get(key)
            if not item:
                return None
            if item.is_expired():
                # Lazy eviction
                del self._store[key]
                return None
            item.hits += 1
            return item.value

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        expires_at = time.time() + ttl if ttl else None
        with self._lock:
            self._store[key] = CacheItem(value=value, expires_at=expires_at, created_at=time.time())

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            live = {k: v for k, v in self._store.items() if not v.is_expired()}
            expired = len(self._store) - len(live)
            return {
                "items": len(live),
                "expired": expired,
                "keys": list(live.keys()),
            }

    def clear(self) -> None:
        with self._lock:
            self._store.clear()
