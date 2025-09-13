# Shared helpers (storage, rate-limiting, validation)
"""
Shared utilities for the Flutter platform-specific backend.

- Simple file storage helpers
- Rate limiter
- InMemoryStore for demo
- helpers for identifying flutter runtime / engine
"""

import os
import time
import hashlib
import threading
from typing import Optional, Dict, Any
from collections import defaultdict

from .config import CONFIG

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def store_file(path: str, data: bytes, mode: str = "wb"):
    ensure_dir(os.path.dirname(path))
    with open(path, mode) as f:
        f.write(data)

def read_file(path: str) -> Optional[bytes]:
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return f.read()

def file_size(path: str) -> int:
    try:
        return os.path.getsize(path)
    except OSError:
        return 0

# ---- Rate limiter (token bucket) ----
class RateLimiter:
    def __init__(self, capacity: int, window_seconds: int = 60):
        self.capacity = capacity
        self.window = window_seconds
        self.lock = threading.Lock()
        self.tokens = defaultdict(lambda: capacity)
        self.updated = defaultdict(lambda: time.time())

    def allow(self, key: str, tokens: int = 1) -> bool:
        now = time.time()
        with self.lock:
            last = self.updated[key]
            elapsed = now - last
            refill = (elapsed / self.window) * self.capacity
            self.tokens[key] = min(self.capacity, self.tokens[key] + refill)
            self.updated[key] = now
            if self.tokens[key] >= tokens:
                self.tokens[key] -= tokens
                return True
            return False

# ---- simple in-memory store for demo ----
class InMemoryStore:
    def __init__(self):
        self.lock = threading.RLock()
        self._data = {}

    def get(self, key: str):
        with self.lock:
            return self._data.get(key)

    def set(self, key: str, value):
        with self.lock:
            self._data[key] = value

    def delete(self, key: str):
        with self.lock:
            if key in self._data:
                del self._data[key]

    def scan(self, prefix: str = ""):
        with self.lock:
            return {k: v for k, v in self._data.items() if k.startswith(prefix)}

# ---- Device id hashing for privacy ----
def device_hash(raw_device_id: str, salt: str = "change-me"):
    h = hashlib.sha256()
    h.update(salt.encode("utf-8"))
    h.update(raw_device_id.encode("utf-8"))
    return h.hexdigest()

# instantiate shared helpers
rate_limiter = RateLimiter(capacity=CONFIG.analytics_rate_limit_per_minute)
store = InMemoryStore()
