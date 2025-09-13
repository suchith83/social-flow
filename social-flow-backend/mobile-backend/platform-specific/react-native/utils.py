# Shared helper functions
"""
Utility helpers for React Native platform module.

Includes:
 - File storage helpers
 - Simple token bucket rate limiter
 - InMemoryStore (demo)
 - Device privacy hashing
 - Small helpers to compute checksums and safe file names
"""

import os
import time
import threading
import hashlib
from typing import Optional, Dict, Any
from collections import defaultdict
import uuid

from .config import CONFIG

# ---- file helpers ----
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

def safe_key(s: str) -> str:
    """Produce filesystem-safe key"""
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in s)

def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

# ---- token-bucket rate limiter ----
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

# ---- device hash ----
def device_hash(raw_device_id: str, salt: str = "react-native-salt") -> str:
    h = hashlib.sha256()
    h.update(salt.encode("utf-8"))
    h.update(raw_device_id.encode("utf-8"))
    return h.hexdigest()

# shared instances
rate_limiter = RateLimiter(capacity=CONFIG.analytics_rate_limit_per_minute)
store = InMemoryStore()
