# Shared helper functions
"""
Utility helpers for iOS package.

- simple file storage helpers
- token bucket rate limiter
- InMemoryStore for demo usage
- privacy preserving device id hashing
- small helper to generate APNs JWT (without external libs for demo)
"""

import os
import time
import threading
import hashlib
from typing import Optional, Dict, Any
from collections import defaultdict
import base64
import json

from .config import CONFIG

# ---- File helpers ----
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

# ---- Token-bucket RateLimiter ----
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

# ---- In-memory store (demo) ----
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
def device_hash(raw_device_id: str, salt: str = "ios-salt"):
    h = hashlib.sha256()
    h.update(salt.encode("utf-8"))
    h.update(raw_device_id.encode("utf-8"))
    return h.hexdigest()

# ---- Simple APNs JWT builder (demo; doesn't sign without PyJWT) ----
def build_apns_jwt(key_id: str, team_id: str, auth_key_p8_path: str) -> Optional[str]:
    """
    Build JWT for APNs (header.payload.signature).
    This demo function creates header+payload and returns base64 parts; in production use PyJWT to sign with ES256 and your .p8 key.
    """
    try:
        header = {"alg": "ES256", "kid": key_id}
        payload = {"iss": team_id, "iat": int(time.time())}
        header_b = base64.urlsafe_b64encode(json.dumps(header).encode()).rstrip(b"=").decode()
        payload_b = base64.urlsafe_b64encode(json.dumps(payload).encode()).rstrip(b"=").decode()
        # Signature requires ES256 signing of header_b + "." + payload_b with private key (.p8)
        # For demo, return unsigned token skeleton to show format; DO NOT use in production.
        return f"{header_b}.{payload_b}."
    except Exception:
        return None

# instantiate shared helpers
rate_limiter = RateLimiter(capacity=CONFIG.analytics_rate_limit_per_minute)
store = InMemoryStore()
