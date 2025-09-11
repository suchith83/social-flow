# common/libraries/python/auth/rate_limiter.py
"""
Rate limiter to prevent brute-force login attempts.
Simple in-memory implementation (can extend to Redis).
"""

import time
from collections import defaultdict
from .config import AuthConfig

class RateLimiter:
    def __init__(self):
        self.attempts = defaultdict(list)

    def is_allowed(self, key: str) -> bool:
        """Check if action is allowed for this key (username/IP)."""
        now = int(time.time())
        window = AuthConfig.RATE_LIMIT_WINDOW
        limit = AuthConfig.RATE_LIMIT_ATTEMPTS

        # Clean old attempts
        self.attempts[key] = [t for t in self.attempts[key] if now - t < window]

        if len(self.attempts[key]) >= limit:
            return False

        self.attempts[key].append(now)
        return True
