# =========================
# File: testing/security/penetration/utils/network.py
# =========================
"""
Network utilities: simple rate limiter, concurrency helpers.
"""

import time
import threading
from collections import deque

class RateLimiter:
    """
    Token-bucket like rate limiter to cap operations per time window.
    Not cryptographically precise but good for polite scanning.
    """
    def __init__(self, max_calls: int, period_seconds: float = 60.0):
        self.max_calls = max_calls
        self.period = period_seconds
        self.lock = threading.Lock()
        self.calls = deque()

    def acquire(self):
        with self.lock:
            now = time.monotonic()
            # purge old
            while self.calls and now - self.calls[0] > self.period:
                self.calls.popleft()
            if len(self.calls) < self.max_calls:
                self.calls.append(now)
                return True
            return False

    def wait(self):
        """
        Wait until we can acquire a slot. Sleeps in small increments to be responsive.
        """
        while not self.acquire():
            time.sleep(0.5)
