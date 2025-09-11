# rate_limit.py
import time
from collections import defaultdict


class RateLimiter:
    """
    Token bucket rate limiter.
    """

    def __init__(self, rate: int, per: int):
        self.rate = rate  # tokens
        self.per = per    # per seconds
        self.allowances = defaultdict(lambda: self.rate)
        self.last_check = defaultdict(time.time)

    def is_allowed(self, key: str) -> bool:
        now = time.time()
        time_passed = now - self.last_check[key]
        self.last_check[key] = now

        self.allowances[key] += time_passed * (self.rate / self.per)
        if self.allowances[key] > self.rate:
            self.allowances[key] = self.rate

        if self.allowances[key] < 1:
            return False
        else:
            self.allowances[key] -= 1
            return True
