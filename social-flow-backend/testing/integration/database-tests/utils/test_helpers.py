"""
Generic helpers for tests (random data, timing, retry).
"""

import random
import string
import time
from tenacity import retry, stop_after_attempt, wait_fixed

def random_string(n=8):
    return "".join(random.choices(string.ascii_lowercase + string.digits, k=n))

def timestamp_ms():
    return int(time.time() * 1000)

@retry(stop=stop_after_attempt(5), wait=wait_fixed(0.5))
def retry_assert(fn):
    """Run fn and retry if AssertionError is thrown (useful for eventually-consistent operations)."""
    return fn()
