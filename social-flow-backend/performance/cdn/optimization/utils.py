# Shared helper functions for optimization
# performance/cdn/optimization/utils.py
"""
Shared utilities for CDN optimization.
"""

import logging
import time
import hashlib
import functools
from typing import Callable, Any

# Logging config
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(name)s - %(message)s"
)
logger = logging.getLogger("cdn.optimization")

def timeit(func: Callable) -> Callable:
    """Decorator to measure execution time."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        res = func(*args, **kwargs)
        elapsed = (time.time() - start) * 1000
        logger.debug(f"{func.__name__} took {elapsed:.2f} ms")
        return res
    return wrapper

def content_hash(content: bytes) -> str:
    """Return SHA256 hash of content."""
    return hashlib.sha256(content).hexdigest()
