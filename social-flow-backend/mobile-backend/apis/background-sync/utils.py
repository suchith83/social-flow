# Helpers: diffing, batching, retry logic
"""
Helper utilities for Background Sync.
Provides retry decorators, logging, and data diffing.
"""

import logging
import functools
import time
from .config import get_config

config = get_config()

logger = logging.getLogger("background-sync")
logger.setLevel(config.LOG_LEVEL)


def retry_with_backoff(max_retries=5, backoff=2):
    """
    Decorator to retry functions with exponential backoff.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            retries = 0
            while retries < max_retries:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    wait = backoff * (2 ** retries)
                    logger.warning(f"Retry {retries+1}/{max_retries} after error: {e}. Waiting {wait}s...")
                    time.sleep(wait)
                    retries += 1
            raise Exception(f"Max retries exceeded for {func.__name__}")
        return wrapper
    return decorator
