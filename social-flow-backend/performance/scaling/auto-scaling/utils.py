# Utility functions (async tools, logging wrappers, retries)
# performance/scaling/auto_scaling/utils.py

import asyncio
import logging
import functools
import random
import time


logger = logging.getLogger("auto_scaling.utils")


def retry(retries: int = 3, delay: float = 1.0, backoff: float = 2.0):
    """
    Retry decorator with exponential backoff.
    """
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            attempt = 0
            while attempt < retries:
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    attempt += 1
                    wait = delay * (backoff ** (attempt - 1))
                    logger.warning(f"Retry {attempt}/{retries} after error: {e}. Waiting {wait:.2f}s")
                    await asyncio.sleep(wait)
            raise
        return wrapper
    return decorator


async def async_sleep_jitter(base: float = 1.0, jitter: float = 0.3):
    """
    Sleep with jitter to avoid thundering herd problem.
    """
    await asyncio.sleep(base + random.uniform(-jitter, jitter))


def timed(func):
    """
    Decorator to log execution time.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        res = func(*args, **kwargs)
        duration = time.time() - start
        logger.info(f"{func.__name__} executed in {duration:.3f}s")
        return res
    return wrapper
