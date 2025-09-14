# Async + retry helpers
# performance/scaling/sharding/utils.py

import asyncio
import logging
import random
import functools
import time


logger = logging.getLogger("sharding.utils")


def retry(retries: int = 3, delay: float = 1.0, backoff: float = 2.0):
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
    await asyncio.sleep(base + random.uniform(-jitter, jitter))


def timed(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        res = func(*args, **kwargs)
        duration = time.time() - start
        logger.info(f"{func.__name__} executed in {duration:.3f}s")
        return res
    return wrapper
