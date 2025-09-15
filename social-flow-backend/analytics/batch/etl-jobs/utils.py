import logging
import time
from functools import wraps
import hashlib
import json


# Logger setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger("etl-jobs")


def retry(exceptions, tries=3, delay=2, backoff=2):
    """Retry decorator with exponential backoff"""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            _tries, _delay = tries, delay
            while _tries > 0:
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    logger.warning(
                        f"Retrying {func.__name__} after error: {e}. {_tries-1} attempts left."
                    )
                    time.sleep(_delay)
                    _tries -= 1
                    _delay *= backoff
            raise RuntimeError(f"Function {func.__name__} failed after {tries} retries.")

        return wrapper

    return decorator


def hash_record(record: dict) -> str:
    """Generate a hash for deduplication purposes"""
    return hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest()
