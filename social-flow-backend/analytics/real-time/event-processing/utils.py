import logging
import time
from functools import wraps


def get_logger(name: str) -> logging.Logger:
    """Standard logger setup with formatting."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("[%(asctime)s] %(levelname)s %(name)s: %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)
    return logger


def retry(retries: int = 3, delay: int = 2):
    """Retry decorator with exponential backoff."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for i in range(retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if i < retries - 1:
                        time.sleep(delay * (2 ** i))
                    else:
                        raise e
        return wrapper
    return decorator
