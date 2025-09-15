"""
Common utilities for pipelines: logging, retry, timing, safe-io.
"""

import logging
import time
from functools import wraps
from typing import Callable, Any
import os
import json

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger("predictive-pipelines")


def retry(exceptions=Exception, tries: int = 3, delay: float = 2.0, backoff: float = 2.0):
    """
    Retry decorator with exponential backoff.
    Usage:
        @retry(Exception, tries=3)
        def work(): ...
    """
    def deco(func: Callable[..., Any]):
        @wraps(func)
        def wrapper(*args, **kwargs):
            _tries, _delay = tries, delay
            while _tries > 1:
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    logger.warning(f"Error in {func.__name__}: {e}. Retrying in {_delay}s ({_tries-1} tries left)")
                    time.sleep(_delay)
                    _tries -= 1
                    _delay *= backoff
            # last attempt
            return func(*args, **kwargs)
        return wrapper
    return deco


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def write_json(path: str, obj):
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)


def read_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
