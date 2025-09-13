# Shared utilities
# ================================================================
# File: utils.py
# Purpose: Shared utilities for config, logging, retries
# ================================================================

import logging
import yaml
import functools
import time


def setup_logger(name, level=logging.INFO):
    logger = logging.getLogger(name)
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(asctime)s] %(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(level)
    return logger


def load_config(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def retry(max_attempts=3, delay=2, exceptions=(Exception,)):
    """
    Retry decorator with exponential backoff
    """
    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            attempt = 0
            while attempt < max_attempts:
                try:
                    return fn(*args, **kwargs)
                except exceptions as e:
                    attempt += 1
                    if attempt >= max_attempts:
                        raise
                    time.sleep(delay * attempt)
        return wrapper
    return decorator
