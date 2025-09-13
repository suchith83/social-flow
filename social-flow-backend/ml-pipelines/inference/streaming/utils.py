# Config, retry, logger
# ================================================================
# File: utils.py
# Purpose: Config, retry, logging helpers
# ================================================================

import logging
import yaml
import time
import functools


def load_config(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def setup_logger(name, level=logging.INFO):
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("[%(asctime)s] %(levelname)s - %(name)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


def retry(max_attempts=3, delay=2, exceptions=(Exception,)):
    def decorator(fn):
        @functools.wraps(fn)
        async def wrapper(*args, **kwargs):
            attempt = 0
            while attempt < max_attempts:
                try:
                    return await fn(*args, **kwargs)
                except exceptions as e:
                    attempt += 1
                    if attempt >= max_attempts:
                        raise
                    time.sleep(delay * attempt)
        return wrapper
    return decorator
