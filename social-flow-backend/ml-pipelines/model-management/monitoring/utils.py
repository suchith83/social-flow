# Shared helpers and utilities
"""
utils.py
Common utilities used by the monitoring package.
 - setup_logger: standardized logger creation
 - write_file: file writer with dirs
 - read_yaml / write_yaml helpers
 - simple exponential backoff retry decorator
"""

import logging
import os
import yaml
import time
import functools
from pathlib import Path
from typing import Any, Callable

DEFAULT_LOG_LEVEL = logging.INFO


def setup_logger(name: str, level: int = DEFAULT_LOG_LEVEL):
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    handler = logging.StreamHandler()
    fmt = "[%(asctime)s] %(levelname)s - %(name)s - %(message)s"
    handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(handler)
    logger.setLevel(level)
    return logger


def write_file(path: str, content: str, make_dirs: bool = True):
    p = Path(path)
    if make_dirs:
        p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content)
    return path


def read_yaml(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def write_yaml(path: str, obj: Any):
    write_file(path, yaml.safe_dump(obj))


def retry(max_attempts: int = 3, initial_delay: float = 1.0, exceptions: tuple = (Exception,)):
    def decorator(fn: Callable):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            attempts = 0
            delay = initial_delay
            while True:
                try:
                    return fn(*args, **kwargs)
                except exceptions as e:
                    attempts += 1
                    if attempts >= max_attempts:
                        raise
                    time.sleep(delay)
                    delay *= 2
        return wrapper
    return decorator
