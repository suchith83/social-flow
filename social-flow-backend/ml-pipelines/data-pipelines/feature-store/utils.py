# Logging, config, helpers
# utils.py
import logging
import yaml
import time
import functools
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("feature_store")

def timed(func):
    """Decorator for timing function execution."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        res = func(*args, **kwargs)
        logger.info(f"{func.__name__} executed in {time.time() - start:.3f}s")
        return res
    return wrapper

def load_config(path: str = "config.yaml") -> dict:
    with open(Path(path), "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
