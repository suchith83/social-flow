# Configs, logging, decorators, helpers
# utils.py
import logging
import time
import functools
import yaml
from pathlib import Path

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("etl")

def timed(func):
    """Decorator for timing function execution."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        logger.info(f"{func.__name__} executed in {time.time() - start:.3f}s")
        return result
    return wrapper

def load_config(path: str = "config.yaml") -> dict:
    """Load ETL config from YAML file."""
    with open(Path(path), "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
