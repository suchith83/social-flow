# Helper utilities
# utils.py
import logging
import time
import functools
import json
from typing import Any, Dict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("data_validation")

def timed(func):
    """Decorator to log execution time of functions."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        duration = time.time() - start
        logger.info(f"{func.__name__} executed in {duration:.4f}s")
        return result
    return wrapper

def save_json(path: str, data: Dict[str, Any]):
    """Save dictionary as JSON."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    logger.info(f"Saved JSON report to {path}")

def load_json(path: str) -> Dict[str, Any]:
    """Load dictionary from JSON."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
