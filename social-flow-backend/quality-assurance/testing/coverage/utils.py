"""
Utility functions for coverage testing.
"""

import os
import shutil
import logging

logger = logging.getLogger("coverage-utils")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def ensure_clean_dir(path: str):
    """Ensure a clean directory exists (remove old, recreate new)."""
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)
    logger.info(f"Clean directory ready: {path}")


def percent(value: float, total: float) -> float:
    """Safe percentage calculation."""
    if total == 0:
        return 0.0
    return round((value / total) * 100, 2)
