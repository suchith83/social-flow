# Helper utilities for infra monitoring (parsing, retries, etc.)
"""
Utility helpers used by infra metrics collectors.
Try to keep third-party imports (like psutil) isolated here so unit tests can mock easily.
"""

import time
import logging

logger = logging.getLogger("infra_metrics")

try:
    import psutil
except Exception:  # psutil may not be installed in some environments used for testing
    psutil = None
    logger.warning("psutil not available — infra_collector will fail if used without psutil installed.")


def safe_psutil():
    if psutil is None:
        raise RuntimeError("psutil is required for infra metrics collection. Install via `pip install psutil`")
    return psutil


def now_ts() -> float:
    """Return unix timestamp in seconds (float)."""
    return time.time()
