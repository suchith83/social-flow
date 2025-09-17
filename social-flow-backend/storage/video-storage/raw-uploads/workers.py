"""
Background worker utilities (simple examples)
- Retry runner for pending uploads
- Integration points for task queues like RQ/Celery in production
"""

import time
import logging
from typing import Callable

logger = logging.getLogger("raw-uploads.workers")


def retry_operation(op: Callable, retries: int = 3, delay_sec: int = 2, backoff: int = 2):
    """Simple retry wrapper."""
    attempt = 0
    while True:
        try:
            return op()
        except Exception as e:
            attempt += 1
            if attempt > retries:
                logger.error(f"Operation failed after {retries} tries: {e}")
                raise
            logger.warning(f"Operation failed: {e}. Retrying in {delay_sec}s...")
            time.sleep(delay_sec)
            delay_sec *= backoff
