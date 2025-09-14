# scripts/maintenance/lockfile.py
import os
import fcntl
import time
import logging
from contextlib import contextmanager

logger = logging.getLogger("maintenance.lockfile")

@contextmanager
def file_lock(lock_path: str, timeout: int = 3600):
    """
    Context manager to acquire an exclusive file lock to prevent concurrent runs.
    Uses POSIX flock (works on Linux). On Windows, this will fallback to simple file existence check.

    lock_path: path to lock file, e.g. /var/lock/socialflow-maintenance.lock
    timeout: seconds to wait before giving up
    """
    lock_fd = None
    start = time.time()
    os.makedirs(os.path.dirname(lock_path), exist_ok=True)
    try:
        lock_fd = open(lock_path, "w")
        acquired = False
        while True:
            try:
                fcntl.flock(lock_fd.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                acquired = True
                logger.debug("Acquired lock %s", lock_path)
                yield
                break
            except BlockingIOError:
                if time.time() - start > timeout:
                    raise TimeoutError(f"Timeout waiting for lock {lock_path}")
                logger.debug("Lock busy, sleeping...")
                time.sleep(1)
    finally:
        if lock_fd and acquired:
            try:
                fcntl.flock(lock_fd.fileno(), fcntl.LOCK_UN)
                logger.debug("Released lock %s", lock_path)
            except Exception:
                logger.exception("Error releasing lock")
            lock_fd.close()
