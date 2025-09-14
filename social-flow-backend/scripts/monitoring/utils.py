# scripts/monitoring/utils.py
import logging
import time
import socket
import subprocess
from typing import Callable, Any
from functools import wraps

logger = logging.getLogger("monitoring.utils")


def retry(exception_types=(Exception,), max_attempts=3, backoff_factor=0.5):
    """
    Decorator to retry a function on certain exceptions with exponential backoff.
    """
    def decorator(fn: Callable[..., Any]):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            attempt = 0
            while True:
                try:
                    return fn(*args, **kwargs)
                except exception_types as e:
                    attempt += 1
                    if attempt >= max_attempts:
                        logger.exception("Function %s failed after %d attempts", fn.__name__, attempt)
                        raise
                    sleep = backoff_factor * (2 ** (attempt - 1))
                    logger.warning("Transient error calling %s: %s (attempt %d), sleeping %.2fs", fn.__name__, e, attempt, sleep)
                    time.sleep(sleep)
        return wrapper
    return decorator


def run_subprocess(cmd, timeout=None):
    """
    Run a subprocess safely and return (stdout, stderr, returncode).
    """
    logger.debug("Running subprocess: %s", cmd)
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=False, text=True)
    try:
        out, err = proc.communicate(timeout=timeout)
    except Exception:
        proc.kill()
        out, err = proc.communicate()
        raise
    return out.strip(), err.strip(), proc.returncode


def is_port_open(host: str, port: int, timeout: float = 1.0) -> bool:
    """
    Quick TCP connect check.
    """
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except Exception:
        return False
