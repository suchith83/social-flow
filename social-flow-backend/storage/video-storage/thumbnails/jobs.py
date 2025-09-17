"""
Background job orchestration utilities.

This file provides:
- Lightweight in-process worker for demo/testing
- Job wrapper that can be replaced by Celery/RQ implementations in production
- Hooks for sending completion callbacks
"""

import threading
import queue
import time
import requests
import logging
from typing import Callable, Dict, Any
from .config import config
from .utils import logger

_job_queue = queue.Queue()


def enqueue(fn: Callable, *args, **kwargs):
    """Enqueue a job for background processing."""
    _job_queue.put((fn, args, kwargs))


def worker_loop(poll_seconds: float = 0.5):
    logger.info("Starting thumbnails worker loop")
    while True:
        try:
            fn, args, kwargs = _job_queue.get(timeout=poll_seconds)
            try:
                fn(*args, **kwargs)
            except Exception as e:
                logger.exception("Job failed: %s", e)
            finally:
                _job_queue.task_done()
        except queue.Empty:
            continue


def start_worker_in_thread():
    t = threading.Thread(target=worker_loop, daemon=True)
    t.start()
    return t


def send_callback(payload: Dict[str, Any]):
    if not config.CALLBACK_URL:
        return
    try:
        requests.post(config.CALLBACK_URL, json=payload, timeout=5)
    except Exception as e:
        logger.warning("Callback failed: %s", e)
