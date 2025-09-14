# Helpers for batching, retries, normalization
# monitoring/logging/centralized/utils.py
"""
Utility helpers for centralized logging system.
Includes batching, retries, and normalization of logs.
"""

import time
import hashlib
import json


def batch_logs(logs, batch_size):
    """Yield logs in batches."""
    for i in range(0, len(logs), batch_size):
        yield logs[i:i + batch_size]


def normalize_log(log: dict) -> dict:
    """Normalize log fields to standard schema."""
    return {
        "timestamp": log.get("timestamp"),
        "level": log.get("level", "INFO").upper(),
        "service": log.get("service", "unknown"),
        "message": log.get("message", ""),
        "metadata": log.get("metadata", {})
    }


def retry(operation, retries=3, delay=1):
    """Retry an operation with exponential backoff."""
    for attempt in range(retries):
        try:
            return operation()
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(delay * (2 ** attempt))
            else:
                raise e


def stable_id(log: dict) -> str:
    """Generate stable ID for a log."""
    return hashlib.sha1(json.dumps(log, sort_keys=True).encode()).hexdigest()
