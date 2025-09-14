"""
Utility helpers for test data package.
"""

import os
import json
import hashlib
import logging
from typing import Any, Dict

logger = logging.getLogger("qa-testing-data")
if not logger.handlers:
    import sys
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(h)
    logger.setLevel(logging.INFO)


def ensure_dir(path: str):
    """Create directory path if it doesn't exist (idempotent)."""
    os.makedirs(path, exist_ok=True)
    logger.debug("Ensured directory exists: %s", path)


def fingerprint(obj: Any) -> str:
    """
    Return a stable fingerprint (sha1 hex) of an object serialized to JSON.
    Useful to determine if a fixture changed.
    """
    data = json.dumps(obj, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha1(data).hexdigest()


def atomic_write(path: str, data: str, mode: str = "w", encoding: str = "utf-8"):
    """Write to a temp file then atomically replace the target file."""
    tmp = path + ".tmp"
    with open(tmp, mode=mode, encoding=encoding) as f:
        f.write(data)
    os.replace(tmp, path)
    logger.debug("Atomically wrote file: %s", path)
