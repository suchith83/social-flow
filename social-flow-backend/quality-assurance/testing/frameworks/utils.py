"""
Utility helpers for running tests and collecting outputs.
"""

import os
import json
import logging
from typing import Any, Dict

logger = logging.getLogger("qa-testing-frameworks")
if not logger.handlers:
    import sys
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(h)
    logger.setLevel(logging.INFO)


def write_json_atomic(path: str, obj: Any):
    """Write JSON to path atomically (write tmp then replace)."""
    tmp = path + ".tmp"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)
    logger.debug("Wrote JSON report to %s", path)


def find_tests(root: str):
    """Yield file paths that look like tests in a directory (pytest/unittest friendly)."""
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if fn.startswith("test_") and fn.endswith((".py",)):
                yield os.path.join(dirpath, fn)
            elif fn.endswith("_test.py"):
                yield os.path.join(dirpath, fn)
