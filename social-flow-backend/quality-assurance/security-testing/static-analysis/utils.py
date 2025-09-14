"""
Helpers used across the static-analysis package.
File scanning helpers, logging, JSON saving, severity mapping, etc.
"""

import os
import json
import logging
import fnmatch
from datetime import datetime
from pathlib import Path
from typing import List, Iterator

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(levelname)s] - %(name)s - %(message)s",
    handlers=[logging.FileHandler("static_analysis.log"), logging.StreamHandler()]
)
logger = logging.getLogger("static-analysis")

# Severity ordering for simple filtering/comparison
SEVERITY_ORDER = {
    "INFO": 0,
    "LOW": 1,
    "MEDIUM": 2,
    "HIGH": 3,
    "CRITICAL": 4
}

def save_json(path: str, obj: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved JSON report to {path}")

def timestamp() -> str:
    return datetime.utcnow().isoformat() + "Z"

def is_excluded(path: str, exclude_patterns: List[str]) -> bool:
    """Return True if path should be excluded based on exclude list (supports glob patterns)."""
    for pat in exclude_patterns:
        if pat in path:
            return True
    return False

def iter_source_files(root: str, patterns: List[str], exclude: List[str], max_file_size_kb: int = 512) -> Iterator[str]:
    """
    Walk root and yield paths matching patterns (e.g. ["*.py", "*.js"]).
    Skips files in excluded paths and files larger than max_file_size_kb.
    """
    logger.debug(f"Scanning files under {root} for {patterns}")
    for base, dirs, files in os.walk(root):
        # remove excluded dirs in-place for efficiency
        dirs[:] = [d for d in dirs if not is_excluded(os.path.join(base, d), exclude)]
        for f in files:
            full = os.path.join(base, f)
            if is_excluded(full, exclude):
                continue
            if any(fnmatch.fnmatch(f, p) for p in patterns):
                try:
                    size_kb = Path(full).stat().st_size / 1024
                    if size_kb > max_file_size_kb:
                        logger.info(f"Skipping {full} (size {size_kb:.1f}KB > {max_file_size_kb}KB)")
                        continue
                except Exception:
                    logger.exception(f"Failed to stat {full}; skipping")
                    continue
                yield full

def normalize_severity(level: str) -> str:
    """Normalize severity strings (ensure uppercase and valid)."""
    lvl = (level or "LOW").upper()
    return lvl if lvl in SEVERITY_ORDER else "LOW"
