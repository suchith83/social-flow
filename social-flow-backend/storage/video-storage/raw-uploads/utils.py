"""
Utility helpers for raw-uploads
"""

import os
import shutil
import logging
import hashlib
from pathlib import Path
from typing import Tuple

from .config import config

logger = logging.getLogger("raw-uploads")
logger.setLevel("INFO")


def ensure_dir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)


def file_hash(path: str, chunk_size: int = 8192) -> str:
    """SHA256 of a file streamed by chunks."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def atomic_move(src: str, dst: str):
    """Move file into place ensuring directories exist."""
    ensure_dir(os.path.dirname(dst))
    shutil.move(src, dst)


def safe_remove(path: str):
    try:
        os.remove(path)
    except FileNotFoundError:
        pass
