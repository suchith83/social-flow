"""
Utility helpers for the thumbnails module
"""

import os
import logging
import subprocess
import json
from pathlib import Path
from typing import Tuple, List

from .config import config

logger = logging.getLogger("thumbnails")
logger.setLevel(logging.INFO)


def ensure_dir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)


def run_command(cmd: List[str], check: bool = True) -> Tuple[int, str, str]:
    """
    Run an external command (list form recommended).
    Returns (returncode, stdout, stderr).
    """
    logger.debug("Running command: %s", " ".join(cmd))
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    out, err = proc.communicate()
    if check and proc.returncode != 0:
        logger.error("Command failed (%s): %s", proc.returncode, err.strip())
        raise RuntimeError(f"Command {' '.join(cmd)} failed: {err.strip()}")
    return proc.returncode, out, err


def ffprobe_metadata(path: str) -> dict:
    """Return parsed ffprobe JSON metadata for the video file."""
    cmd = [config.FFPROBE_PATH, "-v", "quiet", "-print_format", "json", "-show_format", "-show_streams", path]
    _, out, _ = run_command(cmd)
    try:
        return json.loads(out)
    except Exception:
        return {}
