"""
Utility functions for live streaming
"""

import logging
import subprocess
from pathlib import Path
from .config import config

logger = logging.getLogger("live-streaming")
logger.setLevel("INFO")


def ensure_dir(path: str):
    """Ensure a directory exists"""
    Path(path).mkdir(parents=True, exist_ok=True)


def run_command(cmd: str):
    """Run system command (used for ffmpeg)"""
    logger.info(f"Executing: {cmd}")
    proc = subprocess.Popen(
        cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    out, err = proc.communicate()
    if proc.returncode != 0:
        logger.error(f"Command failed: {err.decode()}")
        raise RuntimeError(err.decode())
    return out.decode()
