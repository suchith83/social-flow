"""
Utility helpers
"""

import logging
import subprocess
from pathlib import Path
from .config import config

logger = logging.getLogger("processed-videos")
logger.setLevel("INFO")


def ensure_dir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)


def run_command(cmd: str):
    logger.info(f"Running command: {cmd}")
    proc = subprocess.Popen(
        cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    out, err = proc.communicate()
    if proc.returncode != 0:
        logger.error(f"Command failed: {err.decode()}")
        raise RuntimeError(err.decode())
    return out.decode()
