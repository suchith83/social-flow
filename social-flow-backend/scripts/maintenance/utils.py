# scripts/maintenance/utils.py
import os
import logging
import subprocess
import shutil
import datetime
from typing import Optional

logger = logging.getLogger("maintenance.utils")


def ensure_dir(path: str, mode: int = 0o750):
    os.makedirs(path, exist_ok=True)
    try:
        os.chmod(path, mode)
    except Exception:
        # best effort; ignore if not permitted
        logger.debug("Could not chmod %s", path)


def now_ts() -> str:
    return datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")


def human_days_ago(days: int) -> float:
    """
    Return epoch timestamp for N days ago
    """
    import time
    return (datetime.datetime.utcnow() - datetime.timedelta(days=days)).timestamp()


def run_cmd(cmd: list, check: bool = True, capture_output: bool = True, cwd: Optional[str] = None):
    """
    Run a shell command with logging & error handling
    """
    logger.info("Running command: %s", " ".join(cmd))
    try:
        res = subprocess.run(cmd, check=check, capture_output=capture_output, text=True, cwd=cwd)
        if res.stdout:
            logger.debug("CMD stdout: %s", res.stdout.strip())
        if res.stderr:
            logger.debug("CMD stderr: %s", res.stderr.strip())
        return res
    except subprocess.CalledProcessError as e:
        logger.error("Command failed: %s", e)
        raise
