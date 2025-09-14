# scripts/security/utils.py
import subprocess
import logging
import shutil
import json
import os
from typing import Tuple, Optional

logger = logging.getLogger("security.utils")


def which(cmd: str) -> Optional[str]:
    """
    Return path to executable if available, else None.
    """
    return shutil.which(cmd)


def run_command(cmd: list, cwd: Optional[str] = None, timeout: int = 300) -> Tuple[int, str, str]:
    """
    Run a command, capture stdout/stderr and return (rc, stdout, stderr).
    Raises subprocess.CalledProcessError for non-zero return codes only if check=True pattern desired by caller.
    """
    logger.debug("Running command: %s (cwd=%s)", " ".join(cmd), cwd)
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=cwd, text=True)
    try:
        out, err = proc.communicate(timeout=timeout)
    except Exception:
        proc.kill()
        out, err = proc.communicate()
        raise
    return proc.returncode, (out or "").strip(), (err or "").strip()


def safe_mkdir(path: str):
    os.makedirs(path, exist_ok=True)


def write_json(path: str, data):
    safe_mkdir(os.path.dirname(path) or ".")
    with open(path, "w") as fh:
        json.dump(data, fh, indent=2)
