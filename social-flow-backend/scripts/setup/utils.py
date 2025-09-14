# scripts/setup/utils.py
import subprocess
import logging
import os
import shutil
import stat
from typing import Optional, Tuple

logger = logging.getLogger("setup.utils")

def run(cmd: list, check: bool = True, cwd: Optional[str] = None, env: Optional[dict] = None, capture_output: bool = False, text: bool = True, timeout: Optional[int] = None) -> Tuple[int, str, str]:
    """
    Run a system command safely. Return (rc, stdout, stderr).
    If check=True and rc != 0, raises subprocess.CalledProcessError.
    """
    logger.info("RUN: %s", " ".join(map(str, cmd)))
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE if capture_output else subprocess.DEVNULL,
                            stderr=subprocess.PIPE if capture_output else subprocess.DEVNULL,
                            cwd=cwd, env=env, text=text)
    try:
        out, err = proc.communicate(timeout=timeout)
    except Exception:
        proc.kill()
        out, err = proc.communicate()
        raise
    rc = proc.returncode
    if rc != 0 and check:
        logger.error("Command failed rc=%s stdout=%s stderr=%s", rc, out, err)
        raise subprocess.CalledProcessError(rc, cmd, output=out, stderr=err)
    return rc, (out or "").strip(), (err or "").strip()

def ensure_dir(path: str, mode: int = 0o755):
    os.makedirs(path, exist_ok=True)
    try:
        os.chmod(path, mode)
    except Exception:
        logger.debug("Could not chmod %s", path)

def write_file(path: str, data: str, mode: int = 0o640, as_root: bool = False):
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(data)
    try:
        os.chmod(path, mode)
    except Exception:
        logger.debug("Could not set mode on %s", path)

def which(binary: str) -> Optional[str]:
    return shutil.which(binary)

def is_root() -> bool:
    return os.geteuid() == 0

def add_executable_bit(path: str):
    st = os.stat(path)
    os.chmod(path, st.st_mode | stat.S_IEXEC)
