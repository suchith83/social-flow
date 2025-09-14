# scripts/maintenance/log_rotator.py
import os
import logging
import gzip
import shutil
from datetime import datetime, timedelta
from typing import Dict, Any

from .utils import ensure_dir

logger = logging.getLogger("maintenance.log_rotator")

class LogRotator:
    """
    Rotate and archive logs.

    Features:
    - Rotate logs larger than rotate_after_mb
    - Compress rotated logs
    - Prune logs older than retention_days
    """

    def __init__(self, config: Dict[str, Any]):
        cfg = config.get("maintenance", {}).get("logs", {})
        self.rotate_after_mb = int(cfg.get("rotate_after_mb", 50))
        self.retention_days = int(cfg.get("retention_days", 14))
        self.paths = cfg.get("paths", ["/var/log/socialflow/*.log"])
        self.archive_dir = cfg.get("archive_dir", "/var/log/socialflow/archive")
        ensure_dir(self.archive_dir)

    def _rotate_file(self, filepath: str):
        if not os.path.exists(filepath):
            logger.debug("Log not present: %s", filepath)
            return
        size_mb = os.path.getsize(filepath) / (1024.0 * 1024.0)
        if size_mb < self.rotate_after_mb:
            logger.debug("Skipping rotate (size %.2f MB < %.2f MB): %s", size_mb, self.rotate_after_mb, filepath)
            return

        base = os.path.basename(filepath)
        ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        dest = os.path.join(self.archive_dir, f"{base}.{ts}.gz")
        logger.info("Rotating %s -> %s", filepath, dest)
        with open(filepath, "rb") as f_in, gzip.open(dest, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
        # truncate original
        open(filepath, "w").close()
        logger.info("Truncated original log %s after rotation", filepath)

    def rotate_all(self):
        import glob
        for pattern in self.paths:
            for path in glob.glob(pattern):
                try:
                    self._rotate_file(path)
                except Exception:
                    logger.exception("Failed to rotate %s", path)

    def prune(self):
        cutoff = datetime.utcnow() - timedelta(days=self.retention_days)
        for fname in os.listdir(self.archive_dir):
            fpath = os.path.join(self.archive_dir, fname)
            try:
                mtime = datetime.utcfromtimestamp(os.path.getmtime(fpath))
                if mtime < cutoff:
                    logger.info("Deleting archived log: %s", fpath)
                    os.remove(fpath)
            except Exception:
                logger.exception("Failed to delete archived log: %s", fpath)
