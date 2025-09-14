# scripts/maintenance/cleanup.py
import os
import logging
import glob
import time
from datetime import datetime, timedelta
from typing import Dict, Any

logger = logging.getLogger("maintenance.cleanup")

class Cleaner:
    """
    Cleanup stale files, old deployments, caches.

    Config:
      maintenance:
        cleanup:
          temp_paths: ['/tmp/socialflow', '/var/run/socialflow/tmp']
          older_than_days: 7
          patterns:
            - '*.tmp'
            - '*.cache'
          dry_run: true
    """

    def __init__(self, config: Dict[str, Any]):
        cfg = config.get("maintenance", {}).get("cleanup", {})
        self.paths = cfg.get("temp_paths", [])
        self.patterns = cfg.get("patterns", ["*"])
        self.older_than_days = int(cfg.get("older_than_days", 7))
        self.dry_run = bool(cfg.get("dry_run", True))

    def _remove_path(self, path: str):
        if self.dry_run:
            logger.info("[DRY RUN] Would remove: %s", path)
            return
        try:
            if os.path.isdir(path):
                import shutil
                shutil.rmtree(path)
                logger.info("Removed directory: %s", path)
            else:
                os.remove(path)
                logger.info("Removed file: %s", path)
        except Exception:
            logger.exception("Failed to remove: %s", path)

    def run(self):
        cutoff_ts = (datetime.utcnow() - timedelta(days=self.older_than_days)).timestamp()
        for base in self.paths:
            for pattern in self.patterns:
                glob_path = os.path.join(base, pattern)
                for f in glob.glob(glob_path):
                    try:
                        mtime = os.path.getmtime(f)
                        if mtime < cutoff_ts:
                            self._remove_path(f)
                    except Exception:
                        logger.exception("Error evaluating file: %s", f)
