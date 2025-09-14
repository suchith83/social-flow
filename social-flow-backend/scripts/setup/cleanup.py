# scripts/setup/cleanup.py
import logging
import os
import shutil
from typing import Dict, Any

logger = logging.getLogger("setup.cleanup")

class SetupCleanup:
    """
    Remove temporary files created during setup.
    Designed to be conservative (only removes known paths).
    """

    def __init__(self, config: Dict[str, Any]):
        self.cfg = config.get("setup", {})
        self.tmp_paths = self.cfg.get("tmp_paths", ["/tmp/socialflow-setup", "/var/tmp/socialflow-setup"])

    def run(self):
        for p in self.tmp_paths:
            try:
                if os.path.exists(p):
                    if os.path.isdir(p):
                        shutil.rmtree(p)
                        logger.info("Removed directory %s", p)
                    else:
                        os.remove(p)
                        logger.info("Removed file %s", p)
            except Exception:
                logger.exception("Failed cleanup for %s", p)
