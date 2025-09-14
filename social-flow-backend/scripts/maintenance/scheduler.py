# scripts/maintenance/scheduler.py
import logging
import subprocess
from typing import Callable, Dict, Any

logger = logging.getLogger("maintenance.scheduler")

class Scheduler:
    """
    Lightweight scheduler wrapper to help run maintenance tasks either
    interactively or from a cron/systemd timer.

    Recommended usage:
      - Call maintenance_runner.run_all() from cron or systemd timer (e.g. daily)
      - Use dry-run for testing
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.cron_entries = config.get("maintenance", {}).get("cron", [])

    def show_cron_help(self):
        logger.info("Suggested cron entries:")
        for e in self.cron_entries:
            logger.info("  %s", e)

    def run_external(self, command: str):
        """
        Run an arbitrary external command controlled by config. Use sparingly.
        """
        logger.info("Running scheduled external command: %s", command)
        subprocess.run(command, shell=True, check=True)
