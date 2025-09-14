# Executes log cleanup jobs
# monitoring/logging/retention/cleaner.py
"""
Cleaner module for log retention.
Deletes expired logs, archives old logs based on policy.
"""

import os
import datetime
from pathlib import Path
from .policy import RetentionPolicy
from .archiver import Archiver
from .utils import get_file_age_days
from .config import CONFIG


class LogCleaner:
    def __init__(self, base_path: Path):
        self.base_path = Path(base_path)
        self.policy = RetentionPolicy()
        self.archiver = Archiver()

    def clean(self):
        """Run cleanup job: delete expired logs and archive old ones."""
        for file in self.base_path.glob("**/*.log"):
            age = get_file_age_days(file)
            tier = self.policy.classify(
                datetime.datetime.utcnow() - datetime.timedelta(days=age)
            )

            if self.policy.should_delete(
                datetime.datetime.utcnow() - datetime.timedelta(days=age)
            ):
                file.unlink()
            elif tier == "archive" and CONFIG["ARCHIVE"]["enabled"]:
                self.archiver.archive(file)
                file.unlink()
