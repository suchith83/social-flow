# Defines retention rules
# monitoring/logging/retention/policy.py
"""
Retention policy definitions and evaluation.
Determines log tier (hot, warm, cold, archive) based on age.
"""

import datetime
from .config import CONFIG


class RetentionPolicy:
    def __init__(self):
        self.tiers = CONFIG["TIERED_STORAGE"]

    def classify(self, log_time: datetime.datetime) -> str:
        """Classify a log into a tier based on its age."""
        now = datetime.datetime.utcnow()
        age_days = (now - log_time).days

        if age_days <= self.tiers["hot"]:
            return "hot"
        elif age_days <= self.tiers["warm"]:
            return "warm"
        elif age_days <= self.tiers["cold"]:
            return "cold"
        return "archive"

    def should_delete(self, log_time: datetime.datetime) -> bool:
        """Determine if a log should be deleted based on retention."""
        now = datetime.datetime.utcnow()
        age_days = (now - log_time).days
        return age_days > CONFIG["TIERED_STORAGE"]["archive"]
