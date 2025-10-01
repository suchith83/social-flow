"""
# Data retention, safeguarding, and cross-border transfer rules
"""
# compliance/data-protection/pipeda/pipeda_data_handling.py
"""
PIPEDA Data Handling
--------------------
Implements:
- Retention limits
- Safeguards
- Cross-border data flow compliance
"""

import datetime
from typing import Dict, Any


class PIPEDADataHandler:
    def __init__(self):
        self.records: Dict[str, Dict[str, Any]] = {}

    def store_data(self, user_id: str, data: Dict[str, Any], retention_days: int):
        """Store data with retention period."""
        self.records[user_id] = {
            "data": data,
            "stored_at": datetime.datetime.utcnow(),
            "retention_days": retention_days,
        }

    def purge_expired(self) -> int:
        """Purge data beyond retention limit."""
        now = datetime.datetime.utcnow()
        expired = [uid for uid, rec in self.records.items()
                   if (now - rec["stored_at"]).days > rec["retention_days"]]
        for uid in expired:
            del self.records[uid]
        return len(expired)

    def safeguard(self, user_id: str) -> bool:
        """Simulate applying safeguards (encryption, access control)."""
        return user_id in self.records

    def allow_crossborder_transfer(self, user_id: str, destination: str) -> bool:
        """Check if cross-border transfer meets compliance conditions."""
        if user_id not in self.records:
            return False
        # In real scenario, would check adequacy of foreign jurisdiction
        return destination.lower() in ["canada", "adequate_country"]
