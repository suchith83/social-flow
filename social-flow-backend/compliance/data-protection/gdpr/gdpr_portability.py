"""
# Data portability request support
"""
"""
GDPR Data Portability Service
-----------------------------
Handles export of personal data in portable formats.
"""

import json
from .gdpr_exceptions import GDPRPortabilityError

class GDPRPortabilityService:
    """Exports user data in machine-readable format."""

    def __init__(self, database: dict):
        self.database = database

    def export_data(self, user_id: str, file_path: str) -> str:
        try:
            if user_id not in self.database:
                raise GDPRPortabilityError(f"User {user_id} not found")
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(self.database[user_id], f, indent=4)
            return file_path
        except Exception as e:
            raise GDPRPortabilityError(f"Failed to export data: {e}")
