"""
# Data erasure (Right to be Forgotten) handlers
"""
"""
GDPR Data Deletion Service
--------------------------
Implements "Right to be Forgotten".
"""

from typing import Dict
from .gdpr_exceptions import GDPRRequestError

class GDPRDataDeletionService:
    """Handles user data deletion requests."""

    def __init__(self, database: Dict[str, dict]):
        self.database = database

    def delete_user_data(self, user_id: str) -> Dict[str, str]:
        try:
            if user_id not in self.database:
                raise GDPRRequestError(f"No records found for user {user_id}")
            deleted_data = self.database.pop(user_id)
            return {"status": "success", "deleted_fields": str(list(deleted_data.keys()))}
        except Exception as e:
            raise GDPRRequestError(f"Deletion failed: {e}")
