"""
# Implements erasure/deletion requests
"""
"""
CCPA Data Deletion Service
--------------------------
Implements logic for processing consumer data deletion requests.
"""

from typing import Dict
from .ccpa_exceptions import CCPARequestError

class CCPADataDeletionService:
    """Service to handle consumer data deletion requests."""

    def __init__(self, database: Dict[str, dict]):
        """
        :param database: simulated in-memory database {user_id: user_data}
        """
        self.database = database

    def delete_user_data(self, user_id: str) -> Dict[str, str]:
        """
        Delete all personal data for a user.
        """
        try:
            if user_id not in self.database:
                raise CCPARequestError(f"No records found for user {user_id}")
            
            deleted_data = self.database.pop(user_id)
            return {"status": "success", "deleted_records": str(list(deleted_data.keys()))}
        except Exception as e:
            raise CCPARequestError(f"Deletion failed: {e}")
