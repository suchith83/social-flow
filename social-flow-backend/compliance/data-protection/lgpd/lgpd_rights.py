"""
# Rights of data subjects: access, rectification, deletion, portability
"""
# compliance/data-protection/lgpd/lgpd_rights.py
"""
LGPD Data Subject Rights
------------------------
Implements mechanisms for enforcing LGPD Articles 17–22:
- Right of access
- Right of correction
- Right of deletion
- Right of anonymization
- Right of portability
"""

import datetime
from typing import Dict, Any, List


class LGPDSubjectRights:
    def __init__(self, datastore: Dict[str, Dict[str, Any]]):
        """
        :param datastore: A dict simulating user data storage.
                          {user_id: {"name": "...", "email": "...", ...}}
        """
        self.datastore = datastore

    def access_data(self, user_id: str) -> Dict[str, Any]:
        """Right of access – return all personal data of the user."""
        return self.datastore.get(user_id, {})

    def correct_data(self, user_id: str, updates: Dict[str, Any]) -> bool:
        """Right of correction – update inaccurate data."""
        if user_id in self.datastore:
            self.datastore[user_id].update(updates)
            return True
        return False

    def delete_data(self, user_id: str) -> bool:
        """Right of deletion – erase user’s personal data."""
        return self.datastore.pop(user_id, None) is not None

    def anonymize_data(self, user_id: str) -> bool:
        """Right of anonymization – strip identifiers while retaining non-PII."""
        if user_id not in self.datastore:
            return False
        data = self.datastore[user_id]
        anonymized = {k: ("***" if isinstance(v, str) else None) for k, v in data.items()}
        self.datastore[user_id] = anonymized
        return True

    def export_data(self, user_id: str) -> Dict[str, Any]:
        """Right of portability – export user’s data in portable format (JSON)."""
        return {
            "exported_at": datetime.datetime.utcnow().isoformat(),
            "user_id": user_id,
            "data": self.datastore.get(user_id, {}),
        }
