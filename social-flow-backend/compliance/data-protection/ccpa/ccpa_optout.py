"""
# Opt-out / Do-Not-Sell preference handling
"""
"""
CCPA Opt-Out Service
--------------------
Implements "Do Not Sell My Personal Information" functionality.
"""

from datetime import datetime, timedelta

class CCPAOptOutService:
    """Handles consumer opt-out requests."""

    def __init__(self):
        self.optouts = {}  # user_id -> expiry datetime

    def opt_out(self, user_id: str):
        """Mark a user as opted-out for 12 months."""
        expiry = datetime.utcnow() + timedelta(days=365)
        self.optouts[user_id] = expiry
        return {"status": "success", "optout_expires": expiry.isoformat()}

    def is_opted_out(self, user_id: str) -> bool:
        """Check if a user is still opted-out."""
        expiry = self.optouts.get(user_id)
        return expiry is not None and datetime.utcnow() <= expiry
