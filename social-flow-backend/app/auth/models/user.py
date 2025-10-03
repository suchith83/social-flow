"""
User model re-export.

This module re-exports the consolidated User model from app.models.user.
All imports should now use: from app.models.user import User
"""

# Re-export consolidated User model
from app.models.user import (
    User,
    UserStatus,
    UserRole,
    PrivacyLevel,
)

# For backward compatibility
__all__ = ["User", "UserStatus", "UserRole", "PrivacyLevel"]
