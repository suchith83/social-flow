"""
Authentication token models re-export.

This module re-exports the consolidated EmailVerificationToken and PasswordResetToken 
from app.models.user.
"""

# Re-export consolidated token models
from app.models.user import (
    EmailVerificationToken,
    PasswordResetToken,
)

# For backward compatibility
__all__ = ["EmailVerificationToken", "PasswordResetToken"]
