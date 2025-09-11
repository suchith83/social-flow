# common/libraries/python/auth/mfa.py
"""
Multi-Factor Authentication using TOTP (Google Authenticator compatible).
"""

import pyotp
from .config import AuthConfig

def generate_mfa_secret() -> str:
    """Generate a base32 MFA secret."""
    return pyotp.random_base32()

def get_mfa_uri(username: str, secret: str) -> str:
    """Generate provisioning URI for QR code."""
    return pyotp.totp.TOTP(secret).provisioning_uri(name=username, issuer_name=AuthConfig.MFA_ISSUER)

def verify_mfa_token(secret: str, token: str) -> bool:
    """Verify a TOTP token."""
    totp = pyotp.TOTP(secret)
    return totp.verify(token)
