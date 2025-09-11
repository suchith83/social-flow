# common/libraries/python/auth/jwt_utils.py
"""
JWT utility functions for token generation and validation.
Supports RS256 asymmetric signing.
"""

import jwt
from datetime import datetime, timedelta, timezone
from typing import Dict, Any
from .config import AuthConfig

def generate_jwt(subject: str, additional_claims: Dict[str, Any] = None) -> str:
    """Generate a signed JWT."""
    if not AuthConfig.JWT_PRIVATE_KEY:
        raise RuntimeError("JWT private key not configured")

    now = datetime.now(timezone.utc)
    payload = {
        "sub": subject,
        "iat": now,
        "exp": now + AuthConfig.JWT_EXPIRY,
        "iss": AuthConfig.JWT_ISSUER,
    }
    if additional_claims:
        payload.update(additional_claims)

    return jwt.encode(payload, AuthConfig.JWT_PRIVATE_KEY, algorithm=AuthConfig.JWT_ALGORITHM)

def verify_jwt(token: str) -> Dict[str, Any]:
    """Verify and decode a JWT."""
    if not AuthConfig.JWT_PUBLIC_KEY:
        raise RuntimeError("JWT public key not configured")

    return jwt.decode(
        token,
        AuthConfig.JWT_PUBLIC_KEY,
        algorithms=[AuthConfig.JWT_ALGORITHM],
        issuer=AuthConfig.JWT_ISSUER,
    )
