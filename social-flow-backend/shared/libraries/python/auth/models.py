# common/libraries/python/auth/models.py
"""
Models used in authentication library.
Framework agnostic: pure Python dataclasses.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

@dataclass
class UserCredentials:
    username: str
    password_hash: str
    mfa_secret: Optional[str] = None

@dataclass
class JWTClaims:
    sub: str
    exp: datetime
    iss: str
    iat: datetime
    additional_claims: dict

@dataclass
class Session:
    session_id: str
    user_id: str
    created_at: datetime
    expires_at: datetime
    metadata: dict
