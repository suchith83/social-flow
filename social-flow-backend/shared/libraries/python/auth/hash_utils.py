# common/libraries/python/auth/hash_utils.py
"""
Password hashing utilities with Argon2 and bcrypt fallback.
"""

from passlib.context import CryptContext
from .config import AuthConfig

# Configure hash context
pwd_context = CryptContext(
    schemes=["argon2", "bcrypt"],
    default=AuthConfig.HASH_SCHEME,
    deprecated="auto",
)

def hash_password(password: str) -> str:
    """Generate a secure password hash."""
    return pwd_context.hash(password)

def verify_password(password: str, hashed: str) -> bool:
    """Verify a password against its hash."""
    return pwd_context.verify(password, hashed)
