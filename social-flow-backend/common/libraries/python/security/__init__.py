# __init__.py
"""
Security Library
================
Enterprise-grade Python security library for distributed systems.

Features:
- Cryptography (AES, RSA, HMAC, hashing)
- JWT authentication
- Password hashing (PBKDF2, bcrypt, Argon2)
- Role-based access control (RBAC)
- Rate limiting
- Input sanitization
- Secrets management (Vault, AWS KMS, env)
"""

from .crypto import Crypto
from .jwt_manager import JWTManager
from .password import PasswordHasher
from .auth import Authenticator
from .rbac import RBAC
from .rate_limit import RateLimiter
from .sanitizer import Sanitizer
from .secrets_manager import SecretsManager

__all__ = [
    "Crypto",
    "JWTManager",
    "PasswordHasher",
    "Authenticator",
    "RBAC",
    "RateLimiter",
    "Sanitizer",
    "SecretsManager"
]
