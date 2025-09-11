# common/libraries/python/auth/__init__.py
"""
Authentication Library - Framework Agnostic

Provides:
- Password hashing (Argon2, bcrypt fallback)
- JWT-based authentication
- OAuth2 support
- Session management
- Multi-Factor Authentication (MFA)
- Rate limiting for brute-force protection
"""

__all__ = [
    "config",
    "models",
    "hash_utils",
    "jwt_utils",
    "oauth2",
    "session_manager",
    "mfa",
    "rate_limiter",
    "auth_service",
]
