# common/libraries/python/auth/config.py
"""
Configuration module for authentication library.
Supports environment variable overrides.
"""

import os
from datetime import timedelta

class AuthConfig:
    # JWT
    JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "RS256")
    JWT_EXPIRY = timedelta(minutes=int(os.getenv("JWT_EXPIRY_MINUTES", "30")))
    JWT_PRIVATE_KEY = os.getenv("JWT_PRIVATE_KEY", None)  # Load RSA private key string
    JWT_PUBLIC_KEY = os.getenv("JWT_PUBLIC_KEY", None)    # Load RSA public key string
    JWT_ISSUER = os.getenv("JWT_ISSUER", "social-flow")

    # Password hashing
    HASH_SCHEME = os.getenv("HASH_SCHEME", "argon2")

    # Session
    SESSION_TTL = int(os.getenv("SESSION_TTL", "3600"))  # in seconds

    # MFA
    MFA_ISSUER = os.getenv("MFA_ISSUER", "SocialFlow")

    # Rate limiting
    RATE_LIMIT_ATTEMPTS = int(os.getenv("RATE_LIMIT_ATTEMPTS", "5"))
    RATE_LIMIT_WINDOW = int(os.getenv("RATE_LIMIT_WINDOW", "900"))  # 15 min
