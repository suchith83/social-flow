"""
Security utilities.

This module contains security-related utilities including password hashing,
JWT token creation and validation, and other security functions.
"""

from datetime import datetime, timedelta
from typing import Any, Dict, Optional

from jose import JWTError, jwt
from passlib.context import CryptContext

from app.core.config import settings

# Password hashing context
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def create_access_token(
    data: Dict[str, Any],
    expires_delta: Optional[timedelta] = None,
) -> str:
    """Create JWT access token."""
    to_encode = data.copy()
    
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)
    
    return encoded_jwt


def create_refresh_token(
    data: Dict[str, Any],
    expires_delta: Optional[timedelta] = None,
) -> str:
    """Create JWT refresh token."""
    to_encode = data.copy()
    
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS)
    
    to_encode.update({"exp": expire, "type": "refresh"})
    encoded_jwt = jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)
    
    return encoded_jwt


def verify_token(token: str) -> Optional[Dict[str, Any]]:
    """Verify JWT token and return payload."""
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        return payload
    except JWTError:
        return None


def get_password_hash(password: str) -> str:
    """Hash password using bcrypt."""
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify password against hash."""
    return pwd_context.verify(plain_password, hashed_password)


def generate_password_reset_token(email: str) -> str:
    """Generate password reset token."""
    delta = timedelta(hours=1)  # Token expires in 1 hour
    now = datetime.utcnow()
    expires = now + delta
    exp = expires.timestamp()
    encoded_jwt = jwt.encode(
        {"exp": exp, "nbf": now, "sub": email, "type": "password_reset"},
        settings.SECRET_KEY,
        algorithm=settings.ALGORITHM,
    )
    return encoded_jwt


def verify_password_reset_token(token: str) -> Optional[str]:
    """Verify password reset token and return email."""
    try:
        decoded_token = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        if decoded_token.get("type") != "password_reset":
            return None
        return decoded_token.get("sub")
    except JWTError:
        return None


def generate_email_verification_token(email: str) -> str:
    """Generate email verification token."""
    delta = timedelta(hours=24)  # Token expires in 24 hours
    now = datetime.utcnow()
    expires = now + delta
    exp = expires.timestamp()
    encoded_jwt = jwt.encode(
        {"exp": exp, "nbf": now, "sub": email, "type": "email_verification"},
        settings.SECRET_KEY,
        algorithm=settings.ALGORITHM,
    )
    return encoded_jwt


def verify_email_verification_token(token: str) -> Optional[str]:
    """Verify email verification token and return email."""
    try:
        decoded_token = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        if decoded_token.get("type") != "email_verification":
            return None
        return decoded_token.get("sub")
    except JWTError:
        return None


def generate_api_key() -> str:
    """Generate API key for external integrations."""
    import secrets
    return secrets.token_urlsafe(32)


def generate_otp() -> str:
    """Generate 6-digit OTP for two-factor authentication."""
    import random
    return str(random.randint(100000, 999999))


def validate_password_strength(password: str) -> Dict[str, Any]:
    """Validate password strength and return validation result."""
    result = {
        "is_valid": True,
        "score": 0,
        "feedback": [],
    }
    
    if len(password) < 8:
        result["is_valid"] = False
        result["feedback"].append("Password must be at least 8 characters long")
    else:
        result["score"] += 1
    
    if not any(c.isupper() for c in password):
        result["feedback"].append("Password should contain at least one uppercase letter")
    else:
        result["score"] += 1
    
    if not any(c.islower() for c in password):
        result["feedback"].append("Password should contain at least one lowercase letter")
    else:
        result["score"] += 1
    
    if not any(c.isdigit() for c in password):
        result["feedback"].append("Password should contain at least one number")
    else:
        result["score"] += 1
    
    if not any(c in "!@#$%^&*(),.?\":{}|<>" for c in password):
        result["feedback"].append("Password should contain at least one special character")
    else:
        result["score"] += 1
    
    # Check for common patterns
    common_patterns = ["123", "abc", "password", "qwerty", "admin"]
    if any(pattern in password.lower() for pattern in common_patterns):
        result["feedback"].append("Password contains common patterns")
        result["score"] -= 1
    
    return result


def sanitize_input(input_string: str) -> str:
    """Sanitize user input to prevent XSS attacks."""
    import html
    return html.escape(input_string.strip())


def validate_email(email: str) -> bool:
    """Validate email format."""
    import re
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None


def validate_username(username: str) -> Dict[str, Any]:
    """Validate username format and availability."""
    result = {
        "is_valid": True,
        "feedback": [],
    }
    
    if len(username) < 3:
        result["is_valid"] = False
        result["feedback"].append("Username must be at least 3 characters long")
    
    if len(username) > 20:
        result["is_valid"] = False
        result["feedback"].append("Username must be at most 20 characters long")
    
    if not username.replace("_", "").isalnum():
        result["is_valid"] = False
        result["feedback"].append("Username can only contain letters, numbers, and underscores")
    
    if username.startswith("_") or username.endswith("_"):
        result["is_valid"] = False
        result["feedback"].append("Username cannot start or end with underscore")
    
    # Check for reserved usernames
    reserved_usernames = [
        "admin", "administrator", "root", "api", "www", "mail", "ftp", "support",
        "help", "about", "contact", "privacy", "terms", "legal", "blog", "news",
        "status", "security", "download", "upload", "files", "media", "static",
        "assets", "images", "videos", "audio", "documents", "public", "private",
        "internal", "external", "test", "testing", "dev", "development", "staging",
        "production", "prod", "live", "beta", "alpha", "preview", "demo",
    ]
    
    if username.lower() in reserved_usernames:
        result["is_valid"] = False
        result["feedback"].append("Username is reserved")
    
    return result
