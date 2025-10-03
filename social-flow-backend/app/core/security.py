"""
Security utilities.

This module contains security-related utilities including password hashing,
JWT token creation and validation, and other security functions.
"""

from datetime import datetime, timedelta
from typing import Any, Dict, Optional

import bcrypt
from jose import JWTError, jwt

from app.core.config import settings


def create_access_token(
    data: Optional[Dict[str, Any]] = None,
    subject: Optional[Any] = None,
    expires_delta: Optional[timedelta] = None,
    additional_claims: Optional[dict] = None,
) -> str:
    """
    Create JWT access token.
    
    Supports both old signature (data dict) and new signature (subject + additional_claims).
    """
    import uuid
    
    if subject is not None:
        # New signature
        to_encode = {
            "sub": str(subject),
            "type": "access",
            "jti": str(uuid.uuid4()),  # Unique JWT ID for each token
        }
        if additional_claims:
            to_encode.update(additional_claims)
    elif data is not None:
        # Old signature for backward compatibility
        to_encode = data.copy()
        to_encode["jti"] = str(uuid.uuid4())  # Add JTI for backward compatibility too
    else:
        raise ValueError("Either 'data' or 'subject' must be provided")
    
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)
    
    return encoded_jwt


def create_refresh_token(
    data: Optional[Dict[str, Any]] = None,
    subject: Optional[Any] = None,
    expires_delta: Optional[timedelta] = None,
) -> str:
    """
    Create JWT refresh token.
    
    Supports both old signature (data dict) and new signature (subject).
    """
    if subject is not None:
        # New signature
        to_encode = {
            "sub": str(subject),
            "type": "refresh",
        }
    elif data is not None:
        # Old signature for backward compatibility
        to_encode = data.copy()
        to_encode["type"] = "refresh"
    else:
        raise ValueError("Either 'data' or 'subject' must be provided")
    
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS)
    
    to_encode.update({"exp": expire})
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
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
    return hashed.decode('utf-8')


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify password against hash."""
    try:
        return bcrypt.checkpw(
            plain_password.encode('utf-8'),
            hashed_password.encode('utf-8')
        )
    except Exception:
        return False


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


def generate_otp(digits: int = 6) -> str:
    """Generate a time-independent OTP using cryptographically secure randomness.

    Args:
        digits: Number of digits for the OTP (default 6)

    Returns:
        str: Zero-padded numeric OTP string
    """
    import secrets
    if digits < 4 or digits > 10:
        digits = 6
    max_value = 10**digits
    value = secrets.randbelow(max_value)
    return str(value).zfill(digits)


def validate_password_strength(password: str) -> Dict[str, Any]:
    """Validate password strength and return validation result."""
    result: Dict[str, Any] = {
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
    result: Dict[str, Any] = {
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


def decode_token(token: str) -> dict:
    """
    Decode and validate JWT token.
    
    Args:
        token: JWT token string
        
    Returns:
        dict: Decoded token payload
        
    Raises:
        JWTError: If token is invalid or expired
    """
    return jwt.decode(
        token,
        settings.SECRET_KEY,
        algorithms=[settings.ALGORITHM],
    )


def verify_token_type(token_data: dict, expected_type: str) -> bool:
    """
    Verify that token is of expected type.
    
    Args:
        token_data: Decoded token payload
        expected_type: Expected token type
        
    Returns:
        bool: True if token type matches
    """
    return token_data.get("type") == expected_type


def create_email_verification_token(user_id) -> str:
    """
    Create token for email verification.
    
    Args:
        user_id: User ID
        
    Returns:
        str: Email verification token
    """
    from uuid import UUID
    expire = datetime.utcnow() + timedelta(hours=24)
    to_encode = {
        "exp": expire,
        "sub": str(user_id),
        "type": "email_verification",
    }
    return jwt.encode(
        to_encode,
        settings.SECRET_KEY,
        algorithm=settings.ALGORITHM,
    )


def create_password_reset_token(user_id) -> str:
    """
    Create token for password reset.
    
    Args:
        user_id: User ID
        
    Returns:
        str: Password reset token
    """
    expire = datetime.utcnow() + timedelta(hours=1)
    to_encode = {
        "exp": expire,
        "sub": str(user_id),
        "type": "password_reset",
    }
    return jwt.encode(
        to_encode,
        settings.SECRET_KEY,
        algorithm=settings.ALGORITHM,
    )


def generate_2fa_secret() -> str:
    """
    Generate a secret key for 2FA (TOTP).
    
    Returns:
        str: Base32-encoded secret
    """
    try:
        import pyotp
        return pyotp.random_base32()
    except ImportError:
        # Fallback if pyotp not installed
        import secrets
        import base64
        random_bytes = secrets.token_bytes(20)
        return base64.b32encode(random_bytes).decode('utf-8')


def verify_2fa_token(secret: str, token: str) -> bool:
    """
    Verify 2FA token against secret.
    
    Args:
        secret: Base32-encoded secret
        token: 6-digit TOTP token
        
    Returns:
        bool: True if token is valid
    """
    try:
        import pyotp
        totp = pyotp.TOTP(secret)
        return totp.verify(token, valid_window=1)
    except ImportError:
        # Fallback - always return False if pyotp not installed
        return False


def generate_2fa_qr_uri(secret: str, username: str) -> str:
    """
    Generate QR code URI for 2FA setup.
    
    Args:
        secret: Base32-encoded secret
        username: User's username or email
        
    Returns:
        str: otpauth:// URI for QR code
    """
    try:
        import pyotp
        totp = pyotp.TOTP(secret)
        return totp.provisioning_uri(
            name=username,
            issuer_name=settings.PROJECT_NAME,
        )
    except ImportError:
        # Fallback - return basic otpauth URL
        from urllib.parse import quote
        return f"otpauth://totp/{settings.PROJECT_NAME}:{quote(username)}?secret={secret}&issuer={quote(settings.PROJECT_NAME)}"
