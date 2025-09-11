# common/libraries/python/auth/auth_service.py
"""
Main AuthService tying together all authentication features.
"""

from typing import Optional
from . import hash_utils, jwt_utils, mfa
from .session_manager import InMemorySessionStore
from .rate_limiter import RateLimiter
from .models import UserCredentials

class AuthService:
    def __init__(self):
        self.session_store = InMemorySessionStore()
        self.rate_limiter = RateLimiter()
        self.users: dict[str, UserCredentials] = {}

    def register_user(self, username: str, password: str, enable_mfa: bool = False) -> UserCredentials:
        """Register a new user with optional MFA."""
        password_hash = hash_utils.hash_password(password)
        mfa_secret = mfa.generate_mfa_secret() if enable_mfa else None
        creds = UserCredentials(username=username, password_hash=password_hash, mfa_secret=mfa_secret)
        self.users[username] = creds
        return creds

    def authenticate_user(self, username: str, password: str, mfa_token: Optional[str] = None) -> str:
        """Authenticate user and return JWT."""
        if not self.rate_limiter.is_allowed(username):
            raise PermissionError("Too many login attempts, try later.")

        creds = self.users.get(username)
        if not creds or not hash_utils.verify_password(password, creds.password_hash):
            raise PermissionError("Invalid credentials")

        if creds.mfa_secret and (not mfa_token or not mfa.verify_mfa_token(creds.mfa_secret, mfa_token)):
            raise PermissionError("MFA required or invalid")

        return jwt_utils.generate_jwt(subject=username)

    def validate_token(self, token: str) -> dict:
        """Validate a JWT and return claims."""
        return jwt_utils.verify_jwt(token)

    def create_session(self, user_id: str, metadata: Optional[dict] = None):
        """Create a user session."""
        return self.session_store.create_session(user_id, metadata)

    def end_session(self, session_id: str):
        """Terminate a session."""
        self.session_store.delete_session(session_id)
