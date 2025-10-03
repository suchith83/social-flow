"""
JWT Token Handler - Auth Infrastructure

Handles JWT token generation, validation, and refresh token management.
"""

from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from uuid import UUID

from jose import jwt, JWTError

from app.core.config import settings


class JWTHandler:
    """
    JWT token handler for authentication.
    
    Manages access tokens and refresh tokens with expiration.
    """
    
    def __init__(
        self,
        secret_key: str,
        algorithm: str = "HS256",
        access_token_expire_minutes: int = 30,
        refresh_token_expire_days: int = 7,
    ):
        """
        Initialize JWT handler.
        
        Args:
            secret_key: Secret key for signing tokens
            algorithm: JWT algorithm (default: HS256)
            access_token_expire_minutes: Access token lifetime in minutes
            refresh_token_expire_days: Refresh token lifetime in days
        """
        self._secret_key = secret_key
        self._algorithm = algorithm
        self._access_token_expire = timedelta(minutes=access_token_expire_minutes)
        self._refresh_token_expire = timedelta(days=refresh_token_expire_days)
    
    def create_access_token(
        self,
        user_id: UUID,
        username: str,
        role: str,
        additional_claims: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Create an access token.
        
        Args:
            user_id: User's unique identifier
            username: User's username
            role: User's role
            additional_claims: Optional additional claims to include
            
        Returns:
            JWT access token as a string
        """
        now = datetime.utcnow()
        expires_at = now + self._access_token_expire
        
        payload = {
            "sub": str(user_id),
            "username": username,
            "role": role,
            "type": "access",
            "iat": now,
            "exp": expires_at,
        }
        
        if additional_claims:
            payload.update(additional_claims)
        
        token = jwt.encode(payload, self._secret_key, algorithm=self._algorithm)
        return token
    
    def create_refresh_token(
        self,
        user_id: UUID,
        additional_claims: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Create a refresh token.
        
        Args:
            user_id: User's unique identifier
            additional_claims: Optional additional claims to include
            
        Returns:
            JWT refresh token as a string
        """
        now = datetime.utcnow()
        expires_at = now + self._refresh_token_expire
        
        payload = {
            "sub": str(user_id),
            "type": "refresh",
            "iat": now,
            "exp": expires_at,
        }
        
        if additional_claims:
            payload.update(additional_claims)
        
        token = jwt.encode(payload, self._secret_key, algorithm=self._algorithm)
        return token
    
    def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """
        Verify and decode a JWT token.
        
        Args:
            token: JWT token to verify
            
        Returns:
            Decoded token payload if valid, None otherwise
        """
        try:
            payload = jwt.decode(
                token,
                self._secret_key,
                algorithms=[self._algorithm],
            )
            return payload
        except JWTError:
            return None
    
    def verify_access_token(self, token: str) -> Optional[Dict[str, Any]]:
        """
        Verify an access token specifically.
        
        Args:
            token: Access token to verify
            
        Returns:
            Decoded payload if valid access token, None otherwise
        """
        payload = self.verify_token(token)
        if payload and payload.get("type") == "access":
            return payload
        return None
    
    def verify_refresh_token(self, token: str) -> Optional[Dict[str, Any]]:
        """
        Verify a refresh token specifically.
        
        Args:
            token: Refresh token to verify
            
        Returns:
            Decoded payload if valid refresh token, None otherwise
        """
        payload = self.verify_token(token)
        if payload and payload.get("type") == "refresh":
            return payload
        return None
    
    def get_user_id_from_token(self, token: str) -> Optional[UUID]:
        """
        Extract user ID from a token.
        
        Args:
            token: JWT token
            
        Returns:
            User ID if token is valid, None otherwise
        """
        payload = self.verify_token(token)
        if payload and "sub" in payload:
            try:
                return UUID(payload["sub"])
            except (ValueError, TypeError):
                return None
        return None
    
    def is_token_expired(self, token: str) -> bool:
        """
        Check if a token is expired.
        
        Args:
            token: JWT token to check
            
        Returns:
            True if expired, False otherwise
        """
        try:
            jwt.decode(
                token,
                self._secret_key,
                algorithms=[self._algorithm],
            )
            return False
        except JWTError:
            # Any JWT error is considered expired
            return True


# Create singleton instance for application use
jwt_handler = JWTHandler(
    secret_key=settings.SECRET_KEY,
    algorithm="HS256",
    access_token_expire_minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES,
    refresh_token_expire_days=7,
)
