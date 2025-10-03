"""JWT Token Authentication Tests."""

import pytest
from datetime import datetime, timedelta
from jose import jwt, JWTError
from app.core.config import settings


class TestAuthJWT:
    """Test JWT token generation and validation."""

    def test_token_generation_success(self):
        """Test successful JWT token generation."""
        payload = {
            "user_id": "123",
            "username": "testuser",
            "exp": datetime.utcnow() + timedelta(hours=1)
        }
        token = jwt.encode(payload, settings.SECRET_KEY, algorithm="HS256")
        assert token is not None
        assert isinstance(token, str)
        assert len(token) > 0

    def test_token_validation_success(self):
        """Test successful JWT token validation and decoding."""
        payload = {"user_id": "123", "username": "testuser"}
        token = jwt.encode(payload, settings.SECRET_KEY, algorithm="HS256")
        
        decoded = jwt.decode(token, settings.SECRET_KEY, algorithms=["HS256"])
        assert decoded["user_id"] == payload["user_id"]
        assert decoded["username"] == payload["username"]

    def test_token_expiration(self):
        """Test JWT token expiration handling."""
        payload = {
            "user_id": "123",
            "exp": datetime.utcnow() - timedelta(hours=1)  # Expired
        }
        token = jwt.encode(payload, settings.SECRET_KEY, algorithm="HS256")
        
        with pytest.raises(JWTError):
            jwt.decode(token, settings.SECRET_KEY, algorithms=["HS256"])

    def test_invalid_token(self):
        """Test validation with invalid token."""
        with pytest.raises(JWTError):
            jwt.decode("invalid_token", settings.SECRET_KEY, algorithms=["HS256"])

    def test_wrong_secret_key(self):
        """Test token validation with wrong secret key."""
        payload = {"user_id": "123"}
        token = jwt.encode(payload, "wrong_secret", algorithm="HS256")
        
        with pytest.raises(JWTError):
            jwt.decode(token, settings.SECRET_KEY, algorithms=["HS256"])

    @pytest.mark.parametrize("algorithm", ["HS256", "HS384", "HS512"])
    def test_different_algorithms(self, algorithm):
        """Test JWT with different HS algorithms."""
        payload = {"user_id": "123"}
        token = jwt.encode(payload, settings.SECRET_KEY, algorithm=algorithm)
        decoded = jwt.decode(token, settings.SECRET_KEY, algorithms=[algorithm])
        assert decoded["user_id"] == "123"