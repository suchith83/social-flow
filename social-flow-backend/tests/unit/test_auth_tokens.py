import pytest
from app.core.security import create_access_token, verify_password, get_password_hash
from datetime import timedelta

@pytest.mark.parametrize("password", ["p@ssw0rd123", "anotherSecret!"])
def test_password_hash_roundtrip(password):
    hashed = get_password_hash(password)
    assert verify_password(password, hashed) is True
    assert verify_password(password + "x", hashed) is False

def test_access_token_creation():
    token = create_access_token({"sub": "user@example.com", "user_id": 1}, expires_delta=timedelta(minutes=5))
    assert isinstance(token, str) and len(token) > 10