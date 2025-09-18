import pytest
from unittest.mock import MagicMock

class AuthService:
    def __init__(self, user_repo):
        self.user_repo = user_repo

    def authenticate(self, username, password):
        user = self.user_repo.get(username)
        if not user:
            return False
        return user["password"] == password

def test_authenticate_success():
    repo = {"alice": {"password": "123"}}
    service = AuthService(repo)
    assert service.authenticate("alice", "123") is True

def test_authenticate_failure():
    repo = {"bob": {"password": "abc"}}
    service = AuthService(repo)
    assert service.authenticate("bob", "wrong") is False

def test_authenticate_unknown_user():
    service = AuthService({})
    assert service.authenticate("ghost", "pwd") is False
