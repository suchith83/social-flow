import pytest
from unittest.mock import MagicMock

class UserService:
    def __init__(self, db, cache):
        self.db = db
        self.cache = cache

    def create_user(self, user_id, username):
        if user_id in self.db["users"]:
            raise ValueError("User already exists")
        self.db["users"][user_id] = {"username": username}
        self.cache[user_id] = username
        return True

    def get_user(self, user_id):
        return self.cache.get(user_id) or self.db["users"].get(user_id)

@pytest.fixture
def user_service(fake_db, mock_cache):
    return UserService(fake_db, mock_cache)

def test_create_user_success(user_service):
    assert user_service.create_user("u1", "Alice") is True
    assert user_service.get_user("u1")["username"] == "Alice"

def test_create_user_duplicate(user_service):
    user_service.create_user("u2", "Bob")
    with pytest.raises(ValueError):
        user_service.create_user("u2", "BobAgain")
