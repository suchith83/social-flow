"""
Sync CRUD tests using the sync session fixture.
"""

import pytest
from db.crud import create_user_sync, get_user_by_username_sync, delete_user_sync

@pytest.mark.integration
def test_create_and_get_user(db_session):
    username = f"u_{pytest.temp_label if hasattr(pytest,'temp_label') else 'sync'}_{__import__('random').randint(1000,9999)}"
    user = create_user_sync(db_session, username=username, password="secret", email="x@example.com")
    assert user.id is not None
    fetched = get_user_by_username_sync(db_session, username)
    assert fetched is not None
    assert fetched.username == username

@pytest.mark.integration
def test_delete_user(db_session):
    username = f"del_{__import__('random').randint(1000,9999)}"
    create_user_sync(db_session, username=username, password="pw")
    count = delete_user_sync(db_session, username)
    assert count == 1
    assert get_user_by_username_sync(db_session, username) is None
