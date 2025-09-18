"""
Async CRUD tests (pytest-asyncio required).
"""

import pytest
import asyncio
from db.crud import create_user_async, get_user_by_username_async
from sqlalchemy.ext.asyncio import AsyncSession

@pytest.mark.integration
@pytest.mark.asyncio
async def test_create_and_get_user_async(db_async_session: AsyncSession):
    username = f"u_async_{__import__('random').randint(1000,9999)}"
    user = await create_user_async(db_async_session, username=username, password="secret", email="async@example.com")
    assert getattr(user, "id", None) is not None
    fetched = await get_user_by_username_async(db_async_session, username)
    assert fetched is not None
    assert fetched.username == username
