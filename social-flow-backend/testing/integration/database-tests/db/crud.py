"""
CRUD helpers (sync + async variants) used by tests to exercise DB operations.
These are intentionally simple and robust to illustrate typical patterns.
"""

from typing import Optional, List
from sqlalchemy import select, insert, update, delete
from db.models import User, Item, Base
from db.sync_client import SessionLocal
from sqlalchemy.ext.asyncio import AsyncSession
import sqlalchemy
import hashlib

# Helper hashing for test passwords (not production)
def _hash_password(plain: str) -> str:
    return hashlib.sha256(plain.encode("utf-8")).hexdigest()

# ----------------------
# Sync CRUD
# ----------------------
def create_user_sync(session: SessionLocal, username: str, password: str, email: Optional[str] = None) -> User:
    hashed = _hash_password(password)
    user = User(username=username, hashed_password=hashed, email=email)
    session.add(user)
    session.commit()
    session.refresh(user)
    return user

def get_user_by_username_sync(session: SessionLocal, username: str) -> Optional[User]:
    stmt = select(User).where(User.username == username)
    res = session.execute(stmt).scalars().first()
    return res

def delete_user_sync(session: SessionLocal, username: str) -> int:
    stmt = delete(User).where(User.username == username)
    res = session.execute(stmt)
    session.commit()
    return res.rowcount

# ----------------------
# Async CRUD
# ----------------------
async def create_user_async(session: AsyncSession, username: str, password: str, email: Optional[str] = None) -> User:
    hashed = _hash_password(password)
    stmt = insert(User).values(username=username, hashed_password=hashed, email=email).returning(User)
    res = await session.execute(stmt)
    await session.commit()
    user = res.fetchone()[0]
    return user

async def get_user_by_username_async(session: AsyncSession, username: str) -> Optional[User]:
    stmt = select(User).where(User.username == username)
    res = await session.execute(stmt)
    user = res.scalars().first()
    return user
