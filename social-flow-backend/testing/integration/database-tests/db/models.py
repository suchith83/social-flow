"""
SQLAlchemy models for testing.

This file defines small example tables to exercise migrations and CRUD operations.
The models use SQLAlchemy 1.4+ declarative base.
"""

from sqlalchemy import Column, Integer, String, Boolean, Text, DateTime, func
from sqlalchemy.orm import declarative_base
from config import settings

Base = declarative_base()

TEST_PREFIX = settings.test_schema_prefix or ""

def prefixed(name: str) -> str:
    """Return table name with optional test prefix to avoid collisions."""
    return f"{TEST_PREFIX}{name}"

class User(Base):
    __tablename__ = prefixed("users")
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(150), unique=True, nullable=False, index=True)
    email = Column(String(255), nullable=True)
    hashed_password = Column(String(255), nullable=False)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

class Item(Base):
    __tablename__ = prefixed("items")
    id = Column(Integer, primary_key=True)
    owner_id = Column(Integer, index=True)
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
