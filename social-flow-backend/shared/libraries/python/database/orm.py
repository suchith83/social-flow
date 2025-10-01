# common/libraries/python/database/orm.py
"""
Lightweight ORM (SQLAlchemy-powered).
Provides Base model for inheritance.
"""

from sqlalchemy.orm import declarative_base, declared_attr
from sqlalchemy import Column, DateTime, func

class BaseModel:
    """Base class with id and timestamp columns."""

    @declared_attr
    def __tablename__(cls):
        return cls.__name__.lower()

    id = Column("id", primary_key=True, index=True, autoincrement=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

Base = declarative_base(cls=BaseModel)
