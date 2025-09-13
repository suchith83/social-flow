# Database connection/session manager
"""
Database engine and session management for the caching subsystem.
"""

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from .config import get_config
from .models import Base

config = get_config()

# Use SQLite for example; replace with Postgres/Cloud SQL in prod
DATABASE_URL = getattr(config, "DATABASE_URL", "sqlite:///./content_cache.db")

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_db():
    """
    Create tables if not present. Use Alembic for production migrations.
    """
    Base.metadata.create_all(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
