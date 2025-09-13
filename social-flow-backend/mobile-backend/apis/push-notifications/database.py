# Database connection/session manager
"""
DB engine + session management for push notifications module.
"""

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from .config import get_config
from .models import Base

config = get_config()

# SQLite special connect_args for single-file examples; real deployments use Postgres/MySQL
engine = create_engine(config.DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_db():
    """Create tables. In real projects use Alembic migrations."""
    Base.metadata.create_all(bind=engine)


def get_db():
    """FastAPI dependency generator for DB sessions."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
