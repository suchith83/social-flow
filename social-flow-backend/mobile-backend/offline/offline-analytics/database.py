# Database connection/session manager
"""
Database engine + session management for analytics module.
"""

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from .config import get_config
from .models import Base

config = get_config()
DATABASE_URL = config.DATABASE_URL

# SQLite example, but production should use Postgres or managed DB
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_db():
    Base.metadata.create_all(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
