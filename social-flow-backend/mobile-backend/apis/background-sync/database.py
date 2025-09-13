"""
Database connection manager for Background Sync.
Ensures thread-safe sessions and schema creation.
"""

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from .config import get_config
from .models import Base

config = get_config()

engine = create_engine(config.DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_db():
    Base.metadata.create_all(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
