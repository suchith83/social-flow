"""SQLAlchemy ORM setup with CockroachDB dialect."""
"""
orm.py
------
SQLAlchemy ORM integration with CockroachDB.
Includes advanced session handling, async support, and base models.
"""

from sqlalchemy import create_engine, Column, Integer, String, DateTime, func
from sqlalchemy.orm import declarative_base, sessionmaker, scoped_session
import yaml
from pathlib import Path

Base = declarative_base()


class User(Base):
    """Example User table for CockroachDB ORM testing."""
    __tablename__ = "users"

    id = Column(Integer, primary_key=True)
    username = Column(String(50), unique=True, nullable=False)
    email = Column(String(120), unique=True, nullable=False)
    created_at = Column(DateTime, server_default=func.now())


def load_config():
    """Load YAML CockroachDB config."""
    with open(Path("config/databases/cockroachdb/config.yaml"), "r") as f:
        return yaml.safe_load(f)


def get_engine():
    """Return SQLAlchemy engine with CockroachDB dialect."""
    config = load_config()["database"]
    conn_str = (
        f"cockroachdb://{config['user']}:{config['password']}@"
        f"{config['nodes'][0]['host']}:{config['nodes'][0]['port']}/{config['name']}"
        "?sslmode=require"
    )
    return create_engine(conn_str, pool_size=config["pool"]["max"], echo=False)


engine = get_engine()
SessionLocal = scoped_session(sessionmaker(autocommit=False, autoflush=False, bind=engine))
