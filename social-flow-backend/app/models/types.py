"""
Cross-database type utilities.

This module provides SQLAlchemy types that work across different database backends
(PostgreSQL and SQLite) for testing purposes.
"""

from sqlalchemy import JSON, String, TypeDecorator
from sqlalchemy.dialects import postgresql


class JSONB(TypeDecorator):
    """
    JSONB type that works with both PostgreSQL and SQLite.
    
    Uses PostgreSQL's JSONB when available, falls back to JSON for SQLite.
    """
    
    impl = JSON
    cache_ok = True
    
    def load_dialect_impl(self, dialect):
        if dialect.name == 'postgresql':
            return dialect.type_descriptor(postgresql.JSONB())
        else:
            return dialect.type_descriptor(JSON())


class ARRAY(TypeDecorator):
    """
    ARRAY type that works with both PostgreSQL and SQLite.
    
    Uses PostgreSQL's ARRAY when available, falls back to JSON for SQLite.
    """
    
    impl = JSON
    cache_ok = True
    
    def __init__(self, item_type=None, *args, **kwargs):
        """Initialize with optional item type (ignored for JSON fallback)."""
        self.item_type = item_type
        super().__init__(*args, **kwargs)
    
    def load_dialect_impl(self, dialect):
        if dialect.name == 'postgresql':
            return dialect.type_descriptor(
                postgresql.ARRAY(self.item_type or String)
            )
        else:
            # For SQLite, store as JSON array
            return dialect.type_descriptor(JSON())


class UUID(TypeDecorator):
    """
    UUID type that works with both PostgreSQL and SQLite.
    
    Uses PostgreSQL's UUID when available, falls back to String(36) for SQLite.
    Handles conversion between UUID objects and strings automatically.
    """
    
    impl = String(36)
    cache_ok = True
    
    def load_dialect_impl(self, dialect):
        if dialect.name == 'postgresql':
            return dialect.type_descriptor(postgresql.UUID())
        else:
            # For SQLite, store as string
            return dialect.type_descriptor(String(36))
    
    def process_bind_param(self, value, dialect):
        """Convert UUID to string for SQLite."""
        if value is None:
            return value
        if dialect.name == 'postgresql':
            return value
        else:
            # Convert UUID to string for SQLite
            if isinstance(value, str):
                return value
            return str(value)
    
    def process_result_value(self, value, dialect):
        """Convert string back to UUID from SQLite."""
        if value is None:
            return value
        if dialect.name == 'postgresql':
            return value
        else:
            # Convert string back to UUID for consistency
            import uuid as uuid_module
            if isinstance(value, uuid_module.UUID):
                return value
            return uuid_module.UUID(value) if value else None
