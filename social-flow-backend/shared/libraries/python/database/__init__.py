# common/libraries/python/database/__init__.py
"""
Database Library - Framework Agnostic

Features:
- Connection pooling for PostgreSQL, MySQL, SQLite
- Safe query builder
- Lightweight ORM
- Transaction support
- Migration engine
- Caching (in-memory + Redis-ready)
"""

__all__ = [
    "config",
    "connection",
    "query_builder",
    "orm",
    "migrations",
    "cache",
    "database_service",
]
