"""
Enhanced Database Management with Sharding and Read Replicas

This module provides advanced database functionality:
- Async SQLAlchemy 2.0 engine management
- Database sharding for horizontal scaling
- Read replica support for read-heavy workloads
- Connection pooling with health checks
- Transaction management with retry logic
- Query optimization utilities
"""

import asyncio
import hashlib
import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Dict, Optional, Callable, Any, List

from sqlalchemy import MetaData, event, text
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    AsyncEngine,
    async_sessionmaker,
    create_async_engine,
    AsyncConnection,
)
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.pool import NullPool, QueuePool

try:
    from app.core.config_enhanced import settings
except ImportError:
    from app.core.config import settings  # Fallback

logger = logging.getLogger(__name__)

# ============================================================================
# DATABASE METADATA & BASE
# ============================================================================

metadata = MetaData(
    naming_convention={
        "ix": "ix_%(column_0_label)s",
        "uq": "uq_%(table_name)s_%(column_0_name)s",
        "ck": "ck_%(table_name)s_%(constraint_name)s",
        "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
        "pk": "pk_%(table_name)s",
    }
)


class Base(DeclarativeBase):
    """SQLAlchemy declarative base for all models."""
    metadata = metadata


# ============================================================================
# ENGINE MANAGEMENT
# ============================================================================

class DatabaseManager:
    """
    Manages database connections with support for sharding and read replicas.
    
    Features:
    - Primary write engine
    - Multiple read replica engines for load balancing
    - Shard engines for horizontal partitioning
    - Automatic failover and retry logic
    - Connection pool health monitoring
    """
    
    def __init__(self):
        self._primary_engine: Optional[AsyncEngine] = None
        self._read_replica_engines: List[AsyncEngine] = []
        self._shard_engines: Dict[str, AsyncEngine] = {}
        self._session_maker: Optional[async_sessionmaker] = None
        self._current_read_replica_index: int = 0
        self._initialized: bool = False
    
    def _create_engine(
        self,
        url: str,
        pool_size: int = 20,
        max_overflow: int = 40,
        echo: bool = False,
    ) -> AsyncEngine:
        """Create an async SQLAlchemy engine with optimal settings."""
        
        # Use appropriate pooling based on driver
        if "sqlite" in url.lower():
            poolclass = NullPool  # SQLite doesn't support connection pooling well
            pool_kwargs = {}
        else:
            poolclass = QueuePool
            pool_kwargs = {
                "pool_size": pool_size,
                "max_overflow": max_overflow,
                "pool_timeout": 30,
                "pool_recycle": 3600,
                "pool_pre_ping": True,  # Verify connections before use
            }
        
        engine = create_async_engine(
            url,
            echo=echo,
            future=True,
            poolclass=poolclass,
            **pool_kwargs,
        )
        
        # Add event listeners for monitoring
        self._add_engine_listeners(engine)
        
        return engine
    
    def _add_engine_listeners(self, engine: AsyncEngine) -> None:
        """Add event listeners for connection monitoring and debugging."""
        
        @event.listens_for(engine.sync_engine, "connect")
        def receive_connect(dbapi_conn, connection_record):
            """Log new database connections."""
            logger.debug(f"New database connection established: {id(dbapi_conn)}")
        
        @event.listens_for(engine.sync_engine, "checkout")
        def receive_checkout(dbapi_conn, connection_record, connection_proxy):
            """Log connection checkouts from pool."""
            logger.debug(f"Connection checked out from pool: {id(dbapi_conn)}")
        
        @event.listens_for(engine.sync_engine, "checkin")
        def receive_checkin(dbapi_conn, connection_record):
            """Log connection returns to pool."""
            logger.debug(f"Connection returned to pool: {id(dbapi_conn)}")
    
    async def initialize(self) -> None:
        """
        Initialize database connections.
        
        Sets up:
        - Primary write engine
        - Read replica engines (if configured)
        - Shard engines (if sharding enabled)
        """
        if self._initialized:
            logger.warning("Database manager already initialized")
            return
        
        logger.info("Initializing database manager...")
        
        # Primary database (write operations)
        database_url = str(settings.DATABASE_URL)
        self._primary_engine = self._create_engine(
            database_url,
            pool_size=settings.DB_POOL_SIZE,
            max_overflow=settings.DB_MAX_OVERFLOW,
            echo=settings.DB_ECHO,
        )
        logger.info(f"Primary database engine initialized: {database_url.split('@')[-1]}")
        
        # Read replicas (read operations)
        if settings.DATABASE_READ_REPLICAS:
            for idx, replica_url in enumerate(settings.DATABASE_READ_REPLICAS):
                engine = self._create_engine(
                    replica_url,
                    pool_size=settings.DB_POOL_SIZE,
                    max_overflow=settings.DB_MAX_OVERFLOW,
                    echo=settings.DB_ECHO,
                )
                self._read_replica_engines.append(engine)
                logger.info(f"Read replica {idx + 1} initialized: {replica_url.split('@')[-1]}")
        
        # Shard engines (horizontal partitioning)
        if settings.DB_SHARDING_ENABLED and settings.DB_SHARDS:
            for shard_name, shard_url in settings.DB_SHARDS.items():
                engine = self._create_engine(
                    shard_url,
                    pool_size=settings.DB_POOL_SIZE // 2,  # Smaller pool per shard
                    max_overflow=settings.DB_MAX_OVERFLOW // 2,
                    echo=settings.DB_ECHO,
                )
                self._shard_engines[shard_name] = engine
                logger.info(f"Shard '{shard_name}' initialized: {shard_url.split('@')[-1]}")
        
        # Session maker
        self._session_maker = async_sessionmaker(
            self._primary_engine,
            class_=AsyncSession,
            expire_on_commit=False,
            autoflush=False,
            autocommit=False,
        )
        
        self._initialized = True
        logger.info("Database manager initialization complete")
    
    async def close(self) -> None:
        """Close all database connections gracefully."""
        if not self._initialized:
            return
        
        logger.info("Closing database connections...")
        
        if self._primary_engine:
            await self._primary_engine.dispose()
            logger.info("Primary engine disposed")
        
        for idx, engine in enumerate(self._read_replica_engines):
            await engine.dispose()
            logger.info(f"Read replica {idx + 1} disposed")
        
        for shard_name, engine in self._shard_engines.items():
            await engine.dispose()
            logger.info(f"Shard '{shard_name}' disposed")
        
        self._initialized = False
        logger.info("All database connections closed")
    
    def get_primary_engine(self) -> AsyncEngine:
        """Get the primary (write) database engine."""
        if not self._primary_engine:
            raise RuntimeError("Database manager not initialized")
        return self._primary_engine
    
    def get_read_engine(self) -> AsyncEngine:
        """
        Get a read replica engine using round-robin load balancing.
        Falls back to primary engine if no replicas configured.
        """
        if not self._read_replica_engines:
            return self.get_primary_engine()
        
        # Round-robin selection
        engine = self._read_replica_engines[self._current_read_replica_index]
        self._current_read_replica_index = (
            self._current_read_replica_index + 1
        ) % len(self._read_replica_engines)
        
        return engine
    
    def get_shard_engine(self, shard_key: str) -> AsyncEngine:
        """
        Get a shard engine based on a sharding key.
        
        Args:
            shard_key: Key to determine shard (e.g., user_id, video_id)
        
        Returns:
            Appropriate shard engine
        
        Raises:
            RuntimeError: If sharding not enabled or shard not found
        """
        if not settings.DB_SHARDING_ENABLED:
            return self.get_primary_engine()
        
        if not self._shard_engines:
            raise RuntimeError("Sharding enabled but no shards configured")
        
        # Hash-based sharding
        shard_index = self._calculate_shard_index(shard_key)
        shard_names = sorted(self._shard_engines.keys())
        shard_name = shard_names[shard_index % len(shard_names)]
        
        return self._shard_engines[shard_name]
    
    @staticmethod
    def _calculate_shard_index(shard_key: str) -> int:
        """Calculate shard index from key using consistent hashing."""
        hash_value = int(hashlib.md5(shard_key.encode()).hexdigest(), 16)
        return hash_value % settings.DB_SHARD_COUNT
    
    @asynccontextmanager
    async def session(
        self,
        readonly: bool = False,
        shard_key: Optional[str] = None,
    ) -> AsyncGenerator[AsyncSession, None]:
        """
        Get a database session with automatic transaction management.
        
        Args:
            readonly: If True, use read replica (if available)
            shard_key: If provided, use appropriate shard
        
        Yields:
            AsyncSession: Database session
        
        Example:
            async with db_manager.session() as session:
                result = await session.execute(query)
        """
        if not self._initialized:
            await self.initialize()
        
        # Determine which engine to use
        if shard_key:
            engine = self.get_shard_engine(shard_key)
        elif readonly:
            engine = self.get_read_engine()
        else:
            engine = self.get_primary_engine()
        
        # Create session
        session_factory = async_sessionmaker(
            engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )
        
        async with session_factory() as session:
            try:
                yield session
                if not readonly:
                    await session.commit()
            except Exception as e:
                if not readonly:
                    await session.rollback()
                logger.error(f"Database session error: {e}", exc_info=True)
                raise
            finally:
                await session.close()
    
    async def execute_with_retry(
        self,
        func: Callable,
        *args,
        max_retries: int = 3,
        retry_delay: float = 0.5,
        **kwargs,
    ) -> Any:
        """
        Execute a database operation with automatic retry logic.
        
        Args:
            func: Async function to execute
            max_retries: Maximum number of retry attempts
            retry_delay: Delay between retries (seconds)
        
        Returns:
            Result of the function
        
        Raises:
            Last exception if all retries fail
        """
        last_exception = None
        
        for attempt in range(max_retries):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                if attempt < max_retries - 1:
                    logger.warning(
                        f"Database operation failed (attempt {attempt + 1}/{max_retries}): {e}"
                    )
                    await asyncio.sleep(retry_delay * (attempt + 1))
                else:
                    logger.error(f"Database operation failed after {max_retries} attempts: {e}")
        
        raise last_exception  # type: ignore
    
    async def health_check(self) -> Dict[str, bool]:
        """
        Check health of all database connections.
        
        Returns:
            Dict mapping connection names to health status
        """
        health = {}
        
        # Check primary
        try:
            async with self._primary_engine.connect() as conn:
                await conn.execute(text("SELECT 1"))
            health["primary"] = True
        except Exception as e:
            logger.error(f"Primary database health check failed: {e}")
            health["primary"] = False
        
        # Check replicas
        for idx, engine in enumerate(self._read_replica_engines):
            replica_name = f"replica_{idx + 1}"
            try:
                async with engine.connect() as conn:
                    await conn.execute(text("SELECT 1"))
                health[replica_name] = True
            except Exception as e:
                logger.error(f"Read replica {idx + 1} health check failed: {e}")
                health[replica_name] = False
        
        # Check shards
        for shard_name, engine in self._shard_engines.items():
            try:
                async with engine.connect() as conn:
                    await conn.execute(text("SELECT 1"))
                health[f"shard_{shard_name}"] = True
            except Exception as e:
                logger.error(f"Shard '{shard_name}' health check failed: {e}")
                health[f"shard_{shard_name}"] = False
        
        return health


# ============================================================================
# GLOBAL DATABASE MANAGER
# ============================================================================

db_manager = DatabaseManager()


# ============================================================================
# LEGACY COMPATIBILITY
# ============================================================================

# For backward compatibility with existing code
_engine: Optional[AsyncEngine] = None
_session_maker: Optional[async_sessionmaker] = None


def get_engine() -> AsyncEngine:
    """Get the primary database engine (legacy compatibility)."""
    return db_manager.get_primary_engine()


def get_session_maker() -> async_sessionmaker:
    """Get the session maker (legacy compatibility)."""
    if not db_manager._session_maker:
        raise RuntimeError("Database not initialized")
    return db_manager._session_maker


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    FastAPI dependency to get database session.
    
    Usage:
        @app.get("/users")
        async def get_users(db: AsyncSession = Depends(get_db)):
            result = await db.execute(select(User))
            return result.scalars().all()
    """
    async with db_manager.session() as session:
        yield session


async def get_db_readonly() -> AsyncGenerator[AsyncSession, None]:
    """
    FastAPI dependency to get read-only database session.
    Uses read replicas if available.
    
    Usage:
        @app.get("/users")
        async def get_users(db: AsyncSession = Depends(get_db_readonly)):
            result = await db.execute(select(User))
            return result.scalars().all()
    """
    async with db_manager.session(readonly=True) as session:
        yield session


async def init_db() -> None:
    """Initialize database connections."""
    await db_manager.initialize()


async def close_db() -> None:
    """Close all database connections."""
    await db_manager.close()


# ============================================================================
# QUERY OPTIMIZATION UTILITIES
# ============================================================================

class QueryOptimizer:
    """Utilities for optimizing database queries."""
    
    @staticmethod
    def batch_size_for_table(table_name: str) -> int:
        """Get optimal batch size for bulk operations on a table."""
        # Customize based on table size and complexity
        batch_sizes = {
            "users": 500,
            "videos": 200,
            "posts": 500,
            "comments": 1000,
            "likes": 1000,
            "views": 2000,
        }
        return batch_sizes.get(table_name, 500)
    
    @staticmethod
    async def explain_query(session: AsyncSession, query: Any) -> Dict[str, Any]:
        """Get query execution plan for optimization."""
        explained = await session.execute(text(f"EXPLAIN ANALYZE {query}"))
        return {"plan": explained.fetchall()}


# Export all public APIs
__all__ = [
    "Base",
    "metadata",
    "db_manager",
    "DatabaseManager",
    "get_engine",
    "get_session_maker",
    "get_db",
    "get_db_readonly",
    "init_db",
    "close_db",
    "QueryOptimizer",
]
