
"""
Database Optimization Package

Provides tools for database performance optimization:
- Connection pooling (async + threaded)
- Query optimization & rewriting
- Indexing advisors
- Caching strategies
- Sharding & partitioning utilities
- Replication management
- Monitoring & metrics collection
"""

from .connection_pooling import AsyncConnectionPool, ThreadedConnectionPool
from .query_optimization import QueryOptimizer
from .indexing import IndexAdvisor
from .caching import QueryCache
from .sharding import ShardManager
from .replication import ReplicationManager
from .monitoring import DatabaseMetricsCollector

__all__ = [
    "AsyncConnectionPool",
    "ThreadedConnectionPool",
    "QueryOptimizer",
    "IndexAdvisor",
    "QueryCache",
    "ShardManager",
    "ReplicationManager",
    "DatabaseMetricsCollector",
]
