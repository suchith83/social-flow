
# performance/scaling/sharding/__init__.py

"""
Sharding Package
================

Provides abstractions for horizontal partitioning (sharding) of data
across multiple nodes or databases.

Features:
- Sharding algorithms (hash-based, range-based, consistent hashing)
- Shard metadata management
- Query dispatching to correct shard
- Rebalancing support when nodes are added/removed
- Monitoring of shard distribution

Usage:
    from performance.scaling.sharding import Orchestrator, Config

    config = Config.load("sharding.yaml")
    orchestrator = Orchestrator(config)
    orchestrator.start()
"""

from .config import Config
from .orchestrator import Orchestrator
from .exceptions import ShardingError
