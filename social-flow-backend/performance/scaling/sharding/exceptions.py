# Sharding-related exceptions
# performance/scaling/sharding/exceptions.py

class ShardingError(Exception):
    """Base exception for sharding."""


class AlgorithmError(ShardingError):
    """Raised when sharding algorithm fails."""


class DispatchError(ShardingError):
    """Raised when dispatch to shard fails."""


class RebalanceError(ShardingError):
    """Raised when rebalancing shards fails."""
