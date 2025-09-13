# Package initializer for storage module
"""
Storage Optimization Package

Provides tiered storage management, adaptive compression, deduplication,
asynchronous write pipelines, garbage collection and metrics for a mobile backend.

Author: Enterprise Backend Team
"""

from .storage_manager import StorageManager
from .tiered_storage import TieredStorage
from .deduplication import Deduplicator
from .compression_adapter import CompressionAdapter
from .async_writer import AsyncWriter
from .garbage_collector import GarbageCollector
from .metrics import StorageMetrics
from .storage_backends import LocalDiskBackend, S3Backend, ColdArchiveBackend

__all__ = [
    "StorageManager",
    "TieredStorage",
    "Deduplicator",
    "CompressionAdapter",
    "AsyncWriter",
    "GarbageCollector",
    "StorageMetrics",
    "LocalDiskBackend",
    "S3Backend",
    "ColdArchiveBackend",
]
