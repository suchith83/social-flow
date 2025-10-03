"""
Unified Storage Infrastructure Package.

Provides a clean, unified interface for all storage operations across
multiple cloud providers (AWS S3, Azure Blob, Google Cloud Storage).
"""

from app.infrastructure.storage.base import (
    IStorageBackend,
    IStorageManager,
    IMultipartUpload,
    StorageMetadata,
    StorageProvider,
)
from app.infrastructure.storage.s3_backend import S3Backend
from app.infrastructure.storage.manager import (
    StorageManager,
    get_storage_manager,
    initialize_storage,
)

__all__ = [
    # Interfaces
    "IStorageBackend",
    "IStorageManager",
    "IMultipartUpload",
    
    # Data Classes
    "StorageMetadata",
    "StorageProvider",
    
    # Implementations
    "S3Backend",
    "StorageManager",
    
    # Utility Functions
    "get_storage_manager",
    "initialize_storage",
]
