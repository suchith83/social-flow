"""
Multi-Cloud Object Storage Abstraction Layer.

Provides a unified interface to work with AWS S3, Azure Blob, and Google Cloud Storage.
"""

from .manager import MultiCloudStorageManager
from .factory import StorageFactory
