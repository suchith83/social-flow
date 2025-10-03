"""
Unified Storage Abstraction Layer.

This module provides a clean interface for all storage operations,
supporting multiple cloud providers (AWS S3, Azure Blob, Google Cloud Storage).
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, BinaryIO, Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum


class StorageProvider(str, Enum):
    """Supported storage providers."""
    S3 = "s3"
    AZURE = "azure"
    GCS = "gcs"
    LOCAL = "local"


@dataclass
class StorageMetadata:
    """Metadata for stored objects."""
    key: str
    bucket: str
    size: int
    content_type: str
    provider: StorageProvider
    url: Optional[str] = None
    etag: Optional[str] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    modified_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Optional[Dict[str, str]] = None


class IStorageBackend(ABC):
    """Abstract interface for storage backends."""

    @abstractmethod
    async def upload(
        self,
        data: bytes,
        key: str,
        bucket: Optional[str] = None,
        content_type: str = "application/octet-stream",
        metadata: Optional[Dict[str, str]] = None,
    ) -> StorageMetadata:
        """Upload data to storage."""
        pass

    @abstractmethod
    async def download(
        self,
        key: str,
        bucket: Optional[str] = None,
    ) -> bytes:
        """Download data from storage."""
        pass

    @abstractmethod
    async def delete(
        self,
        key: str,
        bucket: Optional[str] = None,
    ) -> bool:
        """Delete object from storage."""
        pass

    @abstractmethod
    async def exists(
        self,
        key: str,
        bucket: Optional[str] = None,
    ) -> bool:
        """Check if object exists."""
        pass

    @abstractmethod
    async def get_metadata(
        self,
        key: str,
        bucket: Optional[str] = None,
    ) -> StorageMetadata:
        """Get object metadata."""
        pass

    @abstractmethod
    async def generate_presigned_url(
        self,
        key: str,
        bucket: Optional[str] = None,
        expires_in: int = 3600,
        operation: str = "get",
    ) -> str:
        """Generate presigned URL for temporary access."""
        pass

    @abstractmethod
    async def copy(
        self,
        source_key: str,
        dest_key: str,
        source_bucket: Optional[str] = None,
        dest_bucket: Optional[str] = None,
    ) -> StorageMetadata:
        """Copy object to a new location."""
        pass

    @abstractmethod
    async def list_objects(
        self,
        prefix: str = "",
        bucket: Optional[str] = None,
        max_keys: int = 1000,
    ) -> list[StorageMetadata]:
        """List objects with optional prefix filter."""
        pass


class IMultipartUpload(ABC):
    """Interface for multipart upload operations (large files)."""

    @abstractmethod
    async def initiate_multipart_upload(
        self,
        key: str,
        bucket: Optional[str] = None,
        content_type: str = "application/octet-stream",
    ) -> str:
        """Initiate multipart upload and return upload ID."""
        pass

    @abstractmethod
    async def upload_part(
        self,
        upload_id: str,
        part_number: int,
        data: bytes,
        key: str,
        bucket: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Upload a single part."""
        pass

    @abstractmethod
    async def complete_multipart_upload(
        self,
        upload_id: str,
        key: str,
        parts: list[Dict[str, Any]],
        bucket: Optional[str] = None,
    ) -> StorageMetadata:
        """Complete multipart upload."""
        pass

    @abstractmethod
    async def abort_multipart_upload(
        self,
        upload_id: str,
        key: str,
        bucket: Optional[str] = None,
    ) -> bool:
        """Abort multipart upload and clean up."""
        pass


class IStorageManager(ABC):
    """High-level storage manager interface."""

    @abstractmethod
    async def upload_file(
        self,
        file: BinaryIO,
        key: str,
        bucket: Optional[str] = None,
        content_type: Optional[str] = None,
        metadata: Optional[Dict[str, str]] = None,
    ) -> StorageMetadata:
        """Upload file with automatic provider selection."""
        pass

    @abstractmethod
    async def upload_large_file(
        self,
        file: BinaryIO,
        key: str,
        bucket: Optional[str] = None,
        content_type: Optional[str] = None,
        chunk_size: int = 5 * 1024 * 1024,
        on_progress: Optional[Callable[[int, int], None]] = None,
    ) -> StorageMetadata:
        """Upload large file using multipart upload."""
        pass

    @abstractmethod
    async def download_file(
        self,
        key: str,
        bucket: Optional[str] = None,
    ) -> bytes:
        """Download file."""
        pass

    @abstractmethod
    async def get_public_url(
        self,
        key: str,
        bucket: Optional[str] = None,
        expires_in: int = 3600,
    ) -> str:
        """Get temporary public URL."""
        pass

    @abstractmethod
    async def delete_file(
        self,
        key: str,
        bucket: Optional[str] = None,
    ) -> bool:
        """Delete file."""
        pass

    @abstractmethod
    def switch_provider(
        self,
        provider: StorageProvider,
    ) -> None:
        """Switch active storage provider."""
        pass
