"""In-Memory Storage Backend (Fallback).

Provides a lightweight, dependency-free storage backend used automatically
when no cloud storage providers (e.g., S3) are configured. This enables the
test suite and local development flows to proceed without requiring cloud
credentials while exercising the unified storage interfaces.

Limitations:
 - Data is ephemeral (process memory only)
 - Not suitable for production usage
 - Multipart upload operations are not implemented; the manager will
   transparently fall back to regular upload for large files
"""

from __future__ import annotations

import asyncio
import base64
import os
from datetime import datetime, timezone
from typing import Dict, Optional, Any, List

from app.infrastructure.storage.base import (
    IStorageBackend,
    StorageMetadata,
    StorageProvider,
)


class MemoryBackend(IStorageBackend):
    """Simple in-memory storage backend.

    Stored objects are kept in a dictionary keyed by (bucket, key).
    """

    def __init__(self, default_bucket: str = "local-bucket"):
        self.default_bucket = default_bucket
        self._store: Dict[tuple[str, str], Dict[str, Any]] = {}
        self._lock = asyncio.Lock()

    def _bucket(self, bucket: Optional[str]) -> str:
        return bucket or self.default_bucket

    async def upload(
        self,
        data: bytes,
        key: str,
        bucket: Optional[str] = None,
        content_type: str = "application/octet-stream",
        metadata: Optional[Dict[str, str]] = None,
    ) -> StorageMetadata:
        bucket = self._bucket(bucket)
        created = datetime.now(timezone.utc)
        etag = base64.urlsafe_b64encode(os.urandom(9)).decode("utf-8").rstrip("=")
        meta = StorageMetadata(
            key=key,
            bucket=bucket,
            size=len(data),
            content_type=content_type,
            provider=StorageProvider.LOCAL,
            etag=etag,
            created_at=created,
            modified_at=created,
            metadata=metadata or {},
        )
        async with self._lock:
            self._store[(bucket, key)] = {"data": data, "meta": meta}
        return meta

    async def download(
        self,
        key: str,
        bucket: Optional[str] = None,
    ) -> bytes:
        bucket = self._bucket(bucket)
        async with self._lock:
            entry = self._store.get((bucket, key))
            if not entry:
                raise FileNotFoundError(key)
            return entry["data"]

    async def delete(
        self,
        key: str,
        bucket: Optional[str] = None,
    ) -> bool:
        bucket = self._bucket(bucket)
        async with self._lock:
            return self._store.pop((bucket, key), None) is not None

    async def exists(
        self,
        key: str,
        bucket: Optional[str] = None,
    ) -> bool:
        bucket = self._bucket(bucket)
        async with self._lock:
            return (bucket, key) in self._store

    async def get_metadata(
        self,
        key: str,
        bucket: Optional[str] = None,
    ) -> StorageMetadata:
        bucket = self._bucket(bucket)
        async with self._lock:
            entry = self._store.get((bucket, key))
            if not entry:
                raise FileNotFoundError(key)
            return entry["meta"]

    async def generate_presigned_url(
        self,
        key: str,
        bucket: Optional[str] = None,
        expires_in: int = 3600,
        operation: str = "get",
    ) -> str:
        bucket = self._bucket(bucket)
        token = base64.urlsafe_b64encode(os.urandom(12)).decode("utf-8").rstrip("=")
        return f"http://local-storage/{bucket}/{key}?op={operation}&exp={expires_in}&t={token}"

    async def copy(
        self,
        source_key: str,
        dest_key: str,
        source_bucket: Optional[str] = None,
        dest_bucket: Optional[str] = None,
    ) -> StorageMetadata:
        src_bucket = self._bucket(source_bucket)
        dst_bucket = self._bucket(dest_bucket)
        async with self._lock:
            entry = self._store.get((src_bucket, source_key))
            if not entry:
                raise FileNotFoundError(source_key)
            data = entry["data"]
        # Re-upload to destination (creates fresh metadata)
        return await self.upload(
            data=data,
            key=dest_key,
            bucket=dst_bucket,
            content_type=entry["meta"].content_type,
            metadata=entry["meta"].metadata,
        )

    async def list_objects(
        self,
        prefix: str = "",
        bucket: Optional[str] = None,
        max_keys: int = 1000,
    ) -> List[StorageMetadata]:
        bucket = self._bucket(bucket)
        async with self._lock:
            results = [
                entry["meta"]
                for (b, k), entry in self._store.items()
                if b == bucket and k.startswith(prefix)
            ]
        return results[:max_keys]
