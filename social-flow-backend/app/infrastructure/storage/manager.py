"""
Unified Storage Manager.

This module provides a high-level interface for all storage operations,
automatically handling provider selection, multipart uploads, and error handling.
"""

import logging
import mimetypes
from typing import Optional, Dict, Any, BinaryIO, Callable

from app.infrastructure.storage.base import (
    IStorageManager,
    IStorageBackend,
    StorageMetadata,
    StorageProvider,
)
from app.infrastructure.storage.s3_backend import S3Backend
from app.core.config import settings
from app.core.exceptions import StorageServiceError

logger = logging.getLogger(__name__)


class StorageManager(IStorageManager):
    """Unified storage manager with automatic provider management."""

    # Threshold for multipart upload (5MB)
    MULTIPART_THRESHOLD = 5 * 1024 * 1024

    def __init__(
        self,
        default_provider: StorageProvider = StorageProvider.S3,
    ):
        """Initialize storage manager with specified default provider."""
        self._backends: Dict[StorageProvider, IStorageBackend] = {}
        self._active_provider = default_provider
        self._initialize_backends()

    def _initialize_backends(self):
        """Initialize available storage backends based on configuration."""
        # Initialize S3 backend if credentials are available
        if all([settings.AWS_ACCESS_KEY_ID, settings.AWS_SECRET_ACCESS_KEY, settings.AWS_REGION]):
            try:
                self._backends[StorageProvider.S3] = S3Backend()
                logger.info("S3 backend initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize S3 backend: {e}")

        # TODO: Initialize Azure backend
        # if settings.AZURE_STORAGE_CONNECTION_STRING:
        #     self._backends[StorageProvider.AZURE] = AzureBackend()

        # TODO: Initialize GCS backend
        # if settings.GCS_CREDENTIALS_PATH:
        #     self._backends[StorageProvider.GCS] = GCSBackend()

        if not self._backends:
            raise StorageServiceError("No storage backends available. Please configure at least one provider.")

        # Set active provider to first available if default is not available
        if self._active_provider not in self._backends:
            self._active_provider = list(self._backends.keys())[0]
            logger.warning(f"Default provider not available. Using {self._active_provider}")

    def _get_backend(self) -> IStorageBackend:
        """Get current active backend."""
        backend = self._backends.get(self._active_provider)
        if not backend:
            raise StorageServiceError(f"Backend for provider {self._active_provider} not available")
        return backend

    def switch_provider(self, provider: StorageProvider) -> None:
        """Switch active storage provider."""
        if provider not in self._backends:
            raise StorageServiceError(f"Provider {provider} not available")
        
        old_provider = self._active_provider
        self._active_provider = provider
        logger.info(f"Switched storage provider from {old_provider} to {provider}")

    @staticmethod
    def _detect_content_type(filename: str) -> str:
        """Detect content type from filename."""
        content_type, _ = mimetypes.guess_type(filename)
        return content_type or "application/octet-stream"

    async def upload_file(
        self,
        file: BinaryIO,
        key: str,
        bucket: Optional[str] = None,
        content_type: Optional[str] = None,
        metadata: Optional[Dict[str, str]] = None,
    ) -> StorageMetadata:
        """Upload file with automatic provider selection."""
        # Read file data
        data = file.read()
        file_size = len(data)

        # Detect content type if not provided
        if content_type is None:
            content_type = self._detect_content_type(key)

        # Use multipart upload for large files
        if file_size > self.MULTIPART_THRESHOLD:
            # Reset file pointer
            file.seek(0)
            return await self.upload_large_file(
                file=file,
                key=key,
                bucket=bucket,
                content_type=content_type,
            )

        # Regular upload for small files
        backend = self._get_backend()
        
        try:
            result = await backend.upload(
                data=data,
                key=key,
                bucket=bucket,
                content_type=content_type,
                metadata=metadata,
            )
            
            logger.info(f"Uploaded file {key} ({file_size} bytes) using {self._active_provider}")
            return result

        except Exception as e:
            logger.error(f"Failed to upload file {key}: {e}")
            raise StorageServiceError(f"Upload failed: {str(e)}")

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
        backend = self._get_backend()

        # Check if backend supports multipart upload
        supports_multipart = all(
            hasattr(backend, name) and callable(getattr(backend, name, None))
            for name in (
                "initiate_multipart_upload",
                "upload_part",
                "complete_multipart_upload",
                "abort_multipart_upload",
            )
        )
        if not supports_multipart:
            # Fallback to regular upload without re-entering upload_file (avoids recursion)
            logger.warning(
                f"Backend {self._active_provider} doesn't support multipart upload. Using regular upload."
            )
            if content_type is None:
                content_type = self._detect_content_type(key)
            # Ensure we read from the start
            try:
                file.seek(0)
            except Exception:
                pass
            data = file.read()
            try:
                return await backend.upload(
                    data=data,
                    key=key,
                    bucket=bucket,
                    content_type=content_type,
                )
            except Exception as e:
                logger.error(f"Failed to upload file {key} via regular upload fallback: {e}")
                raise StorageServiceError(f"Upload failed: {str(e)}")

        if content_type is None:
            content_type = self._detect_content_type(key)

        upload_id = None
        try:
            # Initiate multipart upload
            upload_id = await backend.initiate_multipart_upload(
                key=key,
                bucket=bucket,
                content_type=content_type,
            )

            parts = []
            part_number = 1
            total_uploaded = 0

            # Upload parts
            while True:
                chunk = file.read(chunk_size)
                if not chunk:
                    break

                part_info = await backend.upload_part(
                    upload_id=upload_id,
                    part_number=part_number,
                    data=chunk,
                    key=key,
                    bucket=bucket,
                )
                parts.append(part_info)

                total_uploaded += len(chunk)
                if on_progress:
                    on_progress(total_uploaded, part_number)

                part_number += 1

            # Complete multipart upload
            result = await backend.complete_multipart_upload(
                upload_id=upload_id,
                key=key,
                parts=parts,
                bucket=bucket,
            )

            logger.info(f"Completed multipart upload for {key} ({total_uploaded} bytes, {len(parts)} parts)")
            return result

        except Exception as e:
            # Abort multipart upload on error
            if upload_id and supports_multipart:
                try:
                    await backend.abort_multipart_upload(upload_id, key, bucket)
                    logger.info(f"Aborted multipart upload {upload_id} for {key}")
                except Exception as abort_error:
                    logger.error(f"Failed to abort multipart upload: {abort_error}")

            logger.error(f"Failed to upload large file {key}: {e}")
            raise StorageServiceError(f"Large file upload failed: {str(e)}")

    async def download_file(
        self,
        key: str,
        bucket: Optional[str] = None,
    ) -> bytes:
        """Download file from storage."""
        backend = self._get_backend()

        try:
            data = await backend.download(key=key, bucket=bucket)
            logger.info(f"Downloaded file {key} ({len(data)} bytes) from {self._active_provider}")
            return data

        except Exception as e:
            logger.error(f"Failed to download file {key}: {e}")
            raise StorageServiceError(f"Download failed: {str(e)}")

    async def get_public_url(
        self,
        key: str,
        bucket: Optional[str] = None,
        expires_in: int = 3600,
    ) -> str:
        """Get temporary public URL for file access."""
        backend = self._get_backend()

        try:
            url = await backend.generate_presigned_url(
                key=key,
                bucket=bucket,
                expires_in=expires_in,
                operation="get",
            )
            logger.debug(f"Generated presigned URL for {key} (expires in {expires_in}s)")
            return url

        except Exception as e:
            logger.error(f"Failed to generate public URL for {key}: {e}")
            raise StorageServiceError(f"Failed to generate public URL: {str(e)}")

    async def delete_file(
        self,
        key: str,
        bucket: Optional[str] = None,
    ) -> bool:
        """Delete file from storage."""
        backend = self._get_backend()

        try:
            success = await backend.delete(key=key, bucket=bucket)
            if success:
                logger.info(f"Deleted file {key} from {self._active_provider}")
            return success

        except Exception as e:
            logger.error(f"Failed to delete file {key}: {e}")
            raise StorageServiceError(f"Delete failed: {str(e)}")

    async def file_exists(
        self,
        key: str,
        bucket: Optional[str] = None,
    ) -> bool:
        """Check if file exists in storage."""
        backend = self._get_backend()

        try:
            return await backend.exists(key=key, bucket=bucket)

        except Exception as e:
            logger.error(f"Failed to check existence of {key}: {e}")
            return False

    async def get_file_metadata(
        self,
        key: str,
        bucket: Optional[str] = None,
    ) -> StorageMetadata:
        """Get file metadata."""
        backend = self._get_backend()

        try:
            return await backend.get_metadata(key=key, bucket=bucket)

        except Exception as e:
            logger.error(f"Failed to get metadata for {key}: {e}")
            raise StorageServiceError(f"Failed to get metadata: {str(e)}")

    async def copy_file(
        self,
        source_key: str,
        dest_key: str,
        source_bucket: Optional[str] = None,
        dest_bucket: Optional[str] = None,
    ) -> StorageMetadata:
        """Copy file to a new location."""
        backend = self._get_backend()

        try:
            result = await backend.copy(
                source_key=source_key,
                dest_key=dest_key,
                source_bucket=source_bucket,
                dest_bucket=dest_bucket,
            )
            logger.info(f"Copied file from {source_key} to {dest_key}")
            return result

        except Exception as e:
            logger.error(f"Failed to copy file from {source_key} to {dest_key}: {e}")
            raise StorageServiceError(f"Copy failed: {str(e)}")

    async def list_files(
        self,
        prefix: str = "",
        bucket: Optional[str] = None,
        max_results: int = 1000,
    ) -> list[StorageMetadata]:
        """List files with optional prefix filter."""
        backend = self._get_backend()

        try:
            results = await backend.list_objects(
                prefix=prefix,
                bucket=bucket,
                max_keys=max_results,
            )
            logger.info(f"Listed {len(results)} files with prefix '{prefix}' from {self._active_provider}")
            return results

        except Exception as e:
            logger.error(f"Failed to list files with prefix {prefix}: {e}")
            raise StorageServiceError(f"List failed: {str(e)}")


# Global storage manager instance
_storage_manager: Optional[StorageManager] = None


def get_storage_manager() -> StorageManager:
    """Get or create global storage manager instance."""
    global _storage_manager
    if _storage_manager is None:
        _storage_manager = StorageManager()
    return _storage_manager


async def initialize_storage() -> StorageManager:
    """Initialize storage manager (for application startup)."""
    global _storage_manager
    _storage_manager = StorageManager()
    logger.info("Storage manager initialized successfully")
    return _storage_manager
