"""
DEPRECATED: Legacy Storage Service Wrapper

This module wraps the new unified storage infrastructure for backward compatibility.
New code should use app.infrastructure.storage.manager.get_storage_manager() directly.

This wrapper will be removed in Phase 3.
"""

import logging
import warnings
from typing import Any, Dict, List, Optional
from datetime import datetime
import io

from app.infrastructure.storage import get_storage_manager, StorageMetadata
from app.core.exceptions import StorageServiceError

logger = logging.getLogger(__name__)

# Deprecation warning
warnings.warn(
    "app.services.storage_service is deprecated. Use app.infrastructure.storage instead.",
    DeprecationWarning,
    stacklevel=2
)


class StorageService:
    """
    DEPRECATED: Legacy wrapper for unified storage infrastructure.
    
    Use app.infrastructure.storage.manager.get_storage_manager() for new code.
    """

    def __init__(self):
        """Initialize legacy storage service wrapper."""
        logger.warning("StorageService is deprecated. Migrate to StorageManager.")
        self._manager = get_storage_manager()
        self.cache = None

    async def _get_cache(self):
        """Get Redis cache instance."""
        if self.cache is None:
            from app.core.redis import get_cache
            self.cache = await get_cache()
        return self.cache

    async def upload_file(
        self, 
        file_data: bytes, 
        file_path: str, 
        bucket: str = None,
        content_type: str = "application/octet-stream"
    ) -> Dict[str, Any]:
        """
        Upload a file to storage.
        
        DEPRECATED: Use StorageManager.upload_file() instead.
        """
        try:
            file_obj = io.BytesIO(file_data)
            metadata = await self._manager.upload_file(
                file=file_obj,
                key=file_path,
                bucket=bucket,
                content_type=content_type
            )
            
            # Generate presigned URL for backward compatibility
            url = await self._manager.get_public_url(
                key=file_path,
                bucket=bucket,
                expires_in=3600
            )
            
            return {
                "file_path": metadata.key,
                "bucket": metadata.bucket,
                "url": url,
                "size": metadata.size,
                "content_type": metadata.content_type,
                "uploaded_at": metadata.created_at.isoformat() if metadata.created_at else datetime.utcnow().isoformat()
            }
        except Exception as e:
            logger.error(f"Legacy upload_file failed: {e}")
            raise StorageServiceError(f"Failed to upload file: {str(e)}")

    async def download_file(self, file_path: str, bucket: str = None) -> bytes:
        """
        Download a file from storage.
        
        DEPRECATED: Use StorageManager.download_file() instead.
        """
        try:
            return await self._manager.download_file(key=file_path, bucket=bucket)
        except Exception as e:
            logger.error(f"Legacy download_file failed: {e}")
            raise StorageServiceError(f"Failed to download file: {str(e)}")

    async def delete_file(self, file_path: str, bucket: str = None) -> Dict[str, Any]:
        """
        Delete a file from storage.
        
        DEPRECATED: Use StorageManager.delete_file() instead.
        """
        try:
            success = await self._manager.delete_file(key=file_path, bucket=bucket)
            
            return {
                "file_path": file_path,
                "bucket": bucket or "default",
                "deleted_at": datetime.utcnow().isoformat(),
                "status": "deleted" if success else "failed"
            }
        except Exception as e:
            logger.error(f"Legacy delete_file failed: {e}")
            raise StorageServiceError(f"Failed to delete file: {str(e)}")

    async def generate_presigned_url(
        self,
        file_path: str,
        bucket: str = None,
        operation: str = "get_object",
        expires_in: int = 3600
    ) -> str:
        """
        Generate a presigned URL for file access.
        
        DEPRECATED: Use StorageManager.get_public_url() instead.
        """
        try:
            return await self._manager.get_public_url(
                key=file_path,
                bucket=bucket,
                expires_in=expires_in
            )
        except Exception as e:
            logger.error(f"Legacy generate_presigned_url failed: {e}")
            raise StorageServiceError(f"Failed to generate presigned URL: {str(e)}")

    async def list_files(
        self,
        prefix: str = "",
        bucket: str = None,
        max_keys: int = 1000
    ) -> List[Dict[str, Any]]:
        """
        List files in storage.
        
        DEPRECATED: Use StorageManager.list_files() instead.
        """
        try:
            results = await self._manager.list_files(
                prefix=prefix,
                bucket=bucket,
                max_results=max_keys
            )
            
            # Convert to legacy format
            files = []
            for metadata in results:
                files.append({
                    "key": metadata.key,
                    "size": metadata.size,
                    "last_modified": metadata.modified_at.isoformat() if metadata.modified_at else datetime.utcnow().isoformat(),
                    "etag": metadata.etag or ""
                })
            
            return files
        except Exception as e:
            logger.error(f"Legacy list_files failed: {e}")
            raise StorageServiceError(f"Failed to list files: {str(e)}")

    async def copy_file(
        self,
        source_path: str,
        dest_path: str,
        source_bucket: str = None,
        dest_bucket: str = None
    ) -> Dict[str, Any]:
        """
        Copy a file within storage.
        
        DEPRECATED: Use StorageManager.copy_file() instead.
        """
        try:
            metadata = await self._manager.copy_file(
                source_key=source_path,
                dest_key=dest_path,
                source_bucket=source_bucket,
                dest_bucket=dest_bucket
            )
            
            return {
                "source_path": source_path,
                "dest_path": dest_path,
                "source_bucket": source_bucket or "default",
                "dest_bucket": dest_bucket or "default",
                "copied_at": datetime.utcnow().isoformat(),
                "status": "copied"
            }
        except Exception as e:
            logger.error(f"Legacy copy_file failed: {e}")
            raise StorageServiceError(f"Failed to copy file: {str(e)}")

    async def get_file_metadata(self, file_path: str, bucket: str = None) -> Dict[str, Any]:
        """
        Get metadata for a file.
        
        DEPRECATED: Use StorageManager.get_file_metadata() instead.
        """
        try:
            metadata = await self._manager.get_file_metadata(key=file_path, bucket=bucket)
            
            return {
                "file_path": metadata.key,
                "bucket": metadata.bucket,
                "size": metadata.size,
                "content_type": metadata.content_type,
                "last_modified": metadata.modified_at.isoformat() if metadata.modified_at else datetime.utcnow().isoformat(),
                "etag": metadata.etag or "",
                "metadata": metadata.metadata or {}
            }
        except Exception as e:
            logger.error(f"Legacy get_file_metadata failed: {e}")
            raise StorageServiceError(f"Failed to get file metadata: {str(e)}")

    async def create_multipart_upload(
        self,
        file_path: str,
        bucket: str = None,
        content_type: str = "application/octet-stream"
    ) -> Dict[str, Any]:
        """
        Initiate a multipart upload.
        
        DEPRECATED: Use StorageManager.upload_large_file() instead (automatic).
        """
        try:
            backend = self._manager._get_backend()
            from app.infrastructure.storage.base import IMultipartUpload
            
            if not isinstance(backend, IMultipartUpload):
                raise StorageServiceError("Backend does not support multipart upload")
            
            upload_id = await backend.initiate_multipart_upload(
                key=file_path,
                bucket=bucket,
                content_type=content_type
            )
            
            return {
                "upload_id": upload_id,
                "file_path": file_path,
                "bucket": bucket or "default",
                "created_at": datetime.utcnow().isoformat()
            }
        except Exception as e:
            logger.error(f"Legacy create_multipart_upload failed: {e}")
            raise StorageServiceError(f"Failed to create multipart upload: {str(e)}")

    async def upload_part(
        self,
        upload_id: str,
        file_path: str,
        part_number: int,
        part_data: bytes,
        bucket: str = None
    ) -> Dict[str, Any]:
        """
        Upload a part of a multipart upload.
        
        DEPRECATED: Use StorageManager.upload_large_file() instead (automatic).
        """
        try:
            backend = self._manager._get_backend()
            from app.infrastructure.storage.base import IMultipartUpload
            
            if not isinstance(backend, IMultipartUpload):
                raise StorageServiceError("Backend does not support multipart upload")
            
            part_info = await backend.upload_part(
                upload_id=upload_id,
                part_number=part_number,
                data=part_data,
                key=file_path,
                bucket=bucket
            )
            
            return {
                "upload_id": upload_id,
                "part_number": part_number,
                "etag": part_info.get("ETag", ""),
                "uploaded_at": datetime.utcnow().isoformat()
            }
        except Exception as e:
            logger.error(f"Legacy upload_part failed: {e}")
            raise StorageServiceError(f"Failed to upload part: {str(e)}")

    async def complete_multipart_upload(
        self,
        upload_id: str,
        file_path: str,
        parts: List[Dict[str, Any]],
        bucket: str = None
    ) -> Dict[str, Any]:
        """
        Complete a multipart upload.
        
        DEPRECATED: Use StorageManager.upload_large_file() instead (automatic).
        """
        try:
            backend = self._manager._get_backend()
            from app.infrastructure.storage.base import IMultipartUpload
            
            if not isinstance(backend, IMultipartUpload):
                raise StorageServiceError("Backend does not support multipart upload")
            
            metadata = await backend.complete_multipart_upload(
                upload_id=upload_id,
                key=file_path,
                parts=parts,
                bucket=bucket
            )
            
            return {
                "upload_id": upload_id,
                "file_path": metadata.key,
                "bucket": metadata.bucket,
                "location": f"s3://{metadata.bucket}/{metadata.key}",
                "etag": metadata.etag or "",
                "completed_at": datetime.utcnow().isoformat()
            }
        except Exception as e:
            logger.error(f"Legacy complete_multipart_upload failed: {e}")
            raise StorageServiceError(f"Failed to complete multipart upload: {str(e)}")

    async def abort_multipart_upload(
        self,
        upload_id: str,
        file_path: str,
        bucket: str = None
    ) -> Dict[str, Any]:
        """
        Abort a multipart upload.
        
        DEPRECATED: Use StorageManager (automatic cleanup on error).
        """
        try:
            backend = self._manager._get_backend()
            from app.infrastructure.storage.base import IMultipartUpload
            
            if not isinstance(backend, IMultipartUpload):
                raise StorageServiceError("Backend does not support multipart upload")
            
            success = await backend.abort_multipart_upload(
                upload_id=upload_id,
                key=file_path,
                bucket=bucket
            )
            
            return {
                "upload_id": upload_id,
                "file_path": file_path,
                "bucket": bucket or "default",
                "aborted_at": datetime.utcnow().isoformat(),
                "status": "aborted" if success else "failed"
            }
        except Exception as e:
            logger.error(f"Legacy abort_multipart_upload failed: {e}")
            raise StorageServiceError(f"Failed to abort multipart upload: {str(e)}")


# Legacy global instance for backward compatibility
storage_service = StorageService()
