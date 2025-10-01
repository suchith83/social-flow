"""
Storage Service for handling all storage operations.

This service integrates all existing storage modules from
the storage directory into the FastAPI application.
"""

import logging
from typing import Any, Dict, List
from datetime import datetime
import uuid
import boto3
from botocore.exceptions import ClientError

from app.core.config import settings
from app.core.exceptions import StorageServiceError
from app.core.redis import get_cache

logger = logging.getLogger(__name__)


class StorageService:
    """Main storage service integrating all storage capabilities."""

    def __init__(self):
        self.s3_client = None
        self.cache = None
        self._initialize_services()

    async def _get_cache(self):
        """Get Redis cache instance."""
        if self.cache is None:
            self.cache = await get_cache()
        return self.cache

    def _initialize_services(self):
        """Initialize storage services."""
        try:
            # Initialize S3 client
            if settings.AWS_ACCESS_KEY_ID and settings.AWS_SECRET_ACCESS_KEY and settings.AWS_REGION:
                self.s3_client = boto3.client(
                    's3',
                    aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
                    aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
                    region_name=settings.AWS_REGION
                )
                logger.info("S3 client initialized successfully")
            else:
                logger.warning("AWS credentials not fully configured. S3 client not initialized.")
            
            # TODO: Initialize other storage backends (Azure Blob, Google Cloud Storage)
            logger.info("Storage Service initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Storage Service: {e}")
            raise StorageServiceError(f"Failed to initialize Storage Service: {e}")

    async def upload_file(self, file_data: bytes, file_path: str, bucket: str = None, 
                         content_type: str = "application/octet-stream") -> Dict[str, Any]:
        """Upload a file to storage."""
        try:
            if not self.s3_client:
                raise StorageServiceError("S3 client not initialized")
            
            if not bucket:
                bucket = settings.S3_BUCKET_NAME
            
            # Generate unique file path if not provided
            if not file_path:
                file_path = f"uploads/{uuid.uuid4()}"
            
            # Upload to S3
            self.s3_client.put_object(
                Bucket=bucket,
                Key=file_path,
                Body=file_data,
                ContentType=content_type
            )
            
            # Generate presigned URL for access
            url = self.s3_client.generate_presigned_url(
                'get_object',
                Params={'Bucket': bucket, 'Key': file_path},
                ExpiresIn=3600  # 1 hour
            )
            
            return {
                "file_path": file_path,
                "bucket": bucket,
                "url": url,
                "size": len(file_data),
                "content_type": content_type,
                "uploaded_at": datetime.utcnow().isoformat()
            }
        except ClientError as e:
            raise StorageServiceError(f"S3 upload failed: {str(e)}")
        except Exception as e:
            raise StorageServiceError(f"Failed to upload file: {str(e)}")

    async def download_file(self, file_path: str, bucket: str = None) -> bytes:
        """Download a file from storage."""
        try:
            if not self.s3_client:
                raise StorageServiceError("S3 client not initialized")
            
            if not bucket:
                bucket = settings.S3_BUCKET_NAME
            
            response = self.s3_client.get_object(Bucket=bucket, Key=file_path)
            return response['Body'].read()
        except ClientError as e:
            raise StorageServiceError(f"S3 download failed: {str(e)}")
        except Exception as e:
            raise StorageServiceError(f"Failed to download file: {str(e)}")

    async def delete_file(self, file_path: str, bucket: str = None) -> Dict[str, Any]:
        """Delete a file from storage."""
        try:
            if not self.s3_client:
                raise StorageServiceError("S3 client not initialized")
            
            if not bucket:
                bucket = settings.S3_BUCKET_NAME
            
            self.s3_client.delete_object(Bucket=bucket, Key=file_path)
            
            return {
                "file_path": file_path,
                "bucket": bucket,
                "deleted_at": datetime.utcnow().isoformat(),
                "status": "deleted"
            }
        except ClientError as e:
            raise StorageServiceError(f"S3 delete failed: {str(e)}")
        except Exception as e:
            raise StorageServiceError(f"Failed to delete file: {str(e)}")

    async def generate_presigned_url(self, file_path: str, bucket: str = None, 
                                   operation: str = "get_object", expires_in: int = 3600) -> str:
        """Generate a presigned URL for file access."""
        try:
            if not self.s3_client:
                raise StorageServiceError("S3 client not initialized")
            
            if not bucket:
                bucket = settings.S3_BUCKET_NAME
            
            url = self.s3_client.generate_presigned_url(
                operation,
                Params={'Bucket': bucket, 'Key': file_path},
                ExpiresIn=expires_in
            )
            
            return url
        except ClientError as e:
            raise StorageServiceError(f"Failed to generate presigned URL: {str(e)}")
        except Exception as e:
            raise StorageServiceError(f"Failed to generate presigned URL: {str(e)}")

    async def list_files(self, prefix: str = "", bucket: str = None, 
                        max_keys: int = 1000) -> List[Dict[str, Any]]:
        """List files in storage."""
        try:
            if not self.s3_client:
                raise StorageServiceError("S3 client not initialized")
            
            if not bucket:
                bucket = settings.S3_BUCKET_NAME
            
            response = self.s3_client.list_objects_v2(
                Bucket=bucket,
                Prefix=prefix,
                MaxKeys=max_keys
            )
            
            files = []
            for obj in response.get('Contents', []):
                files.append({
                    "key": obj['Key'],
                    "size": obj['Size'],
                    "last_modified": obj['LastModified'].isoformat(),
                    "etag": obj['ETag']
                })
            
            return files
        except ClientError as e:
            raise StorageServiceError(f"S3 list failed: {str(e)}")
        except Exception as e:
            raise StorageServiceError(f"Failed to list files: {str(e)}")

    async def copy_file(self, source_path: str, dest_path: str, 
                       source_bucket: str = None, dest_bucket: str = None) -> Dict[str, Any]:
        """Copy a file within storage."""
        try:
            if not self.s3_client:
                raise StorageServiceError("S3 client not initialized")
            
            if not source_bucket:
                source_bucket = settings.S3_BUCKET_NAME
            if not dest_bucket:
                dest_bucket = settings.S3_BUCKET_NAME
            
            copy_source = {'Bucket': source_bucket, 'Key': source_path}
            self.s3_client.copy_object(
                CopySource=copy_source,
                Bucket=dest_bucket,
                Key=dest_path
            )
            
            return {
                "source_path": source_path,
                "dest_path": dest_path,
                "source_bucket": source_bucket,
                "dest_bucket": dest_bucket,
                "copied_at": datetime.utcnow().isoformat(),
                "status": "copied"
            }
        except ClientError as e:
            raise StorageServiceError(f"S3 copy failed: {str(e)}")
        except Exception as e:
            raise StorageServiceError(f"Failed to copy file: {str(e)}")

    async def get_file_metadata(self, file_path: str, bucket: str = None) -> Dict[str, Any]:
        """Get metadata for a file."""
        try:
            if not self.s3_client:
                raise StorageServiceError("S3 client not initialized")
            
            if not bucket:
                bucket = settings.S3_BUCKET_NAME
            
            response = self.s3_client.head_object(Bucket=bucket, Key=file_path)
            
            return {
                "file_path": file_path,
                "bucket": bucket,
                "size": response['ContentLength'],
                "content_type": response.get('ContentType', 'application/octet-stream'),
                "last_modified": response['LastModified'].isoformat(),
                "etag": response['ETag'],
                "metadata": response.get('Metadata', {})
            }
        except ClientError as e:
            raise StorageServiceError(f"S3 metadata fetch failed: {str(e)}")
        except Exception as e:
            raise StorageServiceError(f"Failed to get file metadata: {str(e)}")

    async def create_multipart_upload(self, file_path: str, bucket: str = None, 
                                    content_type: str = "application/octet-stream") -> Dict[str, Any]:
        """Initiate a multipart upload."""
        try:
            if not self.s3_client:
                raise StorageServiceError("S3 client not initialized")
            
            if not bucket:
                bucket = settings.S3_BUCKET_NAME
            
            response = self.s3_client.create_multipart_upload(
                Bucket=bucket,
                Key=file_path,
                ContentType=content_type
            )
            
            return {
                "upload_id": response['UploadId'],
                "file_path": file_path,
                "bucket": bucket,
                "created_at": datetime.utcnow().isoformat()
            }
        except ClientError as e:
            raise StorageServiceError(f"S3 multipart upload creation failed: {str(e)}")
        except Exception as e:
            raise StorageServiceError(f"Failed to create multipart upload: {str(e)}")

    async def upload_part(self, upload_id: str, file_path: str, part_number: int, 
                         part_data: bytes, bucket: str = None) -> Dict[str, Any]:
        """Upload a part of a multipart upload."""
        try:
            if not self.s3_client:
                raise StorageServiceError("S3 client not initialized")
            
            if not bucket:
                bucket = settings.S3_BUCKET_NAME
            
            response = self.s3_client.upload_part(
                Bucket=bucket,
                Key=file_path,
                PartNumber=part_number,
                UploadId=upload_id,
                Body=part_data
            )
            
            return {
                "upload_id": upload_id,
                "part_number": part_number,
                "etag": response['ETag'],
                "uploaded_at": datetime.utcnow().isoformat()
            }
        except ClientError as e:
            raise StorageServiceError(f"S3 part upload failed: {str(e)}")
        except Exception as e:
            raise StorageServiceError(f"Failed to upload part: {str(e)}")

    async def complete_multipart_upload(self, upload_id: str, file_path: str, 
                                       parts: List[Dict[str, Any]], bucket: str = None) -> Dict[str, Any]:
        """Complete a multipart upload."""
        try:
            if not self.s3_client:
                raise StorageServiceError("S3 client not initialized")
            
            if not bucket:
                bucket = settings.S3_BUCKET_NAME
            
            response = self.s3_client.complete_multipart_upload(
                Bucket=bucket,
                Key=file_path,
                UploadId=upload_id,
                MultipartUpload={'Parts': parts}
            )
            
            return {
                "upload_id": upload_id,
                "file_path": file_path,
                "bucket": bucket,
                "location": response['Location'],
                "etag": response['ETag'],
                "completed_at": datetime.utcnow().isoformat()
            }
        except ClientError as e:
            raise StorageServiceError(f"S3 multipart upload completion failed: {str(e)}")
        except Exception as e:
            raise StorageServiceError(f"Failed to complete multipart upload: {str(e)}")

    async def abort_multipart_upload(self, upload_id: str, file_path: str, bucket: str = None) -> Dict[str, Any]:
        """Abort a multipart upload."""
        try:
            if not self.s3_client:
                raise StorageServiceError("S3 client not initialized")
            
            if not bucket:
                bucket = settings.S3_BUCKET_NAME
            
            self.s3_client.abort_multipart_upload(
                Bucket=bucket,
                Key=file_path,
                UploadId=upload_id
            )
            
            return {
                "upload_id": upload_id,
                "file_path": file_path,
                "bucket": bucket,
                "aborted_at": datetime.utcnow().isoformat(),
                "status": "aborted"
            }
        except ClientError as e:
            raise StorageServiceError(f"S3 multipart upload abort failed: {str(e)}")
        except Exception as e:
            raise StorageServiceError(f"Failed to abort multipart upload: {str(e)}")


storage_service = StorageService()
