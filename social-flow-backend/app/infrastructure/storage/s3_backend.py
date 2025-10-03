"""
AWS S3 Storage Backend Implementation.
"""

import logging
from typing import Optional, Dict, Any, BinaryIO
from datetime import datetime

import aioboto3
from botocore.exceptions import ClientError

from app.core.config import settings
from app.infrastructure.storage.base import (
    IStorageBackend,
    IMultipartUpload,
    StorageMetadata,
    StorageProvider,
)
from app.core.exceptions import StorageServiceError

logger = logging.getLogger(__name__)


class S3Backend(IStorageBackend, IMultipartUpload):
    """AWS S3 storage backend implementation."""

    def __init__(
        self,
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
        region_name: Optional[str] = None,
        default_bucket: Optional[str] = None,
    ):
        self.aws_access_key_id = aws_access_key_id or settings.AWS_ACCESS_KEY_ID
        self.aws_secret_access_key = aws_secret_access_key or settings.AWS_SECRET_ACCESS_KEY
        self.region_name = region_name or settings.AWS_REGION
        self.default_bucket = default_bucket or settings.S3_BUCKET_NAME

        if not all([self.aws_access_key_id, self.aws_secret_access_key, self.region_name]):
            raise StorageServiceError("AWS credentials not properly configured")

        self.session = aioboto3.Session(
            aws_access_key_id=self.aws_access_key_id,
            aws_secret_access_key=self.aws_secret_access_key,
            region_name=self.region_name,
        )

    def _get_bucket(self, bucket: Optional[str] = None) -> str:
        """Get bucket name, using default if not provided."""
        return bucket or self.default_bucket

    async def upload(
        self,
        data: bytes,
        key: str,
        bucket: Optional[str] = None,
        content_type: str = "application/octet-stream",
        metadata: Optional[Dict[str, str]] = None,
    ) -> StorageMetadata:
        """Upload data to S3."""
        bucket = self._get_bucket(bucket)
        
        try:
            async with self.session.client("s3") as s3:
                extra_args = {
                    "ContentType": content_type,
                }
                if metadata:
                    extra_args["Metadata"] = metadata

                await s3.put_object(
                    Bucket=bucket,
                    Key=key,
                    Body=data,
                    **extra_args
                )

                # Get object metadata
                response = await s3.head_object(Bucket=bucket, Key=key)

                return StorageMetadata(
                    key=key,
                    bucket=bucket,
                    size=response["ContentLength"],
                    content_type=response.get("ContentType", content_type),
                    provider=StorageProvider.S3,
                    etag=response.get("ETag"),
                    created_at=response.get("LastModified"),
                    modified_at=response.get("LastModified"),
                    metadata=response.get("Metadata"),
                )

        except ClientError as e:
            logger.error(f"S3 upload failed for key {key}: {e}")
            raise StorageServiceError(f"Failed to upload to S3: {str(e)}")

    async def download(
        self,
        key: str,
        bucket: Optional[str] = None,
    ) -> bytes:
        """Download data from S3."""
        bucket = self._get_bucket(bucket)

        try:
            async with self.session.client("s3") as s3:
                response = await s3.get_object(Bucket=bucket, Key=key)
                async with response["Body"] as stream:
                    return await stream.read()

        except ClientError as e:
            logger.error(f"S3 download failed for key {key}: {e}")
            raise StorageServiceError(f"Failed to download from S3: {str(e)}")

    async def delete(
        self,
        key: str,
        bucket: Optional[str] = None,
    ) -> bool:
        """Delete object from S3."""
        bucket = self._get_bucket(bucket)

        try:
            async with self.session.client("s3") as s3:
                await s3.delete_object(Bucket=bucket, Key=key)
                return True

        except ClientError as e:
            logger.error(f"S3 delete failed for key {key}: {e}")
            raise StorageServiceError(f"Failed to delete from S3: {str(e)}")

    async def exists(
        self,
        key: str,
        bucket: Optional[str] = None,
    ) -> bool:
        """Check if object exists in S3."""
        bucket = self._get_bucket(bucket)

        try:
            async with self.session.client("s3") as s3:
                await s3.head_object(Bucket=bucket, Key=key)
                return True

        except ClientError as e:
            if e.response["Error"]["Code"] == "404":
                return False
            raise StorageServiceError(f"Failed to check existence in S3: {str(e)}")

    async def get_metadata(
        self,
        key: str,
        bucket: Optional[str] = None,
    ) -> StorageMetadata:
        """Get object metadata from S3."""
        bucket = self._get_bucket(bucket)

        try:
            async with self.session.client("s3") as s3:
                response = await s3.head_object(Bucket=bucket, Key=key)

                return StorageMetadata(
                    key=key,
                    bucket=bucket,
                    size=response["ContentLength"],
                    content_type=response.get("ContentType", "application/octet-stream"),
                    provider=StorageProvider.S3,
                    etag=response.get("ETag"),
                    created_at=response.get("LastModified"),
                    modified_at=response.get("LastModified"),
                    metadata=response.get("Metadata"),
                )

        except ClientError as e:
            logger.error(f"Failed to get S3 metadata for key {key}: {e}")
            raise StorageServiceError(f"Failed to get metadata from S3: {str(e)}")

    async def generate_presigned_url(
        self,
        key: str,
        bucket: Optional[str] = None,
        expires_in: int = 3600,
        operation: str = "get",
    ) -> str:
        """Generate presigned URL for S3 object."""
        bucket = self._get_bucket(bucket)
        
        operation_map = {
            "get": "get_object",
            "put": "put_object",
            "delete": "delete_object",
        }
        client_method = operation_map.get(operation, "get_object")

        try:
            async with self.session.client("s3") as s3:
                url = await s3.generate_presigned_url(
                    ClientMethod=client_method,
                    Params={"Bucket": bucket, "Key": key},
                    ExpiresIn=expires_in,
                )
                return url

        except ClientError as e:
            logger.error(f"Failed to generate presigned URL for key {key}: {e}")
            raise StorageServiceError(f"Failed to generate presigned URL: {str(e)}")

    async def copy(
        self,
        source_key: str,
        dest_key: str,
        source_bucket: Optional[str] = None,
        dest_bucket: Optional[str] = None,
    ) -> StorageMetadata:
        """Copy object within S3."""
        source_bucket = self._get_bucket(source_bucket)
        dest_bucket = self._get_bucket(dest_bucket)

        try:
            async with self.session.client("s3") as s3:
                copy_source = {"Bucket": source_bucket, "Key": source_key}
                await s3.copy_object(
                    CopySource=copy_source,
                    Bucket=dest_bucket,
                    Key=dest_key,
                )

                return await self.get_metadata(dest_key, dest_bucket)

        except ClientError as e:
            logger.error(f"S3 copy failed from {source_key} to {dest_key}: {e}")
            raise StorageServiceError(f"Failed to copy in S3: {str(e)}")

    async def list_objects(
        self,
        prefix: str = "",
        bucket: Optional[str] = None,
        max_keys: int = 1000,
    ) -> list[StorageMetadata]:
        """List objects in S3 bucket."""
        bucket = self._get_bucket(bucket)

        try:
            async with self.session.client("s3") as s3:
                response = await s3.list_objects_v2(
                    Bucket=bucket,
                    Prefix=prefix,
                    MaxKeys=max_keys,
                )

                objects = []
                for obj in response.get("Contents", []):
                    objects.append(
                        StorageMetadata(
                            key=obj["Key"],
                            bucket=bucket,
                            size=obj["Size"],
                            content_type="application/octet-stream",  # Not available in list
                            provider=StorageProvider.S3,
                            etag=obj.get("ETag"),
                            modified_at=obj.get("LastModified"),
                        )
                    )

                return objects

        except ClientError as e:
            logger.error(f"Failed to list S3 objects with prefix {prefix}: {e}")
            raise StorageServiceError(f"Failed to list S3 objects: {str(e)}")

    # Multipart Upload Implementation

    async def initiate_multipart_upload(
        self,
        key: str,
        bucket: Optional[str] = None,
        content_type: str = "application/octet-stream",
    ) -> str:
        """Initiate multipart upload."""
        bucket = self._get_bucket(bucket)

        try:
            async with self.session.client("s3") as s3:
                response = await s3.create_multipart_upload(
                    Bucket=bucket,
                    Key=key,
                    ContentType=content_type,
                )
                return response["UploadId"]

        except ClientError as e:
            logger.error(f"Failed to initiate multipart upload for key {key}: {e}")
            raise StorageServiceError(f"Failed to initiate multipart upload: {str(e)}")

    async def upload_part(
        self,
        upload_id: str,
        part_number: int,
        data: bytes,
        key: str,
        bucket: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Upload a single part."""
        bucket = self._get_bucket(bucket)

        try:
            async with self.session.client("s3") as s3:
                response = await s3.upload_part(
                    Bucket=bucket,
                    Key=key,
                    UploadId=upload_id,
                    PartNumber=part_number,
                    Body=data,
                )

                return {
                    "PartNumber": part_number,
                    "ETag": response["ETag"],
                }

        except ClientError as e:
            logger.error(f"Failed to upload part {part_number} for key {key}: {e}")
            raise StorageServiceError(f"Failed to upload part: {str(e)}")

    async def complete_multipart_upload(
        self,
        upload_id: str,
        key: str,
        parts: list[Dict[str, Any]],
        bucket: Optional[str] = None,
    ) -> StorageMetadata:
        """Complete multipart upload."""
        bucket = self._get_bucket(bucket)

        try:
            async with self.session.client("s3") as s3:
                await s3.complete_multipart_upload(
                    Bucket=bucket,
                    Key=key,
                    UploadId=upload_id,
                    MultipartUpload={"Parts": parts},
                )

                return await self.get_metadata(key, bucket)

        except ClientError as e:
            logger.error(f"Failed to complete multipart upload for key {key}: {e}")
            raise StorageServiceError(f"Failed to complete multipart upload: {str(e)}")

    async def abort_multipart_upload(
        self,
        upload_id: str,
        key: str,
        bucket: Optional[str] = None,
    ) -> bool:
        """Abort multipart upload."""
        bucket = self._get_bucket(bucket)

        try:
            async with self.session.client("s3") as s3:
                await s3.abort_multipart_upload(
                    Bucket=bucket,
                    Key=key,
                    UploadId=upload_id,
                )
                return True

        except ClientError as e:
            logger.error(f"Failed to abort multipart upload for key {key}: {e}")
            raise StorageServiceError(f"Failed to abort multipart upload: {str(e)}")
