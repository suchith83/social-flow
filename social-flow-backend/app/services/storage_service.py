"""
Storage Service for unified storage operations.

This service integrates all existing storage modules from storage directory
into the FastAPI application.
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional, Union
from pathlib import Path
import sys
import uuid
from datetime import datetime

# Add storage to path
sys.path.append(str(Path(__file__).parent.parent.parent / "storage"))

import boto3
from botocore.exceptions import ClientError
from fastapi import UploadFile

from app.core.config import settings
from app.core.exceptions import StorageServiceError
from app.core.redis import get_cache

logger = logging.getLogger(__name__)


class StorageService:
    """Main storage service for unified storage operations."""
    
    def __init__(self):
        self.s3_client = None
        self.redis_client = None
        self.elasticsearch_client = None
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
            self._init_s3_client()
            
            # Initialize Redis client
            self._init_redis_client()
            
            # Initialize Elasticsearch client
            self._init_elasticsearch_client()
            
            # Initialize storage modules
            self._init_storage_modules()
            
            logger.info("Storage Service initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Storage Service: {e}")
            raise StorageServiceError(f"Storage Service initialization failed: {e}")
    
    def _init_s3_client(self):
        """Initialize S3 client for object storage."""
        try:
            self.s3_client = boto3.client(
                's3',
                aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
                aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
                region_name=settings.AWS_REGION
            )
            logger.info("S3 client initialized")
        except Exception as e:
            logger.warning(f"S3 client initialization failed: {e}")
    
    def _init_redis_client(self):
        """Initialize Redis client for caching."""
        try:
            import redis
            self.redis_client = redis.Redis.from_url(settings.REDIS_URL)
            logger.info("Redis client initialized")
        except Exception as e:
            logger.warning(f"Redis client initialization failed: {e}")
    
    def _init_elasticsearch_client(self):
        """Initialize Elasticsearch client for search."""
        try:
            from elasticsearch import Elasticsearch
            self.elasticsearch_client = Elasticsearch([settings.ELASTICSEARCH_URL])
            logger.info("Elasticsearch client initialized")
        except Exception as e:
            logger.warning(f"Elasticsearch client initialization failed: {e}")
    
    def _init_storage_modules(self):
        """Initialize storage modules."""
        try:
            # S3 storage
            from storage.object_storage.aws_s3.client import S3Client
            self.s3_storage = S3Client()
            
            # Multi-cloud storage
            from storage.object_storage.multi_cloud.manager import MultiCloudManager
            self.multi_cloud_manager = MultiCloudManager()
            
            # Video storage
            from storage.video_storage.raw_uploads.uploader import VideoUploader
            self.video_uploader = VideoUploader()
            
            logger.info("Storage modules initialized")
        except ImportError as e:
            logger.warning(f"Storage modules not available: {e}")
    
    async def upload_file(self, file: UploadFile, bucket: str, key: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Upload file to object storage."""
        try:
            if not self.s3_client:
                raise StorageServiceError("S3 client not initialized")
            
            # Read file content
            content = await file.read()
            
            # Upload to S3
            extra_args = {}
            if metadata:
                extra_args['Metadata'] = metadata
            
            self.s3_client.put_object(
                Bucket=bucket,
                Key=key,
                Body=content,
                ContentType=file.content_type,
                **extra_args
            )
            
            # Generate URL
            url = f"https://{bucket}.s3.{settings.AWS_REGION}.amazonaws.com/{key}"
            
            return {
                'success': True,
                'bucket': bucket,
                'key': key,
                'url': url,
                'size': len(content),
                'content_type': file.content_type
            }
        except ClientError as e:
            logger.error(f"S3 upload failed: {e}")
            raise StorageServiceError(f"File upload failed: {e}")
        except Exception as e:
            logger.error(f"File upload failed: {e}")
            raise StorageServiceError(f"File upload failed: {e}")
    
    async def download_file(self, bucket: str, key: str) -> bytes:
        """Download file from object storage."""
        try:
            if not self.s3_client:
                raise StorageServiceError("S3 client not initialized")
            
            response = self.s3_client.get_object(Bucket=bucket, Key=key)
            return response['Body'].read()
        except ClientError as e:
            logger.error(f"S3 download failed: {e}")
            raise StorageServiceError(f"File download failed: {e}")
        except Exception as e:
            logger.error(f"File download failed: {e}")
            raise StorageServiceError(f"File download failed: {e}")
    
    async def delete_file(self, bucket: str, key: str) -> bool:
        """Delete file from object storage."""
        try:
            if not self.s3_client:
                raise StorageServiceError("S3 client not initialized")
            
            self.s3_client.delete_object(Bucket=bucket, Key=key)
            return True
        except ClientError as e:
            logger.error(f"S3 delete failed: {e}")
            raise StorageServiceError(f"File deletion failed: {e}")
        except Exception as e:
            logger.error(f"File deletion failed: {e}")
            raise StorageServiceError(f"File deletion failed: {e}")
    
    async def generate_presigned_url(self, bucket: str, key: str, operation: str = "get_object", expires_in: int = 3600) -> str:
        """Generate presigned URL for object storage operations."""
        try:
            if not self.s3_client:
                raise StorageServiceError("S3 client not initialized")
            
            url = self.s3_client.generate_presigned_url(
                operation,
                Params={'Bucket': bucket, 'Key': key},
                ExpiresIn=expires_in
            )
            return url
        except ClientError as e:
            logger.error(f"Presigned URL generation failed: {e}")
            raise StorageServiceError(f"Presigned URL generation failed: {e}")
        except Exception as e:
            logger.error(f"Presigned URL generation failed: {e}")
            raise StorageServiceError(f"Presigned URL generation failed: {e}")
    
    async def list_files(self, bucket: str, prefix: str = "", max_keys: int = 1000) -> List[Dict[str, Any]]:
        """List files in object storage."""
        try:
            if not self.s3_client:
                raise StorageServiceError("S3 client not initialized")
            
            response = self.s3_client.list_objects_v2(
                Bucket=bucket,
                Prefix=prefix,
                MaxKeys=max_keys
            )
            
            files = []
            for obj in response.get('Contents', []):
                files.append({
                    'key': obj['Key'],
                    'size': obj['Size'],
                    'last_modified': obj['LastModified'].isoformat(),
                    'etag': obj['ETag']
                })
            
            return files
        except ClientError as e:
            logger.error(f"S3 list failed: {e}")
            raise StorageServiceError(f"File listing failed: {e}")
        except Exception as e:
            logger.error(f"File listing failed: {e}")
            raise StorageServiceError(f"File listing failed: {e}")
    
    async def cache_set(self, key: str, value: Any, expire: Optional[int] = None) -> bool:
        """Set value in cache."""
        try:
            cache = await self._get_cache()
            return await cache.set(key, value, expire)
        except Exception as e:
            logger.error(f"Cache set failed: {e}")
            raise StorageServiceError(f"Cache operation failed: {e}")
    
    async def cache_get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        try:
            cache = await self._get_cache()
            return await cache.get(key)
        except Exception as e:
            logger.error(f"Cache get failed: {e}")
            raise StorageServiceError(f"Cache operation failed: {e}")
    
    async def cache_delete(self, key: str) -> bool:
        """Delete value from cache."""
        try:
            cache = await self._get_cache()
            return await cache.delete(key)
        except Exception as e:
            logger.error(f"Cache delete failed: {e}")
            raise StorageServiceError(f"Cache operation failed: {e}")
    
    async def search_documents(self, index: str, query: Dict[str, Any], size: int = 10) -> List[Dict[str, Any]]:
        """Search documents in Elasticsearch."""
        try:
            if not self.elasticsearch_client:
                raise StorageServiceError("Elasticsearch client not initialized")
            
            response = self.elasticsearch_client.search(
                index=index,
                body=query,
                size=size
            )
            
            documents = []
            for hit in response['hits']['hits']:
                documents.append({
                    'id': hit['_id'],
                    'score': hit['_score'],
                    'source': hit['_source']
                })
            
            return documents
        except Exception as e:
            logger.error(f"Elasticsearch search failed: {e}")
            raise StorageServiceError(f"Search failed: {e}")
    
    async def index_document(self, index: str, document: Dict[str, Any], doc_id: Optional[str] = None) -> str:
        """Index document in Elasticsearch."""
        try:
            if not self.elasticsearch_client:
                raise StorageServiceError("Elasticsearch client not initialized")
            
            response = self.elasticsearch_client.index(
                index=index,
                body=document,
                id=doc_id
            )
            
            return response['_id']
        except Exception as e:
            logger.error(f"Elasticsearch indexing failed: {e}")
            raise StorageServiceError(f"Document indexing failed: {e}")
    
    async def delete_document(self, index: str, doc_id: str) -> bool:
        """Delete document from Elasticsearch."""
        try:
            if not self.elasticsearch_client:
                raise StorageServiceError("Elasticsearch client not initialized")
            
            response = self.elasticsearch_client.delete(
                index=index,
                id=doc_id
            )
            
            return response['result'] == 'deleted'
        except Exception as e:
            logger.error(f"Elasticsearch deletion failed: {e}")
            raise StorageServiceError(f"Document deletion failed: {e}")
    
    async def create_index(self, index: str, mapping: Dict[str, Any]) -> bool:
        """Create Elasticsearch index."""
        try:
            if not self.elasticsearch_client:
                raise StorageServiceError("Elasticsearch client not initialized")
            
            if not self.elasticsearch_client.indices.exists(index=index):
                self.elasticsearch_client.indices.create(
                    index=index,
                    body=mapping
                )
                return True
            return False
        except Exception as e:
            logger.error(f"Elasticsearch index creation failed: {e}")
            raise StorageServiceError(f"Index creation failed: {e}")
    
    async def delete_index(self, index: str) -> bool:
        """Delete Elasticsearch index."""
        try:
            if not self.elasticsearch_client:
                raise StorageServiceError("Elasticsearch client not initialized")
            
            if self.elasticsearch_client.indices.exists(index=index):
                self.elasticsearch_client.indices.delete(index=index)
                return True
            return False
        except Exception as e:
            logger.error(f"Elasticsearch index deletion failed: {e}")
            raise StorageServiceError(f"Index deletion failed: {e}")
    
    async def get_storage_stats(self) -> Dict[str, Any]:
        """Get storage statistics."""
        try:
            stats = {
                's3': await self._get_s3_stats(),
                'redis': await self._get_redis_stats(),
                'elasticsearch': await self._get_elasticsearch_stats()
            }
            return stats
        except Exception as e:
            logger.error(f"Storage stats retrieval failed: {e}")
            raise StorageServiceError(f"Storage stats retrieval failed: {e}")
    
    async def _get_s3_stats(self) -> Dict[str, Any]:
        """Get S3 storage statistics."""
        try:
            if not self.s3_client:
                return {'status': 'not_initialized'}
            
            # Get bucket size and object count
            response = self.s3_client.list_objects_v2(Bucket=settings.AWS_S3_BUCKET)
            
            total_size = sum(obj['Size'] for obj in response.get('Contents', []))
            object_count = len(response.get('Contents', []))
            
            return {
                'status': 'active',
                'bucket': settings.AWS_S3_BUCKET,
                'total_size': total_size,
                'object_count': object_count
            }
        except Exception as e:
            logger.warning(f"S3 stats retrieval failed: {e}")
            return {'status': 'error', 'error': str(e)}
    
    async def _get_redis_stats(self) -> Dict[str, Any]:
        """Get Redis storage statistics."""
        try:
            if not self.redis_client:
                return {'status': 'not_initialized'}
            
            info = self.redis_client.info()
            
            return {
                'status': 'active',
                'used_memory': info.get('used_memory', 0),
                'connected_clients': info.get('connected_clients', 0),
                'total_commands_processed': info.get('total_commands_processed', 0)
            }
        except Exception as e:
            logger.warning(f"Redis stats retrieval failed: {e}")
            return {'status': 'error', 'error': str(e)}
    
    async def _get_elasticsearch_stats(self) -> Dict[str, Any]:
        """Get Elasticsearch storage statistics."""
        try:
            if not self.elasticsearch_client:
                return {'status': 'not_initialized'}
            
            stats = self.elasticsearch_client.indices.stats()
            
            total_docs = sum(index_stats['total']['docs']['count'] for index_stats in stats['indices'].values())
            total_size = sum(index_stats['total']['store']['size_in_bytes'] for index_stats in stats['indices'].values())
            
            return {
                'status': 'active',
                'total_documents': total_docs,
                'total_size': total_size,
                'index_count': len(stats['indices'])
            }
        except Exception as e:
            logger.warning(f"Elasticsearch stats retrieval failed: {e}")
            return {'status': 'error', 'error': str(e)}


# Global storage service instance
storage_service = StorageService()
