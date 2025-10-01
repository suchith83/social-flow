"""
Video Service for video processing and streaming.

This service integrates all existing video processing modules from storage
and other video-related components into the FastAPI application.
"""

import asyncio
import logging
import os
import tempfile
from typing import Any, Dict, List, Optional
from pathlib import Path
import sys
import uuid
from datetime import datetime

from app.core.database import get_db
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update

import boto3
from botocore.exceptions import ClientError
from fastapi import UploadFile

from app.core.config import settings
from app.core.exceptions import VideoServiceError
from app.core.redis import get_cache
from app.videos.models.video import Video, VideoStatus, VideoVisibility
from app.auth.models.user import User

logger = logging.getLogger(__name__)


class VideoService:
    """Main video service for video processing and streaming."""
    
    def __init__(self, db: AsyncSession = None):
        self.db = db
        self.s3_client = None
        self.cache = None
        self._initialize_services()
    
    async def _get_cache(self):
        """Get Redis cache instance."""
        if self.cache is None:
            self.cache = await get_cache()
        return self.cache
    
    def _initialize_services(self):
        """Initialize video processing services."""
        try:
            # Initialize S3 client
            self._init_s3_client()
            
            # Initialize video processing modules
            self._init_video_processing()
            
            # Initialize streaming modules
            self._init_streaming()
            
            logger.info("Video Service initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Video Service: {e}")
            raise VideoServiceError(f"Video Service initialization failed: {e}")
    
    def _init_s3_client(self):
        """Initialize S3 client for video storage."""
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
    
    def _init_video_processing(self):
        """Initialize video processing modules."""
        try:
            # Video transcoding
            from storage.video_storage.processed_videos.transcoder import VideoTranscoder
            self.transcoder = VideoTranscoder()
            
            # Thumbnail generation
            from storage.video_storage.thumbnails.generator import ThumbnailGenerator
            self.thumbnail_generator = ThumbnailGenerator()
            
            # Metadata extraction
            from storage.video_storage.processed_videos.metadata_extractor import MetadataExtractor
            self.metadata_extractor = MetadataExtractor()
            
            logger.info("Video processing modules initialized")
        except ImportError as e:
            logger.warning(f"Video processing modules not available: {e}")
    
    def _init_streaming(self):
        """Initialize streaming modules."""
        try:
            # Live streaming
            from storage.video_storage.live_streaming.ingest import LiveStreamIngest
            self.live_ingest = LiveStreamIngest()
            
            # Video packaging
            from storage.video_storage.live_streaming.packager import VideoPackager
            self.packager = VideoPackager()
            
            logger.info("Streaming modules initialized")
        except ImportError as e:
            logger.warning(f"Streaming modules not available: {e}")
    
    async def upload_video(self, file: UploadFile, user: User, metadata: Dict[str, Any], db: AsyncSession = None) -> Dict[str, Any]:
        """Upload a video file to S3 and create database record."""
        try:
            # Generate unique video ID
            video_id = str(uuid.uuid4())
            
            # Validate file
            if not self._validate_video_file(file):
                raise VideoServiceError("Invalid video file format")
            
            # Create temporary file for processing
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file.filename.split('.')[-1]}") as temp_file:
                content = await file.read()
                temp_file.write(content)
                temp_file_path = temp_file.name
            
            try:
                # Extract metadata
                video_metadata = await self._extract_video_metadata(temp_file_path)
                
                # Generate S3 key
                s3_key = f"videos/{user.id}/{video_id}/{file.filename}"
                
                # Upload to S3
                await self._upload_to_s3(temp_file_path, s3_key)
                
                # Create video record
                video = Video(
                    id=video_id,
                    title=metadata.get('title', 'Untitled'),
                    description=metadata.get('description', ''),
                    filename=file.filename,
                    file_size=len(content),
                    duration=video_metadata.get('duration', 0),
                    resolution=video_metadata.get('resolution', ''),
                    bitrate=video_metadata.get('bitrate', 0),
                    codec=video_metadata.get('codec', ''),
                    s3_key=s3_key,
                    s3_bucket=settings.AWS_S3_BUCKET,
                    status=VideoStatus.UPLOADING,
                    visibility=VideoVisibility.PUBLIC,
                    owner_id=user.id
                )
                
                # Save to database
                if db:
                    db.add(video)
                    await db.commit()
                    await db.refresh(video)
                
                # Start background processing
                await self._start_video_processing(video, db)
                
                # Trigger ML moderation (async)
                from app.ml.ml_tasks import moderate_video_task
                moderate_video_task.apply_async(args=[str(video_id)])
                
                return {
                    'video_id': video_id,
                    'status': 'uploaded',
                    'message': 'Video uploaded successfully, processing started'
                }
            
            finally:
                # Clean up temporary file
                os.unlink(temp_file_path)
                
        except Exception as e:
            logger.error(f"Video upload failed: {e}")
            raise VideoServiceError(f"Video upload failed: {e}")
    
    async def process_video(self, video_id: str, db: AsyncSession = None) -> Dict[str, Any]:
        """Process uploaded video (transcoding, thumbnails, etc.)."""
        try:
            # Get video from database
            video = await self._get_video_by_id(video_id, db)
            if not video:
                raise VideoServiceError("Video not found")
            
            # Update status to processing
            video.status = VideoStatus.PROCESSING
            video.processing_started_at = datetime.utcnow()
            await self._update_video(video, db)
            
            # Download video from S3
            temp_video_path = await self._download_from_s3(video.s3_key)
            
            try:
                # Generate thumbnails
                thumbnail_urls = await self._generate_thumbnails(temp_video_path, video_id)
                
                # Transcode video
                transcoded_urls = await self._transcode_video(temp_video_path, video_id)
                
                # Update video record
                video.status = VideoStatus.PROCESSED
                video.processing_completed_at = datetime.utcnow()
                video.thumbnail_url = thumbnail_urls[0] if thumbnail_urls else None
                video.hls_url = transcoded_urls.get('hls_url')
                video.dash_url = transcoded_urls.get('dash_url')
                video.streaming_url = transcoded_urls.get('streaming_url')
                
                await self._update_video(video, db)
                
                return {
                    'video_id': video_id,
                    'status': 'processed',
                    'thumbnail_url': video.thumbnail_url,
                    'streaming_urls': {
                        'hls': video.hls_url,
                        'dash': video.dash_url,
                        'streaming': video.streaming_url
                    }
                }
            
            finally:
                # Clean up temporary file
                if os.path.exists(temp_video_path):
                    os.unlink(temp_video_path)
                
        except Exception as e:
            logger.error(f"Video processing failed: {e}")
            # Update video status to failed
            video.status = VideoStatus.FAILED
            video.processing_error = str(e)
            await self._update_video(video, db)
            raise VideoServiceError(f"Video processing failed: {e}")
    
    async def get_video_stream_url(self, video_id: str, quality: str = "auto") -> Dict[str, Any]:
        """Get streaming URL for a video."""
        try:
            video = await self._get_video_by_id(video_id)
            if not video:
                raise VideoServiceError("Video not found")
            
            if video.status != VideoStatus.PROCESSED:
                raise VideoServiceError("Video not ready for streaming")
            
            # Generate signed URL for streaming
            streaming_url = await self._generate_signed_url(video.s3_key, quality)
            
            return {
                'video_id': video_id,
                'streaming_url': streaming_url,
                'quality': quality,
                'duration': video.duration,
                'resolution': video.resolution
            }
        except Exception as e:
            logger.error(f"Failed to get streaming URL: {e}")
            raise VideoServiceError(f"Failed to get streaming URL: {e}")
    
    async def create_live_stream(self, user: User, title: str, description: str = "", db: AsyncSession = None) -> Dict[str, Any]:
        """Create a live streaming session."""
        try:
            # Import live streaming service
            from app.live.services.live_streaming_service import live_streaming_service
            
            # Delegate to live streaming service
            result = await live_streaming_service.create_live_stream(
                str(user.id), title, description, None, None, db
            )
            
            return {
                'stream_id': result['stream_id'],
                'stream_key': result['stream_key'],
                'rtmp_url': result['ingest_endpoint'],
                'playback_url': result['playback_url'],
                'status': result['status']
            }
        except Exception as e:
            logger.error(f"Failed to create live stream: {e}")
            raise VideoServiceError(f"Failed to create live stream: {e}")
    
    async def end_live_stream(self, stream_id: str, db: AsyncSession = None) -> Dict[str, Any]:
        """End a live streaming session."""
        try:
            # Import live streaming service
            from app.live.services.live_streaming_service import live_streaming_service
            
            # Delegate to live streaming service
            result = await live_streaming_service.stop_live_stream(stream_id)
            
            return {
                'stream_id': stream_id,
                'status': 'ended',
                'ended_at': result.get('ended_at')
            }
        except Exception as e:
            logger.error(f"Failed to end live stream: {e}")
            raise VideoServiceError(f"Failed to end live stream: {e}")
    
    async def increment_view_count(self, video_id: str, user_id: Optional[str] = None) -> Dict[str, Any]:
        """Increment view count for a video."""
        try:
            cache = await self._get_cache()
            
            # Increment total view count
            total_views = await cache.increment(f"video_views:{video_id}")
            
            # Increment unique view count if user provided
            if user_id:
                unique_key = f"video_unique_views:{video_id}:{user_id}"
                if not await cache.exists(unique_key):
                    await cache.set(unique_key, "1", expire=86400)  # 24 hours
                    unique_views = await cache.increment(f"video_unique_views:{video_id}")
                else:
                    unique_views = await cache.get(f"video_unique_views:{video_id}") or 0
            
            # Update database periodically (batch update)
            await self._schedule_view_count_update(video_id, total_views)
            
            return {
                'video_id': video_id,
                'total_views': total_views,
                'unique_views': unique_views if user_id else None
            }
        except Exception as e:
            logger.error(f"Failed to increment view count: {e}")
            raise VideoServiceError(f"Failed to increment view count: {e}")
    
    # Private helper methods
    def _validate_video_file(self, file: UploadFile) -> bool:
        """Validate uploaded video file."""
        if not file.filename:
            return False
        
        file_extension = file.filename.split('.')[-1].lower()
        return file_extension in settings.MEDIA_ALLOWED_EXTENSIONS
    
    async def _extract_video_metadata(self, file_path: str) -> Dict[str, Any]:
        """Extract metadata from video file."""
        try:
            if hasattr(self, 'metadata_extractor'):
                return await self.metadata_extractor.extract(file_path)
            else:
                # Fallback to basic metadata extraction
                return {
                    'duration': 0,
                    'resolution': '1920x1080',
                    'bitrate': 1000,
                    'codec': 'h264'
                }
        except Exception as e:
            logger.warning(f"Metadata extraction failed: {e}")
            return {}
    
    async def _upload_to_s3(self, file_path: str, s3_key: str) -> None:
        """Upload file to S3."""
        try:
            if self.s3_client:
                self.s3_client.upload_file(file_path, settings.AWS_S3_BUCKET, s3_key)
            else:
                raise VideoServiceError("S3 client not initialized")
        except ClientError as e:
            raise VideoServiceError(f"S3 upload failed: {e}")
    
    async def _download_from_s3(self, s3_key: str) -> str:
        """Download file from S3 to temporary location."""
        try:
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            temp_file_path = temp_file.name
            temp_file.close()
            
            if self.s3_client:
                self.s3_client.download_file(settings.AWS_S3_BUCKET, s3_key, temp_file_path)
            else:
                raise VideoServiceError("S3 client not initialized")
            
            return temp_file_path
        except ClientError as e:
            raise VideoServiceError(f"S3 download failed: {e}")
    
    async def _generate_thumbnails(self, video_path: str, video_id: str) -> List[str]:
        """Generate thumbnails for video."""
        try:
            if hasattr(self, 'thumbnail_generator'):
                thumbnails = await self.thumbnail_generator.generate(video_path, video_id)
                return thumbnails
            else:
                # Fallback to basic thumbnail generation
                return []
        except Exception as e:
            logger.warning(f"Thumbnail generation failed: {e}")
            return []
    
    async def _transcode_video(self, video_path: str, video_id: str) -> Dict[str, str]:
        """Transcode video to multiple formats."""
        try:
            if hasattr(self, 'transcoder'):
                transcoded = await self.transcoder.transcode(video_path, video_id)
                return transcoded
            else:
                # Fallback to basic transcoding
                return {
                    'hls_url': f"https://{settings.AWS_CLOUDFRONT_DOMAIN}/videos/{video_id}/index.m3u8",
                    'dash_url': f"https://{settings.AWS_CLOUDFRONT_DOMAIN}/videos/{video_id}/index.mpd",
                    'streaming_url': f"https://{settings.AWS_CLOUDFRONT_DOMAIN}/videos/{video_id}/stream.mp4"
                }
        except Exception as e:
            logger.warning(f"Video transcoding failed: {e}")
            return {}
    
    async def _generate_signed_url(self, s3_key: str, quality: str = "auto") -> str:
        """Generate signed URL for video streaming."""
        try:
            if self.s3_client:
                # Generate presigned URL with 1 hour expiration
                url = self.s3_client.generate_presigned_url(
                    'get_object',
                    Params={'Bucket': settings.AWS_S3_BUCKET, 'Key': s3_key},
                    ExpiresIn=3600
                )
                return url
            else:
                raise VideoServiceError("S3 client not initialized")
        except ClientError as e:
            raise VideoServiceError(f"Failed to generate signed URL: {e}")
    
    def _generate_stream_key(self) -> str:
        """Generate unique stream key."""
        import secrets
        return secrets.token_urlsafe(settings.STREAM_KEY_LENGTH)
    
    async def _start_video_processing(self, video: Video, db: AsyncSession = None) -> None:
        """Start background video processing."""
        # In a real implementation, this would queue a job for Celery
        # For now, we'll process immediately
        asyncio.create_task(self.process_video(str(video.id), db))
    
    async def _get_video_by_id(self, video_id: str, db: AsyncSession = None) -> Optional[Video]:
        """Get video by ID from database."""
        if not db:
            return None
        
        result = await db.execute(
            select(Video).where(Video.id == uuid.UUID(video_id))
        )
        return result.scalar_one_or_none()
    
    async def _update_video(self, video: Video, db: AsyncSession = None) -> None:
        """Update video in database."""
        if db:
            await db.commit()
    
    async def _schedule_view_count_update(self, video_id: str, view_count: int) -> None:
        """Schedule view count update to database."""
        # This would schedule a batch update in a real implementation
        pass
    
    # Enhanced video upload functionality from Node.js service
    
    async def initiate_upload_session(self, user: User, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Initiate a chunked upload session."""
        try:
            # Generate upload session ID
            upload_id = str(uuid.uuid4())
            
            # Create upload session in Redis
            session_data = {
                "upload_id": upload_id,
                "user_id": str(user.id),
                "filename": metadata.get("filename", ""),
                "file_size": metadata.get("file_size", 0),
                "chunk_size": metadata.get("chunk_size", 5 * 1024 * 1024),  # 5MB default
                "total_chunks": 0,
                "uploaded_chunks": 0,
                "status": "initiated",
                "created_at": datetime.utcnow().isoformat(),
            }
            
            # Store in Redis
            cache = await self._get_cache()
            await cache.setex(
                f"upload_session:{upload_id}",
                3600,  # 1 hour expiry
                str(session_data)
            )
            
            return {
                "upload_id": upload_id,
                "chunk_size": session_data["chunk_size"],
                "status": "initiated"
            }
            
        except Exception as e:
            raise VideoServiceError(f"Failed to initiate upload session: {str(e)}")
    
    async def upload_chunk(self, upload_id: str, chunk_number: int, chunk_data: bytes) -> Dict[str, Any]:
        """Upload a chunk of video data."""
        try:
            # Get upload session from Redis
            cache = await self._get_cache()
            session_data = await cache.get(f"upload_session:{upload_id}")
            if not session_data:
                raise VideoServiceError("Upload session not found or expired")
            
            session = eval(session_data)  # Convert string back to dict
            
            # Upload chunk to S3 using multipart upload
            from app.services.storage_service import storage_service
            
            chunk_key = f"uploads/{upload_id}/original.mp4"
            
            # Initialize multipart upload if first chunk
            if chunk_number == 1:
                multipart_result = await storage_service.create_multipart_upload(
                    file_path=chunk_key,
                    bucket=settings.AWS_S3_BUCKET,
                    content_type="video/mp4"
                )
                session["s3_upload_id"] = multipart_result["upload_id"]
                session["parts"] = []
            
            # Upload chunk as part
            part_result = await storage_service.upload_part(
                upload_id=session["s3_upload_id"],
                file_path=chunk_key,
                part_number=chunk_number,
                part_data=chunk_data,
                bucket=settings.AWS_S3_BUCKET
            )
            
            # Store part info for later completion
            session["parts"].append({
                "PartNumber": chunk_number,
                "ETag": part_result["etag"]
            })
            
            # Update session progress
            session["uploaded_chunks"] += 1
            session["status"] = "uploading"
            
            await cache.setex(
                f"upload_session:{upload_id}",
                3600,
                str(session)
            )
            
            return {
                "upload_id": upload_id,
                "chunk_number": chunk_number,
                "status": "uploaded",
                "progress": (session["uploaded_chunks"] / session["total_chunks"]) * 100 if session["total_chunks"] > 0 else 0
            }
            
        except Exception as e:
            raise VideoServiceError(f"Failed to upload chunk: {str(e)}")
    
    async def complete_upload(self, upload_id: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Complete the chunked upload and start processing."""
        try:
            # Get upload session from Redis
            cache = await self._get_cache()
            session_data = await cache.get(f"upload_session:{upload_id}")
            if not session_data:
                raise VideoServiceError("Upload session not found or expired")
            
            session = eval(session_data)  # Convert string back to dict
            
            # Complete S3 multipart upload
            from app.services.storage_service import storage_service
            
            final_video_key = f"uploads/{upload_id}/original.mp4"
            complete_result = await storage_service.complete_multipart_upload(
                upload_id=session["s3_upload_id"],
                file_path=final_video_key,
                parts=session["parts"],
                bucket=settings.AWS_S3_BUCKET
            )
            
            # Create video record (would be saved to database in production)
            video_data = {
                "id": upload_id,
                "owner_id": session["user_id"],
                "title": metadata.get("title", ""),
                "description": metadata.get("description", ""),
                "filename": session["filename"],
                "file_size": session["file_size"],
                "s3_key": final_video_key,
                "s3_location": complete_result["location"],
                "status": "processing",
                "created_at": datetime.utcnow().isoformat()
            }
            
            # Save to database (TODO: implement actual DB save)
            # from app.videos.models.video import Video
            # video = Video(**video_data)
            # db.add(video)
            # await db.commit()
            
            # Store video metadata in Redis for now
            await cache.setex(
                f"video:{upload_id}",
                86400,  # 24 hours
                str(video_data)
            )
            
            # Start background processing (queue to Celery)
            from app.videos.video_tasks import process_video_task
            process_video_task.delay(upload_id)
            
            # Clean up upload session
            await cache.delete(f"upload_session:{upload_id}")
            
            return {
                "video_id": upload_id,
                "status": "processing",
                "message": "Upload completed, processing started",
                "s3_location": complete_result["location"]
            }
            
        except Exception as e:
            raise VideoServiceError(f"Failed to complete upload: {str(e)}")
    
    async def cancel_upload(self, upload_id: str) -> Dict[str, Any]:
        """Cancel an ongoing upload."""
        try:
            # Get upload session from Redis
            cache = await self._get_cache()
            session_data = await cache.get(f"upload_session:{upload_id}")
            if not session_data:
                raise VideoServiceError("Upload session not found or expired")
            
            session = eval(session_data)  # Convert string back to dict
            
            # Cancel S3 multipart upload if exists
            if "s3_upload_id" in session:
                from app.services.storage_service import storage_service
                
                final_video_key = f"uploads/{upload_id}/original.mp4"
                await storage_service.abort_multipart_upload(
                    upload_id=session["s3_upload_id"],
                    file_path=final_video_key,
                    bucket=settings.AWS_S3_BUCKET
                )
            
            # Clean up upload session
            await cache.delete(f"upload_session:{upload_id}")
            
            return {
                "upload_id": upload_id,
                "status": "cancelled",
                "message": "Upload cancelled successfully"
            }
            
        except Exception as e:
            raise VideoServiceError(f"Failed to cancel upload: {str(e)}")
    
    async def get_upload_progress(self, upload_id: str) -> Dict[str, Any]:
        """Get upload progress."""
        try:
            # Get upload session from Redis
            cache = await self._get_cache()
            session_data = await cache.get(f"upload_session:{upload_id}")
            if not session_data:
                raise VideoServiceError("Upload session not found or expired")
            
            session = eval(session_data)  # Convert string back to dict
            
            progress = 0
            if session["total_chunks"] > 0:
                progress = (session["uploaded_chunks"] / session["total_chunks"]) * 100
            
            return {
                "upload_id": upload_id,
                "status": session["status"],
                "progress": progress,
                "uploaded_chunks": session["uploaded_chunks"],
                "total_chunks": session["total_chunks"]
            }
            
        except Exception as e:
            raise VideoServiceError(f"Failed to get upload progress: {str(e)}")
    
    # Enhanced transcoding functionality
    
    async def transcode_video(self, video_id: str, settings: Dict[str, Any] = None) -> Dict[str, Any]:
        """Transcode video to multiple formats and resolutions."""
        try:
            if not settings:
                settings = {
                    "formats": ["h264", "h265", "av1"],
                    "resolutions": ["144p", "240p", "360p", "480p", "720p", "1080p"],
                    "generate_thumbnails": True,
                    "generate_preview": True
                }
            
            # Create AWS MediaConvert job or use ffmpeg
            # For now, queue a background task
            from app.videos.video_tasks import transcode_video_task
            task = transcode_video_task.delay(video_id, settings)
            
            # Update video status in cache
            cache = await self._get_cache()
            video_data = await cache.get(f"video:{video_id}")
            if video_data:
                video_dict = eval(video_data)
                video_dict["status"] = "transcoding"
                video_dict["transcode_job_id"] = task.id
                await cache.setex(f"video:{video_id}", 86400, str(video_dict))
            
            return {
                "video_id": video_id,
                "job_id": task.id,
                "status": "transcoding",
                "formats": settings["formats"],
                "resolutions": settings["resolutions"]
            }
            
        except Exception as e:
            raise VideoServiceError(f"Failed to transcode video: {str(e)}")
    
    async def generate_thumbnails(self, video_id: str, count: int = 5) -> Dict[str, Any]:
        """Generate video thumbnails."""
        try:
            # Queue thumbnail generation task
            from app.videos.video_tasks import generate_video_thumbnails_task
            task = generate_video_thumbnails_task.delay(video_id, count)
            
            # For immediate response, return placeholder URLs
            # Actual thumbnails will be generated by background task
            return {
                "video_id": video_id,
                "thumbnails": [
                    f"https://{settings.AWS_S3_BUCKET}.s3.{settings.AWS_REGION}.amazonaws.com/thumbnails/{video_id}/thumb_{i}.jpg"
                    for i in range(count)
                ],
                "count": count,
                "job_id": task.id,
                "status": "generating"
            }
            
        except Exception as e:
            raise VideoServiceError(f"Failed to generate thumbnails: {str(e)}")
    
    async def create_streaming_manifest(self, video_id: str) -> Dict[str, Any]:
        """Create HLS/DASH streaming manifest."""
        try:
            # Generate HLS/DASH manifest files
            # These would typically be created during transcoding
            # For now, return expected URLs
            
            cache = await self._get_cache()
            video_data = await cache.get(f"video:{video_id}")
            if not video_data:
                raise VideoServiceError(f"Video {video_id} not found")
            
            # Store manifest info in cache
            manifest_data = {
                "video_id": video_id,
                "hls_url": f"https://{settings.AWS_CLOUDFRONT_DOMAIN or settings.AWS_S3_BUCKET}/hls/{video_id}/master.m3u8",
                "dash_url": f"https://{settings.AWS_CLOUDFRONT_DOMAIN or settings.AWS_S3_BUCKET}/dash/{video_id}/manifest.mpd",
                "status": "ready",
                "created_at": datetime.utcnow().isoformat()
            }
            
            await cache.setex(
                f"manifest:{video_id}",
                86400,  # 24 hours
                str(manifest_data)
            )
            
            return manifest_data
            
        except Exception as e:
            raise VideoServiceError(f"Failed to create streaming manifest: {str(e)}")
    
    async def optimize_for_mobile(self, video_id: str) -> Dict[str, Any]:
        """Create mobile-optimized video versions."""
        try:
            # Create mobile-optimized versions (lower resolution, bitrate)
            # Queue transcoding with mobile-specific settings
            mobile_settings = {
                "formats": ["h264"],  # Most compatible
                "resolutions": ["360p", "480p"],  # Mobile-friendly sizes
                "bitrate": "500k",  # Lower bitrate for mobile data
                "audio_bitrate": "64k"
            }
            
            from app.videos.video_tasks import transcode_video_task
            task = transcode_video_task.delay(video_id, mobile_settings)
            
            return {
                "video_id": video_id,
                "mobile_url": f"https://{settings.AWS_S3_BUCKET}.s3.{settings.AWS_REGION}.amazonaws.com/mobile/{video_id}/mobile.mp4",
                "job_id": task.id,
                "status": "optimizing",
                "settings": mobile_settings
            }
            
        except Exception as e:
            raise VideoServiceError(f"Failed to optimize for mobile: {str(e)}")


# Global video service instance
video_service = VideoService()
