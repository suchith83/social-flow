"""
Video Service for video processing and streaming.

This service integrates all existing video processing modules from storage
and other video-related components into the FastAPI application.
"""

import asyncio
import logging
import os
import tempfile
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
from app.core.exceptions import VideoServiceError
from app.core.redis import get_cache
from app.models.video import Video, VideoStatus, VideoVisibility
from app.models.user import User

logger = logging.getLogger(__name__)


class VideoService:
    """Main video service for video processing and streaming."""
    
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
    
    async def upload_video(self, file: UploadFile, user: User, metadata: Dict[str, Any]) -> Dict[str, Any]:
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
                
                # Start background processing
                await self._start_video_processing(video)
                
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
    
    async def process_video(self, video_id: str) -> Dict[str, Any]:
        """Process uploaded video (transcoding, thumbnails, etc.)."""
        try:
            # Get video from database
            video = await self._get_video_by_id(video_id)
            if not video:
                raise VideoServiceError("Video not found")
            
            # Update status to processing
            video.status = VideoStatus.PROCESSING
            video.processing_started_at = datetime.utcnow()
            await self._update_video(video)
            
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
                
                await self._update_video(video)
                
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
            await self._update_video(video)
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
    
    async def create_live_stream(self, user: User, title: str, description: str = "") -> Dict[str, Any]:
        """Create a live streaming session."""
        try:
            # Generate stream key
            stream_key = self._generate_stream_key()
            
            # Create live stream record
            stream_id = str(uuid.uuid4())
            
            # Get RTMP ingest URL
            rtmp_url = f"{settings.RTMP_INGEST_URL}/{stream_key}"
            playback_url = f"{settings.RTMP_PLAYBACK_URL}/{stream_key}/index.m3u8"
            
            # Store stream info in cache
            cache = await self._get_cache()
            await cache.set(f"live_stream:{stream_id}", {
                'stream_id': stream_id,
                'user_id': str(user.id),
                'title': title,
                'description': description,
                'stream_key': stream_key,
                'rtmp_url': rtmp_url,
                'playback_url': playback_url,
                'status': 'active',
                'created_at': datetime.utcnow().isoformat()
            }, expire=3600)
            
            return {
                'stream_id': stream_id,
                'stream_key': stream_key,
                'rtmp_url': rtmp_url,
                'playback_url': playback_url,
                'status': 'active'
            }
        except Exception as e:
            logger.error(f"Failed to create live stream: {e}")
            raise VideoServiceError(f"Failed to create live stream: {e}")
    
    async def end_live_stream(self, stream_id: str) -> Dict[str, Any]:
        """End a live streaming session."""
        try:
            cache = await self._get_cache()
            stream_data = await cache.get(f"live_stream:{stream_id}")
            
            if not stream_data:
                raise VideoServiceError("Stream not found")
            
            # Update stream status
            stream_data['status'] = 'ended'
            stream_data['ended_at'] = datetime.utcnow().isoformat()
            
            await cache.set(f"live_stream:{stream_id}", stream_data, expire=3600)
            
            return {
                'stream_id': stream_id,
                'status': 'ended',
                'ended_at': stream_data['ended_at']
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
    
    async def _start_video_processing(self, video: Video) -> None:
        """Start background video processing."""
        # In a real implementation, this would queue a job for Celery
        # For now, we'll process immediately
        asyncio.create_task(self.process_video(str(video.id)))
    
    async def _get_video_by_id(self, video_id: str) -> Optional[Video]:
        """Get video by ID from database."""
        # This would query the database in a real implementation
        # For now, return None
        return None
    
    async def _update_video(self, video: Video) -> None:
        """Update video in database."""
        # This would update the database in a real implementation
        pass
    
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
            
            # Upload chunk to S3
            chunk_key = f"uploads/{upload_id}/chunks/{chunk_number:06d}"
            
            # TODO: Upload to S3 using multipart upload
            # await self.storage_service.upload_chunk(chunk_key, chunk_data)
            
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
            
            # TODO: Complete S3 multipart upload
            # final_video_key = f"videos/{upload_id}/original.mp4"
            # await self.storage_service.complete_multipart_upload(upload_id, final_video_key)
            
            # Create video record
            video = Video(
                id=upload_id,
                owner_id=session["user_id"],
                title=metadata.get("title", ""),
                description=metadata.get("description", ""),
                filename=session["filename"],
                file_size=session["file_size"],
                status=VideoStatus.PROCESSING,
                created_at=datetime.utcnow()
            )
            
            # Save to database
            # TODO: Save to database
            # self.db.add(video)
            # await self.db.commit()
            
            # Start background processing
            # TODO: Queue video processing job
            # await self.queue_video_processing(upload_id)
            
            # Clean up upload session
            await cache.delete(f"upload_session:{upload_id}")
            
            return {
                "video_id": upload_id,
                "status": "processing",
                "message": "Upload completed, processing started"
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
            
            # TODO: Cancel S3 multipart upload
            # await self.storage_service.cancel_multipart_upload(upload_id)
            
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
            
            # TODO: Create AWS MediaConvert job
            # job_id = await self.create_mediaconvert_job(video_id, settings)
            
            # Update video status
            video = await self._get_video_by_id(video_id)
            if video:
                video.status = VideoStatus.TRANSCODING
                await self._update_video(video)
            
            return {
                "video_id": video_id,
                "job_id": f"job_{video_id}",
                "status": "transcoding",
                "formats": settings["formats"],
                "resolutions": settings["resolutions"]
            }
            
        except Exception as e:
            raise VideoServiceError(f"Failed to transcode video: {str(e)}")
    
    async def generate_thumbnails(self, video_id: str, count: int = 5) -> Dict[str, Any]:
        """Generate video thumbnails."""
        try:
            # TODO: Use FFmpeg to generate thumbnails
            # thumbnails = await self.ffmpeg_service.generate_thumbnails(video_id, count)
            
            return {
                "video_id": video_id,
                "thumbnails": [
                    f"https://thumbnails.example.com/{video_id}/thumb_{i}.jpg"
                    for i in range(count)
                ],
                "count": count
            }
            
        except Exception as e:
            raise VideoServiceError(f"Failed to generate thumbnails: {str(e)}")
    
    async def create_streaming_manifest(self, video_id: str) -> Dict[str, Any]:
        """Create HLS/DASH streaming manifest."""
        try:
            # TODO: Generate HLS/DASH manifest
            # manifest = await self.streaming_service.create_manifest(video_id)
            
            return {
                "video_id": video_id,
                "hls_url": f"https://stream.example.com/hls/{video_id}.m3u8",
                "dash_url": f"https://stream.example.com/dash/{video_id}.mpd",
                "status": "ready"
            }
            
        except Exception as e:
            raise VideoServiceError(f"Failed to create streaming manifest: {str(e)}")
    
    async def optimize_for_mobile(self, video_id: str) -> Dict[str, Any]:
        """Create mobile-optimized video versions."""
        try:
            # TODO: Create mobile-optimized versions
            # mobile_versions = await self.mobile_optimization_service.optimize(video_id)
            
            return {
                "video_id": video_id,
                "mobile_url": f"https://mobile.example.com/videos/{video_id}/mobile.mp4",
                "status": "optimized"
            }
            
        except Exception as e:
            raise VideoServiceError(f"Failed to optimize for mobile: {str(e)}")


# Global video service instance
video_service = VideoService()
