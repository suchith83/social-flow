"""
Live Streaming Service for handling live video streaming operations.

This service integrates all existing live streaming modules from
the storage/video-storage/live-streaming directory into the FastAPI application.
"""

import logging
from typing import Any, Dict, List, Optional
from datetime import datetime
import uuid
import boto3
from botocore.exceptions import ClientError

from app.core.config import settings
from app.core.exceptions import LiveStreamingServiceError
from app.core.redis import get_cache

logger = logging.getLogger(__name__)


class LiveStreamingService:
    """Main live streaming service integrating all live streaming capabilities."""

    def __init__(self):
        self.ivs_client = None
        self.media_live_client = None
        self.cache = None
        self._initialize_services()

    async def _get_cache(self):
        """Get Redis cache instance."""
        if self.cache is None:
            self.cache = await get_cache()
        return self.cache

    def _initialize_services(self):
        """Initialize live streaming services."""
        try:
            # Initialize AWS IVS client
            if settings.AWS_ACCESS_KEY_ID and settings.AWS_SECRET_ACCESS_KEY and settings.AWS_REGION:
                self.ivs_client = boto3.client(
                    'ivs',
                    aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
                    aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
                    region_name=settings.AWS_REGION
                )
                
                self.media_live_client = boto3.client(
                    'medialive',
                    aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
                    aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
                    region_name=settings.AWS_REGION
                )
                
                logger.info("AWS IVS and MediaLive clients initialized successfully")
            else:
                logger.warning("AWS credentials not fully configured. Live streaming clients not initialized.")
            
            logger.info("Live Streaming Service initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Live Streaming Service: {e}")
            raise LiveStreamingServiceError(f"Failed to initialize Live Streaming Service: {e}")

    async def create_live_stream(self, user_id: str, title: str, description: str = None, 
                               tags: List[str] = None, thumbnail_url: str = None) -> Dict[str, Any]:
        """Create a new live stream."""
        try:
            if not self.ivs_client:
                raise LiveStreamingServiceError("IVS client not initialized")
            
            stream_id = str(uuid.uuid4())
            
            # Create IVS channel
            response = self.ivs_client.create_channel(
                name=f"{title}_{stream_id}",
                latencyMode='LOW',
                type='STANDARD',
                authorized=False,
                tags={
                    'user_id': user_id,
                    'stream_id': stream_id,
                    'title': title
                }
            )
            
            channel_arn = response['channel']['arn']
            ingest_endpoint = response['channel']['ingestEndpoint']
            stream_key = response['streamKey']['value']
            playback_url = response['channel']['playbackUrl']
            
            # Store stream info in cache
            cache = await self._get_cache()
            stream_data = {
                "stream_id": stream_id,
                "user_id": user_id,
                "title": title,
                "description": description,
                "tags": tags or [],
                "thumbnail_url": thumbnail_url,
                "channel_arn": channel_arn,
                "ingest_endpoint": ingest_endpoint,
                "stream_key": stream_key,
                "playback_url": playback_url,
                "status": "created",
                "created_at": datetime.utcnow().isoformat()
            }
            
            await cache.setex(f"live_stream:{stream_id}", 3600, json.dumps(stream_data))
            
            return stream_data
        except ClientError as e:
            raise LiveStreamingServiceError(f"AWS IVS channel creation failed: {str(e)}")
        except Exception as e:
            raise LiveStreamingServiceError(f"Failed to create live stream: {str(e)}")

    async def start_live_stream(self, stream_id: str) -> Dict[str, Any]:
        """Start a live stream."""
        try:
            cache = await self._get_cache()
            stream_data = await cache.get(f"live_stream:{stream_id}")
            
            if not stream_data:
                raise LiveStreamingServiceError("Stream not found")
            
            stream_info = json.loads(stream_data)
            
            # Update stream status
            stream_info["status"] = "live"
            stream_info["started_at"] = datetime.utcnow().isoformat()
            
            await cache.setex(f"live_stream:{stream_id}", 3600, json.dumps(stream_info))
            
            return stream_info
        except Exception as e:
            raise LiveStreamingServiceError(f"Failed to start live stream: {str(e)}")

    async def stop_live_stream(self, stream_id: str) -> Dict[str, Any]:
        """Stop a live stream."""
        try:
            cache = await self._get_cache()
            stream_data = await cache.get(f"live_stream:{stream_id}")
            
            if not stream_data:
                raise LiveStreamingServiceError("Stream not found")
            
            stream_info = json.loads(stream_data)
            
            # Update stream status
            stream_info["status"] = "ended"
            stream_info["ended_at"] = datetime.utcnow().isoformat()
            
            await cache.setex(f"live_stream:{stream_id}", 3600, json.dumps(stream_info))
            
            return stream_info
        except Exception as e:
            raise LiveStreamingServiceError(f"Failed to stop live stream: {str(e)}")

    async def get_live_stream(self, stream_id: str) -> Dict[str, Any]:
        """Get live stream information."""
        try:
            cache = await self._get_cache()
            stream_data = await cache.get(f"live_stream:{stream_id}")
            
            if not stream_data:
                raise LiveStreamingServiceError("Stream not found")
            
            return json.loads(stream_data)
        except Exception as e:
            raise LiveStreamingServiceError(f"Failed to get live stream: {str(e)}")

    async def get_user_live_streams(self, user_id: str, limit: int = 50, 
                                  offset: int = 0) -> List[Dict[str, Any]]:
        """Get live streams for a user."""
        try:
            # TODO: Retrieve streams from database
            streams = []
            
            # Placeholder data
            for i in range(min(limit, 10)):
                streams.append({
                    "stream_id": str(uuid.uuid4()),
                    "user_id": user_id,
                    "title": f"Live Stream {i+1}",
                    "status": "live" if i < 5 else "ended",
                    "created_at": datetime.utcnow().isoformat()
                })
            
            return streams
        except Exception as e:
            raise LiveStreamingServiceError(f"Failed to get user live streams: {str(e)}")

    async def get_live_streams(self, limit: int = 50, offset: int = 0, 
                             status: str = "live") -> List[Dict[str, Any]]:
        """Get all live streams."""
        try:
            # TODO: Retrieve streams from database
            streams = []
            
            # Placeholder data
            for i in range(min(limit, 20)):
                streams.append({
                    "stream_id": str(uuid.uuid4()),
                    "user_id": str(uuid.uuid4()),
                    "title": f"Live Stream {i+1}",
                    "status": status,
                    "viewer_count": i * 10,
                    "created_at": datetime.utcnow().isoformat()
                })
            
            return streams
        except Exception as e:
            raise LiveStreamingServiceError(f"Failed to get live streams: {str(e)}")

    async def update_live_stream(self, stream_id: str, updates: Dict[str, Any]) -> Dict[str, Any]:
        """Update live stream information."""
        try:
            cache = await self._get_cache()
            stream_data = await cache.get(f"live_stream:{stream_id}")
            
            if not stream_data:
                raise LiveStreamingServiceError("Stream not found")
            
            stream_info = json.loads(stream_data)
            
            # Update stream info
            for key, value in updates.items():
                if key in stream_info:
                    stream_info[key] = value
            
            stream_info["updated_at"] = datetime.utcnow().isoformat()
            
            await cache.setex(f"live_stream:{stream_id}", 3600, json.dumps(stream_info))
            
            return stream_info
        except Exception as e:
            raise LiveStreamingServiceError(f"Failed to update live stream: {str(e)}")

    async def delete_live_stream(self, stream_id: str) -> Dict[str, Any]:
        """Delete a live stream."""
        try:
            cache = await self._get_cache()
            stream_data = await cache.get(f"live_stream:{stream_id}")
            
            if not stream_data:
                raise LiveStreamingServiceError("Stream not found")
            
            stream_info = json.loads(stream_data)
            
            # Delete from cache
            await cache.delete(f"live_stream:{stream_id}")
            
            # TODO: Delete from database
            # TODO: Clean up AWS resources if needed
            
            return {
                "stream_id": stream_id,
                "status": "deleted",
                "deleted_at": datetime.utcnow().isoformat()
            }
        except Exception as e:
            raise LiveStreamingServiceError(f"Failed to delete live stream: {str(e)}")

    async def get_stream_analytics(self, stream_id: str, time_range: str = "1h") -> Dict[str, Any]:
        """Get analytics for a live stream."""
        try:
            # TODO: Retrieve analytics from database
            return {
                "stream_id": stream_id,
                "time_range": time_range,
                "viewer_count": 0,
                "peak_viewers": 0,
                "total_views": 0,
                "average_watch_time": 0,
                "engagement_rate": 0.0,
                "generated_at": datetime.utcnow().isoformat()
            }
        except Exception as e:
            raise LiveStreamingServiceError(f"Failed to get stream analytics: {str(e)}")

    async def record_viewer_join(self, stream_id: str, viewer_id: str) -> Dict[str, Any]:
        """Record a viewer joining the stream."""
        try:
            # TODO: Record viewer join in database
            return {
                "stream_id": stream_id,
                "viewer_id": viewer_id,
                "joined_at": datetime.utcnow().isoformat()
            }
        except Exception as e:
            raise LiveStreamingServiceError(f"Failed to record viewer join: {str(e)}")

    async def record_viewer_leave(self, stream_id: str, viewer_id: str) -> Dict[str, Any]:
        """Record a viewer leaving the stream."""
        try:
            # TODO: Record viewer leave in database
            return {
                "stream_id": stream_id,
                "viewer_id": viewer_id,
                "left_at": datetime.utcnow().isoformat()
            }
        except Exception as e:
            raise LiveStreamingServiceError(f"Failed to record viewer leave: {str(e)}")

    async def get_stream_viewers(self, stream_id: str) -> List[Dict[str, Any]]:
        """Get current viewers of a stream."""
        try:
            # TODO: Retrieve current viewers from database
            viewers = []
            
            # Placeholder data
            for i in range(5):
                viewers.append({
                    "viewer_id": str(uuid.uuid4()),
                    "username": f"viewer{i+1}",
                    "joined_at": datetime.utcnow().isoformat()
                })
            
            return viewers
        except Exception as e:
            raise LiveStreamingServiceError(f"Failed to get stream viewers: {str(e)}")


live_streaming_service = LiveStreamingService()
