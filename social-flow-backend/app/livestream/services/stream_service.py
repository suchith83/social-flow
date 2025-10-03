"""
Live Streaming Service

Core service for managing live streams with AWS MediaLive/IVS integration.
Handles stream lifecycle, viewer tracking, and recording management.
"""

import secrets
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from uuid import UUID

import boto3
from botocore.exceptions import ClientError
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.livestream.models.livestream import (
    LiveStream, StreamViewer, ChatMessage,
    StreamStatus, StreamQuality, ChatMessageType
)
from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)


class StreamServiceError(Exception):
    """Base exception for stream service errors"""
    pass


class StreamNotFoundError(StreamServiceError):
    """Raised when stream is not found"""
    pass


class StreamAlreadyLiveError(StreamServiceError):
    """Raised when user already has a live stream"""
    pass


class LiveStreamService:
    """
    Live streaming service with AWS MediaLive/IVS integration
    
    Features:
    - Stream lifecycle management (create, start, stop)
    - AWS MediaLive/IVS channel management
    - RTMP ingest configuration
    - Viewer tracking and analytics
    - Stream recording to S3
    - Chat moderation
    """
    
    def __init__(self, db: AsyncSession):
        self.db = db
        
        # AWS clients
        if not settings.TESTING:
            self.ivs_client = boto3.client(
                'ivs',
                region_name=settings.AWS_REGION,
                aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
                aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY
            )
            
            self.s3_client = boto3.client(
                's3',
                region_name=settings.AWS_REGION,
                aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
                aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY
            )
        else:
            self.ivs_client = None
            self.s3_client = None
    
    # ==================== Stream Creation ====================
    
    async def create_stream(
        self,
        user_id: UUID,
        title: str,
        description: Optional[str] = None,
        category: Optional[str] = None,
        tags: Optional[List[str]] = None,
        scheduled_start: Optional[datetime] = None,
        is_public: bool = True,
        quality: StreamQuality = StreamQuality.HIGH,
        **kwargs
    ) -> LiveStream:
        """
        Create a new live stream
        
        Args:
            user_id: User creating the stream
            title: Stream title
            description: Stream description
            category: Stream category
            tags: Stream tags
            scheduled_start: Scheduled start time
            is_public: Whether stream is public
            quality: Stream quality setting
            
        Returns:
            Created LiveStream instance
            
        Raises:
            StreamAlreadyLiveError: User already has a live stream
        """
        # Check if user already has an active stream
        existing_stream = await self._get_active_stream(user_id)
        if existing_stream:
            raise StreamAlreadyLiveError(
                f"User {user_id} already has an active stream"
            )
        
        # Generate unique stream key
        stream_key = self._generate_stream_key()
        
        # Create AWS IVS channel
        ivs_channel = await self._create_ivs_channel(
            user_id=user_id,
            stream_key=stream_key,
            title=title
        )
        
        # Create stream record
        stream = LiveStream(
            user_id=user_id,
            title=title,
            description=description,
            category=category,
            tags=tags or [],
            stream_key=stream_key,
            scheduled_start=scheduled_start,
            is_public=is_public,
            quality=quality.value,
            status=StreamStatus.SCHEDULED.value,
            ivs_channel_arn=ivs_channel.get('channel', {}).get('arn'),
            ivs_ingest_endpoint=ivs_channel.get('channel', {}).get('ingestEndpoint'),
            ivs_playback_url=ivs_channel.get('channel', {}).get('playbackUrl'),
            stream_url=self._build_rtmp_url(ivs_channel, stream_key),
            playback_url=ivs_channel.get('channel', {}).get('playbackUrl'),
            **kwargs
        )
        
        self.db.add(stream)
        await self.db.commit()
        await self.db.refresh(stream)
        
        logger.info(f"Created stream {stream.id} for user {user_id}")
        return stream
    
    async def start_stream(self, stream_id: UUID, user_id: UUID) -> LiveStream:
        """
        Start a live stream
        
        Args:
            stream_id: Stream ID
            user_id: User starting the stream
            
        Returns:
            Updated LiveStream instance
            
        Raises:
            StreamNotFoundError: Stream not found
            PermissionError: User doesn't own stream
        """
        stream = await self.get_stream(stream_id)
        
        if stream.user_id != user_id:
            raise PermissionError("User doesn't own this stream")
        
        if stream.status == StreamStatus.LIVE.value:
            logger.warning(f"Stream {stream_id} is already live")
            return stream
        
        # Update stream status
        stream.status = StreamStatus.STARTING.value
        stream.started_at = datetime.utcnow()
        await self.db.commit()
        
        # Start AWS IVS channel (if not auto-started)
        try:
            if stream.ivs_channel_arn:
                await self._start_ivs_channel(stream.ivs_channel_arn)
        except Exception as e:
            logger.error(f"Error starting IVS channel: {e}")
        
        # Update to live status
        stream.status = StreamStatus.LIVE.value
        await self.db.commit()
        await self.db.refresh(stream)
        
        logger.info(f"Started stream {stream_id}")
        return stream
    
    async def stop_stream(self, stream_id: UUID, user_id: UUID) -> LiveStream:
        """
        Stop a live stream
        
        Args:
            stream_id: Stream ID
            user_id: User stopping the stream
            
        Returns:
            Updated LiveStream instance
        """
        stream = await self.get_stream(stream_id)
        
        if stream.user_id != user_id:
            raise PermissionError("User doesn't own this stream")
        
        if stream.status != StreamStatus.LIVE.value:
            logger.warning(f"Stream {stream_id} is not live")
            return stream
        
        # Calculate duration
        if stream.started_at:
            duration = (datetime.utcnow() - stream.started_at).total_seconds()
            stream.duration_seconds = int(duration)
        
        # Update stream status
        stream.status = StreamStatus.ENDING.value
        await self.db.commit()
        
        # Stop AWS IVS channel
        try:
            if stream.ivs_channel_arn:
                await self._stop_ivs_channel(stream.ivs_channel_arn)
        except Exception as e:
            logger.error(f"Error stopping IVS channel: {e}")
        
        # Mark stream as ended
        stream.status = StreamStatus.ENDED.value
        stream.ended_at = datetime.utcnow()
        
        # Clean up active viewers
        await self._cleanup_viewers(stream_id)
        
        await self.db.commit()
        await self.db.refresh(stream)
        
        logger.info(f"Stopped stream {stream_id} after {stream.duration_seconds}s")
        return stream
    
    # ==================== Stream Retrieval ====================
    
    async def get_stream(self, stream_id: UUID) -> LiveStream:
        """Get stream by ID"""
        result = await self.db.execute(
            select(LiveStream).where(LiveStream.id == stream_id)
        )
        stream = result.scalar_one_or_none()
        
        if not stream:
            raise StreamNotFoundError(f"Stream {stream_id} not found")
        
        return stream
    
    async def get_live_streams(
        self,
        category: Optional[str] = None,
        limit: int = 20,
        offset: int = 0
    ) -> List[LiveStream]:
        """
        Get currently live streams
        
        Args:
            category: Filter by category
            limit: Maximum number of results
            offset: Result offset
            
        Returns:
            List of live streams
        """
        query = select(LiveStream).where(
            LiveStream.status == StreamStatus.LIVE.value,
            LiveStream.is_public.is_(True)
        )
        
        if category:
            query = query.where(LiveStream.category == category)
        
        query = query.order_by(LiveStream.current_viewers.desc())
        query = query.limit(limit).offset(offset)
        
        result = await self.db.execute(query)
        return list(result.scalars().all())
    
    async def get_user_streams(
        self,
        user_id: UUID,
        include_ended: bool = False,
        limit: int = 20
    ) -> List[LiveStream]:
        """Get streams for a specific user"""
        query = select(LiveStream).where(LiveStream.user_id == user_id)
        
        if not include_ended:
            query = query.where(
                LiveStream.status.in_([
                    StreamStatus.SCHEDULED.value,
                    StreamStatus.LIVE.value
                ])
            )
        
        query = query.order_by(LiveStream.created_at.desc()).limit(limit)
        
        result = await self.db.execute(query)
        return list(result.scalars().all())
    
    # ==================== Viewer Management ====================
    
    async def add_viewer(
        self,
        stream_id: UUID,
        user_id: Optional[UUID],
        session_id: str,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None
    ) -> StreamViewer:
        """
        Add a viewer to a stream
        
        Args:
            stream_id: Stream ID
            user_id: User ID (optional for anonymous)
            session_id: Unique session identifier
            ip_address: Viewer IP address
            user_agent: Viewer user agent
            
        Returns:
            Created StreamViewer instance
        """
    # Removed unused fetch; no need to retrieve stream here
        
        # Check if viewer already exists
        result = await self.db.execute(
            select(StreamViewer).where(
                StreamViewer.stream_id == stream_id,
                StreamViewer.session_id == session_id,
                StreamViewer.left_at.is_(None)
            )
        )
        existing = result.scalar_one_or_none()
        
        if existing:
            # Update heartbeat
            existing.last_heartbeat = datetime.utcnow()
            await self.db.commit()
            return existing
        
        # Create new viewer
        viewer = StreamViewer(
            stream_id=stream_id,
            user_id=user_id,
            session_id=session_id,
            ip_address=ip_address,
            user_agent=user_agent
        )
        
        self.db.add(viewer)
        
        # Update stream viewer count
        await self._update_viewer_count(stream_id)
        
        await self.db.commit()
        await self.db.refresh(viewer)
        
        logger.debug(f"Added viewer {session_id} to stream {stream_id}")
        return viewer
    
    async def update_viewer_heartbeat(
        self,
        stream_id: UUID,
        session_id: str
    ) -> Optional[StreamViewer]:
        """Update viewer heartbeat to keep them active"""
        result = await self.db.execute(
            select(StreamViewer).where(
                StreamViewer.stream_id == stream_id,
                StreamViewer.session_id == session_id,
                StreamViewer.left_at.is_(None)
            )
        )
        viewer = result.scalar_one_or_none()
        
        if viewer:
            viewer.last_heartbeat = datetime.utcnow()
            
            # Update watch time
            if viewer.joined_at:
                watch_time = (datetime.utcnow() - viewer.joined_at).total_seconds()
                viewer.watch_time_seconds = int(watch_time)
            
            await self.db.commit()
        
        return viewer
    
    async def remove_viewer(
        self,
        stream_id: UUID,
        session_id: str
    ) -> None:
        """Remove a viewer from a stream"""
        result = await self.db.execute(
            select(StreamViewer).where(
                StreamViewer.stream_id == stream_id,
                StreamViewer.session_id == session_id,
                StreamViewer.left_at.is_(None)
            )
        )
        viewer = result.scalar_one_or_none()
        
        if viewer:
            viewer.left_at = datetime.utcnow()
            
            # Update watch time
            if viewer.joined_at:
                watch_time = (datetime.utcnow() - viewer.joined_at).total_seconds()
                viewer.watch_time_seconds = int(watch_time)
            
            await self.db.commit()
            
            # Update stream viewer count
            await self._update_viewer_count(stream_id)
            
            logger.debug(f"Removed viewer {session_id} from stream {stream_id}")
    
    async def get_active_viewers(self, stream_id: UUID) -> List[StreamViewer]:
        """Get all active viewers for a stream"""
        cutoff_time = datetime.utcnow() - timedelta(seconds=30)
        
        result = await self.db.execute(
            select(StreamViewer).where(
                StreamViewer.stream_id == stream_id,
                StreamViewer.left_at.is_(None),
                StreamViewer.last_heartbeat >= cutoff_time
            )
        )
        return list(result.scalars().all())
    
    # ==================== Chat Management ====================
    
    async def send_chat_message(
        self,
        stream_id: UUID,
        user_id: UUID,
        content: str,
        message_type: ChatMessageType = ChatMessageType.MESSAGE
    ) -> ChatMessage:
        """
        Send a chat message to a stream
        
        Args:
            stream_id: Stream ID
            user_id: User sending message
            content: Message content
            message_type: Type of message
            
        Returns:
            Created ChatMessage instance
        """
        stream = await self.get_stream(stream_id)
        
        if not stream.chat_enabled:
            raise StreamServiceError("Chat is disabled for this stream")
        
        # Create message
        message = ChatMessage(
            stream_id=stream_id,
            user_id=user_id,
            content=content,
            message_type=message_type.value
        )
        
        self.db.add(message)
        
        # Update stream chat count
        stream.chat_messages_count += 1
        
        await self.db.commit()
        await self.db.refresh(message)
        
        return message
    
    async def delete_chat_message(
        self,
        message_id: UUID,
        deleted_by_user_id: UUID,
        reason: Optional[str] = None
    ) -> ChatMessage:
        """Delete a chat message (moderation)"""
        result = await self.db.execute(
            select(ChatMessage).where(ChatMessage.id == message_id)
        )
        message = result.scalar_one_or_none()
        
        if not message:
            raise StreamServiceError(f"Message {message_id} not found")
        
        message.is_deleted = True
        message.deleted_at = datetime.utcnow()
        message.deleted_by_user_id = deleted_by_user_id
        message.deletion_reason = reason
        
        await self.db.commit()
        await self.db.refresh(message)
        
        return message
    
    async def get_chat_messages(
        self,
        stream_id: UUID,
        limit: int = 100,
        before: Optional[datetime] = None
    ) -> List[ChatMessage]:
        """Get chat messages for a stream"""
        query = select(ChatMessage).where(
            ChatMessage.stream_id == stream_id,
            ChatMessage.is_deleted.is_(False)
        )
        
        if before:
            query = query.where(ChatMessage.sent_at < before)
        
        query = query.order_by(ChatMessage.sent_at.desc()).limit(limit)
        
        result = await self.db.execute(query)
        messages = list(result.scalars().all())
        return list(reversed(messages))  # Return in chronological order
    
    # ==================== Stream Analytics ====================
    
    async def get_stream_analytics(self, stream_id: UUID) -> Dict[str, Any]:
        """
        Get analytics for a stream
        
        Returns:
            Dictionary with analytics metrics
        """
        stream = await self.get_stream(stream_id)
        
        # Get viewer stats
        viewer_stats = await self._get_viewer_stats(stream_id)
        
        # Get chat stats
        chat_stats = await self._get_chat_stats(stream_id)
        
        return {
            "stream_id": str(stream_id),
            "status": stream.status,
            "duration_seconds": stream.duration_seconds,
            "current_viewers": stream.current_viewers,
            "peak_viewers": stream.peak_viewers,
            "total_views": stream.total_views,
            "likes_count": stream.likes_count,
            "chat_messages_count": stream.chat_messages_count,
            "engagement_rate": stream.engagement_rate,
            "viewer_stats": viewer_stats,
            "chat_stats": chat_stats,
            "revenue": {
                "total": stream.total_revenue,
                "is_monetized": stream.is_monetized
            }
        }
    
    # ==================== AWS IVS Integration ====================
    
    async def _create_ivs_channel(
        self,
        user_id: UUID,
        stream_key: str,
        title: str
    ) -> Dict[str, Any]:
        """
        Create AWS IVS channel
        
        Args:
            user_id: User ID
            stream_key: Stream key
            title: Channel title
            
        Returns:
            IVS channel details
        """
        # In tests, return a fake channel response
        if settings.TESTING or not self.ivs_client:
            fake_arn = f"arn:aws:ivs:us-west-2:000000000000:channel/{user_id}"
            return {
                'channel': {
                    'arn': fake_arn,
                    'ingestEndpoint': 'fake.ingest.ivs.amazonaws.com',
                    'playbackUrl': f'https://fake.playback/{user_id}'
                },
                'streamKey': {
                    'arn': f"arn:aws:ivs:us-west-2:000000000000:stream-key/{user_id}",
                    'value': stream_key
                }
            }
        try:
            response = self.ivs_client.create_channel(
                name=f"stream-{user_id}",
                latencyMode='LOW',  # LOW or NORMAL
                type='STANDARD',  # STANDARD or BASIC
                authorized=False,  # Allow public streaming
                tags={
                    'user_id': str(user_id),
                    'stream_key': stream_key,
                    'title': title
                }
            )
            
            # Create stream key
            stream_key_response = self.ivs_client.create_stream_key(
                channelArn=response['channel']['arn'],
                tags={
                    'user_id': str(user_id)
                }
            )
            
            response['streamKey'] = stream_key_response['streamKey']
            
            logger.info(f"Created IVS channel for user {user_id}")
            return response
            
        except ClientError as e:
            logger.error(f"Error creating IVS channel: {e}")
            raise StreamServiceError(f"Failed to create IVS channel: {e}")
    
    async def _start_ivs_channel(self, channel_arn: str) -> None:
        """Start AWS IVS channel"""
        try:
            # IVS channels start automatically when streaming begins
            logger.info(f"IVS channel {channel_arn} ready for streaming")
        except ClientError as e:
            logger.error(f"Error starting IVS channel: {e}")
    
    async def _stop_ivs_channel(self, channel_arn: str) -> None:
        """Stop AWS IVS channel"""
        try:
            # Stop stream
            self.ivs_client.stop_stream(
                channelArn=channel_arn
            )
            logger.info(f"Stopped IVS channel {channel_arn}")
        except ClientError as e:
            logger.error(f"Error stopping IVS channel: {e}")
    
    # ==================== Helper Methods ====================
    
    def _generate_stream_key(self) -> str:
        """Generate a unique stream key"""
        return secrets.token_urlsafe(32)
    
    def _build_rtmp_url(
        self,
        ivs_channel: Dict[str, Any],
        stream_key: str
    ) -> str:
        """Build RTMP ingest URL"""
        ingest_endpoint = ivs_channel.get('channel', {}).get('ingestEndpoint')
        if ingest_endpoint:
            return f"rtmp://{ingest_endpoint}/live/{stream_key}"
        return ""
    
    async def _get_active_stream(self, user_id: UUID) -> Optional[LiveStream]:
        """Get user's active stream if exists"""
        result = await self.db.execute(
            select(LiveStream).where(
                LiveStream.user_id == user_id,
                LiveStream.status.in_([
                    StreamStatus.SCHEDULED.value,
                    StreamStatus.STARTING.value,
                    StreamStatus.LIVE.value
                ])
            )
        )
        return result.scalar_one_or_none()
    
    async def _update_viewer_count(self, stream_id: UUID) -> None:
        """Update current viewer count for a stream"""
        cutoff_time = datetime.utcnow() - timedelta(seconds=30)
        
        result = await self.db.execute(
            select(func.count(StreamViewer.id)).where(
                StreamViewer.stream_id == stream_id,
                StreamViewer.left_at.is_(None),
                StreamViewer.last_heartbeat >= cutoff_time
            )
        )
        count = result.scalar() or 0
        
        # Get stream
        stream = await self.get_stream(stream_id)
        stream.current_viewers = count
        
        # Update peak viewers
        if count > stream.peak_viewers:
            stream.peak_viewers = count
        
        await self.db.commit()
    
    async def _cleanup_viewers(self, stream_id: UUID) -> None:
        """Mark all viewers as left when stream ends"""
        result = await self.db.execute(
            select(StreamViewer).where(
                StreamViewer.stream_id == stream_id,
                StreamViewer.left_at.is_(None)
            )
        )
        viewers = result.scalars().all()
        
        for viewer in viewers:
            viewer.left_at = datetime.utcnow()
            if viewer.joined_at:
                watch_time = (datetime.utcnow() - viewer.joined_at).total_seconds()
                viewer.watch_time_seconds = int(watch_time)
        
        await self.db.commit()
    
    async def _get_viewer_stats(self, stream_id: UUID) -> Dict[str, Any]:
        """Get viewer statistics"""
        result = await self.db.execute(
            select(
                func.count(StreamViewer.id).label('total_viewers'),
                func.avg(StreamViewer.watch_time_seconds).label('avg_watch_time'),
                func.sum(StreamViewer.watch_time_seconds).label('total_watch_time')
            ).where(StreamViewer.stream_id == stream_id)
        )
        row = result.first()
        
        return {
            "total_viewers": row.total_viewers or 0,
            "avg_watch_time_seconds": int(row.avg_watch_time or 0),
            "total_watch_time_seconds": int(row.total_watch_time or 0)
        }
    
    async def _get_chat_stats(self, stream_id: UUID) -> Dict[str, Any]:
        """Get chat statistics"""
        result = await self.db.execute(
            select(
                func.count(ChatMessage.id).label('total_messages'),
                func.count(ChatMessage.id).filter(
                    ChatMessage.is_flagged.is_(True)
                ).label('flagged_messages')
            ).where(ChatMessage.stream_id == stream_id)
        )
        row = result.first()
        
        return {
            "total_messages": row.total_messages or 0,
            "flagged_messages": row.flagged_messages or 0
        }
