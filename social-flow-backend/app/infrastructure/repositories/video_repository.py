"""
Video Repository Implementation

SQLAlchemy-based implementation of IVideoRepository interface.
"""

from datetime import datetime, timedelta
from typing import List, Optional
from uuid import UUID

from sqlalchemy import select, func, and_, or_
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.video import Video as VideoModel, VideoStatus as DBVideoStatus, VideoVisibility as DBVideoVisibility
from app.domain.entities.video import VideoEntity
from app.domain.repositories.video_repository import IVideoRepository
from app.domain.value_objects import VideoStatus
from app.infrastructure.repositories.mappers import VideoMapper


class VideoRepository(IVideoRepository):
    """
    SQLAlchemy implementation of video repository.
    
    Handles persistence and retrieval of video entities.
    """
    
    def __init__(self, session: AsyncSession):
        self._session = session
        self._mapper = VideoMapper()
    
    async def get_by_id(self, id: UUID) -> Optional[VideoEntity]:
        """Get video by ID."""
        result = await self._session.execute(
            select(VideoModel).where(VideoModel.id == id)
        )
        model = result.scalar_one_or_none()
        
        if model is None:
            return None
        
        return self._mapper.to_entity(model)
    
    async def get_by_owner(
        self,
        owner_id: UUID,
        skip: int = 0,
        limit: int = 20,
    ) -> List[VideoEntity]:
        """Get videos by owner."""
        result = await self._session.execute(
            select(VideoModel)
            .where(VideoModel.owner_id == owner_id)
            .offset(skip)
            .limit(limit)
            .order_by(VideoModel.created_at.desc())
        )
        models = result.scalars().all()
        
        return [self._mapper.to_entity(model) for model in models]
    
    async def get_by_status(
        self,
        status: VideoStatus,
        skip: int = 0,
        limit: int = 100,
    ) -> List[VideoEntity]:
        """Get videos by status."""
        db_status = DBVideoStatus(status.value)
        result = await self._session.execute(
            select(VideoModel)
            .where(VideoModel.status == db_status)
            .offset(skip)
            .limit(limit)
            .order_by(VideoModel.created_at.desc())
        )
        models = result.scalars().all()
        
        return [self._mapper.to_entity(model) for model in models]
    
    async def get_public_videos(
        self,
        skip: int = 0,
        limit: int = 20,
    ) -> List[VideoEntity]:
        """Get public videos."""
        result = await self._session.execute(
            select(VideoModel)
            .where(
                and_(
                    VideoModel.visibility == DBVideoVisibility.PUBLIC,
                    VideoModel.status == DBVideoStatus.PROCESSED,
                    VideoModel.is_approved == True
                )
            )
            .offset(skip)
            .limit(limit)
            .order_by(VideoModel.created_at.desc())
        )
        models = result.scalars().all()
        
        return [self._mapper.to_entity(model) for model in models]
    
    async def get_trending_videos(
        self,
        hours: int = 24,
        skip: int = 0,
        limit: int = 20,
    ) -> List[VideoEntity]:
        """Get trending videos based on recent engagement."""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        
        result = await self._session.execute(
            select(VideoModel)
            .where(
                and_(
                    VideoModel.visibility == DBVideoVisibility.PUBLIC,
                    VideoModel.status == DBVideoStatus.PROCESSED,
                    VideoModel.is_approved == True,
                    VideoModel.created_at >= cutoff_time
                )
            )
            .offset(skip)
            .limit(limit)
            .order_by(
                (VideoModel.views_count * 0.5 + 
                 VideoModel.likes_count * 2 + 
                 VideoModel.comments_count * 3 + 
                 VideoModel.shares_count * 5).desc()
            )
        )
        models = result.scalars().all()
        
        return [self._mapper.to_entity(model) for model in models]
    
    async def get_recommended_for_user(
        self,
        user_id: UUID,
        skip: int = 0,
        limit: int = 20,
    ) -> List[VideoEntity]:
        """Get recommended videos for a user."""
        # Simple recommendation: public videos ordered by engagement
        # TODO: Implement ML-based recommendations
        result = await self._session.execute(
            select(VideoModel)
            .where(
                and_(
                    VideoModel.visibility == DBVideoVisibility.PUBLIC,
                    VideoModel.status == DBVideoStatus.PROCESSED,
                    VideoModel.is_approved == True,
                    VideoModel.owner_id != user_id  # Exclude user's own videos
                )
            )
            .offset(skip)
            .limit(limit)
            .order_by(
                (VideoModel.views_count + VideoModel.likes_count * 2).desc()
            )
        )
        models = result.scalars().all()
        
        return [self._mapper.to_entity(model) for model in models]
    
    async def search_by_title(
        self,
        query: str,
        skip: int = 0,
        limit: int = 20,
    ) -> List[VideoEntity]:
        """Search videos by title."""
        result = await self._session.execute(
            select(VideoModel)
            .where(
                and_(
                    VideoModel.title.ilike(f"%{query}%"),
                    VideoModel.visibility == DBVideoVisibility.PUBLIC,
                    VideoModel.status == DBVideoStatus.PROCESSED,
                    VideoModel.is_approved == True
                )
            )
            .offset(skip)
            .limit(limit)
            .order_by(VideoModel.views_count.desc())
        )
        models = result.scalars().all()
        
        return [self._mapper.to_entity(model) for model in models]
    
    async def search_by_tags(
        self,
        tags: List[str],
        skip: int = 0,
        limit: int = 20,
    ) -> List[VideoEntity]:
        """Search videos by tags."""
        # Build conditions for each tag
        tag_conditions = [
            VideoModel.tags.ilike(f"%{tag}%") for tag in tags
        ]
        
        result = await self._session.execute(
            select(VideoModel)
            .where(
                and_(
                    or_(*tag_conditions),
                    VideoModel.visibility == DBVideoVisibility.PUBLIC,
                    VideoModel.status == DBVideoStatus.PROCESSED,
                    VideoModel.is_approved == True
                )
            )
            .offset(skip)
            .limit(limit)
            .order_by(VideoModel.views_count.desc())
        )
        models = result.scalars().all()
        
        return [self._mapper.to_entity(model) for model in models]
    
    async def get_flagged_videos(
        self,
        skip: int = 0,
        limit: int = 100,
    ) -> List[VideoEntity]:
        """Get flagged videos for moderation."""
        result = await self._session.execute(
            select(VideoModel)
            .where(VideoModel.is_flagged == True)
            .offset(skip)
            .limit(limit)
            .order_by(VideoModel.flagged_at.desc())
        )
        models = result.scalars().all()
        
        return [self._mapper.to_entity(model) for model in models]
    
    async def get_pending_approval(
        self,
        skip: int = 0,
        limit: int = 100,
    ) -> List[VideoEntity]:
        """Get videos pending approval."""
        result = await self._session.execute(
            select(VideoModel)
            .where(
                and_(
                    VideoModel.status == DBVideoStatus.PROCESSED,
                    VideoModel.is_approved == False,
                    VideoModel.is_rejected == False
                )
            )
            .offset(skip)
            .limit(limit)
            .order_by(VideoModel.processing_completed_at.desc())
        )
        models = result.scalars().all()
        
        return [self._mapper.to_entity(model) for model in models]
    
    async def get_monetized_videos(
        self,
        owner_id: UUID,
        skip: int = 0,
        limit: int = 100,
    ) -> List[VideoEntity]:
        """Get monetized videos for a user."""
        result = await self._session.execute(
            select(VideoModel)
            .where(
                and_(
                    VideoModel.owner_id == owner_id,
                    VideoModel.is_monetized == True
                )
            )
            .offset(skip)
            .limit(limit)
            .order_by(VideoModel.ad_revenue.desc())
        )
        models = result.scalars().all()
        
        return [self._mapper.to_entity(model) for model in models]
    
    async def get_total_views_for_user(self, owner_id: UUID) -> int:
        """Get total views across all user's videos."""
        result = await self._session.execute(
            select(func.sum(VideoModel.views_count))
            .where(VideoModel.owner_id == owner_id)
        )
        total = result.scalar()
        return total or 0
    
    async def get_total_revenue_for_user(self, owner_id: UUID) -> float:
        """Get total ad revenue across all user's videos."""
        result = await self._session.execute(
            select(func.sum(VideoModel.ad_revenue))
            .where(VideoModel.owner_id == owner_id)
        )
        total = result.scalar()
        return total or 0.0
    
    async def get_all(
        self,
        skip: int = 0,
        limit: int = 100,
    ) -> List[VideoEntity]:
        """Get all videos with pagination."""
        result = await self._session.execute(
            select(VideoModel)
            .offset(skip)
            .limit(limit)
            .order_by(VideoModel.created_at.desc())
        )
        models = result.scalars().all()
        
        return [self._mapper.to_entity(model) for model in models]
    
    async def add(self, entity: VideoEntity) -> VideoEntity:
        """Add new video."""
        model = self._mapper.to_model(entity)
        self._session.add(model)
        await self._session.flush()
        await self._session.refresh(model)
        
        return self._mapper.to_entity(model)
    
    async def update(self, entity: VideoEntity) -> VideoEntity:
        """Update existing video."""
        # Get existing model
        result = await self._session.execute(
            select(VideoModel).where(VideoModel.id == entity.id)
        )
        model = result.scalar_one_or_none()
        
        if model is None:
            raise ValueError(f"Video with id {entity.id} not found")
        
        # Update model from entity
        model = self._mapper.to_model(entity, model)
        await self._session.flush()
        await self._session.refresh(model)
        
        return self._mapper.to_entity(model)
    
    async def delete(self, id: UUID) -> bool:
        """Delete video by ID."""
        result = await self._session.execute(
            select(VideoModel).where(VideoModel.id == id)
        )
        model = result.scalar_one_or_none()
        
        if model is None:
            return False
        
        await self._session.delete(model)
        await self._session.flush()
        
        return True
    
    async def exists(self, id: UUID) -> bool:
        """Check if video exists."""
        result = await self._session.execute(
            select(func.count(VideoModel.id)).where(VideoModel.id == id)
        )
        count = result.scalar()
        return count > 0
    
    async def count(self) -> int:
        """Get total count of videos."""
        result = await self._session.execute(
            select(func.count(VideoModel.id))
        )
        return result.scalar()
    
    async def get_for_discovery(
        self,
        skip: int = 0,
        limit: int = 20,
    ) -> List[VideoEntity]:
        """
        Get videos for discovery feed.
        Returns public, processed videos ordered by recent engagement.
        """
        return await self.get_trending_videos(hours=72, skip=skip, limit=limit)

