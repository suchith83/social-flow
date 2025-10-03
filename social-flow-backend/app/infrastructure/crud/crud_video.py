"""
CRUD operations for Video model.
"""

from typing import List, Optional
from uuid import UUID

from sqlalchemy import select, func, and_
from sqlalchemy.ext.asyncio import AsyncSession

from app.infrastructure.crud.base import CRUDBase
from app.models.video import Video, VideoStatus, VideoVisibility
from app.schemas.video import VideoCreate, VideoUpdate


class CRUDVideo(CRUDBase[Video, VideoCreate, VideoUpdate]):
    """CRUD operations for Video model."""

    async def create_with_owner(
        self,
        db: AsyncSession,
        *,
        obj_in: VideoCreate,
        owner_id: UUID,
        commit: bool = True,
    ) -> Video:
        """
        Create a new video with owner.
        
        Args:
            db: Database session
            obj_in: Pydantic schema with video data
            owner_id: Owner user ID
            commit: Whether to commit the transaction
            
        Returns:
            Created video instance
        """
        from fastapi.encoders import jsonable_encoder
        
        obj_in_data = jsonable_encoder(obj_in)
        
        # Map schema fields to model fields
        model_data = {}
        field_mapping = {
            'title': 'title',
            'description': 'description',
            'tags': 'tags',
            'visibility': 'visibility',
            'original_filename': 'filename',  # Schema uses original_filename, model uses filename
            'file_size': 'file_size',
            'duration': 'duration',
            # Skip fields that don't exist in model:
            # - is_age_restricted -> age_restricted (but we'll add it as age_restricted)
            # - allow_comments (doesn't exist in model)
            # - allow_likes (doesn't exist in model)
            # - is_monetized (doesn't exist in model)
        }
        
        for schema_field, model_field in field_mapping.items():
            if schema_field in obj_in_data and obj_in_data[schema_field] is not None:
                model_data[model_field] = obj_in_data[schema_field]
        
        # Handle special mappings
        if 'is_age_restricted' in obj_in_data:
            model_data['age_restricted'] = obj_in_data['is_age_restricted']
        
        # Add owner_id and set initial status
        model_data['owner_id'] = owner_id
        if 'status' not in model_data:
            model_data['status'] = VideoStatus.UPLOADING
        
        # Set required S3 fields with placeholders (will be updated after upload)
        if 's3_bucket' not in model_data:
            model_data['s3_bucket'] = 'social-flow-videos'  # Default bucket
        if 's3_key' not in model_data:
            # Generate a placeholder key using owner_id
            model_data['s3_key'] = f"uploads/{owner_id}/pending"
        if 's3_region' not in model_data:
            model_data['s3_region'] = 'us-east-1'
        
        db_obj = self.model(**model_data)
        db.add(db_obj)
        
        if commit:
            await db.commit()
            await db.refresh(db_obj)
        else:
            await db.flush()
        
        return db_obj

    async def get_by_user(
        self,
        db: AsyncSession,
        *,
        user_id: UUID,
        skip: int = 0,
        limit: int = 100,
        status: Optional[VideoStatus] = None,
        visibility: Optional[VideoVisibility] = None,
    ) -> List[Video]:
        """
        Get videos by user.
        
        Args:
            db: Database session
            user_id: User ID (owner_id)
            skip: Number of records to skip
            limit: Maximum number of records to return
            status: Filter by video status
            visibility: Filter by video visibility
            
        Returns:
            List of video instances
        """
        query = select(self.model).where(self.model.owner_id == user_id)
        
        if status:
            query = query.where(self.model.status == status)
        if visibility:
            query = query.where(self.model.visibility == visibility)
        
        query = query.order_by(self.model.created_at.desc()).offset(skip).limit(limit)
        result = await db.execute(query)
        return list(result.scalars().all())

    async def get_public_videos(
        self,
        db: AsyncSession,
        *,
        skip: int = 0,
        limit: int = 100,
    ) -> List[Video]:
        """
        Get all public videos.
        
        Args:
            db: Database session
            skip: Number of records to skip
            limit: Maximum number of records to return
            
        Returns:
            List of public video instances
        """
        query = (
            select(self.model)
            .where(
                and_(
                    self.model.visibility == VideoVisibility.PUBLIC,
                    self.model.status == VideoStatus.PROCESSED,
                )
            )
            .order_by(self.model.created_at.desc())
            .offset(skip)
            .limit(limit)
        )
        result = await db.execute(query)
        return list(result.scalars().all())

    async def get_trending(
        self,
        db: AsyncSession,
        *,
        skip: int = 0,
        limit: int = 100,
        days: int = 7,
    ) -> List[Video]:
        """
        Get trending videos based on view count.
        
        Args:
            db: Database session
            skip: Number of records to skip
            limit: Maximum number of records to return
            days: Number of days to look back
            
        Returns:
            List of trending video instances
        """
        from datetime import datetime, timedelta, timezone
        
        since = datetime.now(timezone.utc) - timedelta(days=days)
        
        query = (
            select(self.model)
            .where(
                and_(
                    self.model.visibility == VideoVisibility.PUBLIC,
                    self.model.status == VideoStatus.PROCESSED,
                    self.model.created_at >= since,
                )
            )
            .order_by(self.model.view_count.desc())
            .offset(skip)
            .limit(limit)
        )
        result = await db.execute(query)
        return list(result.scalars().all())

    async def increment_view_count(
        self,
        db: AsyncSession,
        *,
        video_id: UUID,
    ) -> Optional[Video]:
        """
        Increment video view count.
        
        Args:
            db: Database session
            video_id: Video ID
            
        Returns:
            Updated video instance or None if not found
        """
        video = await self.get(db, video_id)
        if not video:
            return None
        
        video.view_count += 1
        db.add(video)
        await db.commit()
        await db.refresh(video)
        
        return video

    async def update_status(
        self,
        db: AsyncSession,
        *,
        video_id: UUID,
        status: VideoStatus,
    ) -> Optional[Video]:
        """
        Update video processing status.
        
        Args:
            db: Database session
            video_id: Video ID
            status: New video status
            
        Returns:
            Updated video instance or None if not found
        """
        video = await self.get(db, video_id)
        if not video:
            return None
        
        video.status = status
        db.add(video)
        await db.commit()
        await db.refresh(video)
        
        return video

    async def search(
        self,
        db: AsyncSession,
        *,
        query_text: str,
        skip: int = 0,
        limit: int = 100,
    ) -> List[Video]:
        """
        Search videos by title or description.
        
        Args:
            db: Database session
            query_text: Search query
            skip: Number of records to skip
            limit: Maximum number of records to return
            
        Returns:
            List of matching video instances
        """
        from sqlalchemy import or_
        
        search_pattern = f"%{query_text}%"
        query = (
            select(self.model)
            .where(
                and_(
                    or_(
                        self.model.title.ilike(search_pattern),
                        self.model.description.ilike(search_pattern),
                    ),
                    self.model.visibility == VideoVisibility.PUBLIC,
                    self.model.status == VideoStatus.PROCESSED,
                )
            )
            .order_by(self.model.created_at.desc())
            .offset(skip)
            .limit(limit)
        )
        result = await db.execute(query)
        return list(result.scalars().all())

    async def get_user_video_count(
        self,
        db: AsyncSession,
        *,
        user_id: UUID,
        status: Optional[VideoStatus] = None,
    ) -> int:
        """
        Get count of user's videos.
        
        Args:
            db: Database session
            user_id: User ID
            status: Filter by video status
            
        Returns:
            Number of videos
        """
        query = select(func.count()).select_from(self.model).where(self.model.owner_id == user_id)
        
        if status:
            query = query.where(self.model.status == status)
        
        result = await db.execute(query)
        return result.scalar_one()


# Create singleton instance
video = CRUDVideo(Video)
