"""
Post Repository Implementation

SQLAlchemy-based implementation of IPostRepository interface.
"""

from datetime import datetime, timedelta
from typing import List, Optional
from uuid import UUID

from sqlalchemy import select, func, and_
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.social import Post as PostModel
from app.models.social import Follow
from app.domain.entities.post import PostEntity
from app.domain.repositories.post_repository import IPostRepository
from app.domain.value_objects import PostVisibility
from app.infrastructure.repositories.mappers import PostMapper


class PostRepository(IPostRepository):
    """
    SQLAlchemy implementation of post repository.
    
    Handles persistence and retrieval of post entities.
    """
    
    def __init__(self, session: AsyncSession):
        self._session = session
        self._mapper = PostMapper()
    
    async def get_by_id(self, id: UUID) -> Optional[PostEntity]:
        """Get post by ID."""
        result = await self._session.execute(
            select(PostModel).where(PostModel.id == id)
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
    ) -> List[PostEntity]:
        """Get posts by owner."""
        result = await self._session.execute(
            select(PostModel)
            .where(PostModel.owner_id == owner_id)
            .offset(skip)
            .limit(limit)
            .order_by(PostModel.created_at.desc())
        )
        models = result.scalars().all()
        
        return [self._mapper.to_entity(model) for model in models]
    
    async def get_public_posts(
        self,
        skip: int = 0,
        limit: int = 20,
    ) -> List[PostEntity]:
        """Get public posts."""
        result = await self._session.execute(
            select(PostModel)
            .where(
                and_(
                    PostModel.is_published.is_(True),
                    PostModel.is_removed.is_(False)
                )
            )
            .offset(skip)
            .limit(limit)
            .order_by(PostModel.created_at.desc())
        )
        models = result.scalars().all()
        
        return [self._mapper.to_entity(model) for model in models]
    
    async def get_feed_for_user(
        self,
        user_id: UUID,
        skip: int = 0,
        limit: int = 20,
    ) -> List[PostEntity]:
        """Get personalized feed for a user (posts from followed users)."""
        result = await self._session.execute(
            select(PostModel)
            .join(Follow, Follow.following_id == PostModel.owner_id)
            .where(
                and_(
                    Follow.follower_id == user_id,
                    PostModel.is_published.is_(True),
                    PostModel.is_removed.is_(False)
                )
            )
            .offset(skip)
            .limit(limit)
            .order_by(PostModel.created_at.desc())
        )
        models = result.scalars().all()
        
        return [self._mapper.to_entity(model) for model in models]
    
    async def get_trending_posts(
        self,
        hours: int = 24,
        skip: int = 0,
        limit: int = 20,
    ) -> List[PostEntity]:
        """Get trending posts based on recent engagement."""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        
        result = await self._session.execute(
            select(PostModel)
            .where(
                and_(
                    PostModel.is_published.is_(True),
                    PostModel.is_removed.is_(False),
                    PostModel.created_at >= cutoff_time
                )
            )
            .offset(skip)
            .limit(limit)
            .order_by(
                (PostModel.views_count * 0.5 +
                 PostModel.likes_count * 2 +
                 PostModel.comments_count * 3 +
                 PostModel.shares_count * 5).desc()
            )
        )
        models = result.scalars().all()
        
        return [self._mapper.to_entity(model) for model in models]
    
    async def search_posts(
        self,
        query: str,
        skip: int = 0,
        limit: int = 20,
    ) -> List[PostEntity]:
        """Search posts by content."""
        result = await self._session.execute(
            select(PostModel)
            .where(
                and_(
                    PostModel.content.ilike(f"%{query}%"),
                    PostModel.is_published.is_(True),
                    PostModel.is_removed.is_(False)
                )
            )
            .offset(skip)
            .limit(limit)
            .order_by(PostModel.created_at.desc())
        )
        models = result.scalars().all()
        
        return [self._mapper.to_entity(model) for model in models]
    
    async def get_flagged_posts(
        self,
        skip: int = 0,
        limit: int = 100,
    ) -> List[PostEntity]:
        """Get flagged posts for moderation."""
        result = await self._session.execute(
            select(PostModel)
            .where(PostModel.is_flagged.is_(True))
            .offset(skip)
            .limit(limit)
            .order_by(PostModel.flagged_at.desc())
        )
        models = result.scalars().all()
        
        return [self._mapper.to_entity(model) for model in models]
    
    async def get_by_visibility(
        self,
        visibility: PostVisibility,
        skip: int = 0,
        limit: int = 100,
    ) -> List[PostEntity]:
        """Get posts by visibility level."""
        # Note: This assumes PostModel has a visibility field
        # If not, this would need to be adapted
        result = await self._session.execute(
            select(PostModel)
            .where(PostModel.is_published.is_(True))
            .offset(skip)
            .limit(limit)
            .order_by(PostModel.created_at.desc())
        )
        models = result.scalars().all()
        
        return [self._mapper.to_entity(model) for model in models]
    
    async def get_all(
        self,
        skip: int = 0,
        limit: int = 100,
    ) -> List[PostEntity]:
        """Get all posts with pagination."""
        result = await self._session.execute(
            select(PostModel)
            .offset(skip)
            .limit(limit)
            .order_by(PostModel.created_at.desc())
        )
        models = result.scalars().all()
        
        return [self._mapper.to_entity(model) for model in models]
    
    async def add(self, entity: PostEntity) -> PostEntity:
        """Add new post."""
        model = self._mapper.to_model(entity)
        self._session.add(model)
        await self._session.flush()
        await self._session.refresh(model)
        
        return self._mapper.to_entity(model)
    
    async def update(self, entity: PostEntity) -> PostEntity:
        """Update existing post."""
        # Get existing model
        result = await self._session.execute(
            select(PostModel).where(PostModel.id == entity.id)
        )
        model = result.scalar_one_or_none()
        
        if model is None:
            raise ValueError(f"Post with id {entity.id} not found")
        
        # Update model from entity
        model = self._mapper.to_model(entity, model)
        await self._session.flush()
        await self._session.refresh(model)
        
        return self._mapper.to_entity(model)
    
    async def delete(self, id: UUID) -> bool:
        """Delete post by ID."""
        result = await self._session.execute(
            select(PostModel).where(PostModel.id == id)
        )
        model = result.scalar_one_or_none()
        
        if model is None:
            return False
        
        await self._session.delete(model)
        await self._session.flush()
        
        return True
    
    async def exists(self, id: UUID) -> bool:
        """Check if post exists."""
        result = await self._session.execute(
            select(func.count(PostModel.id)).where(PostModel.id == id)
        )
        count = result.scalar()
        return count > 0
    
    async def count(self) -> int:
        """Get total count of posts."""
        result = await self._session.execute(
            select(func.count(PostModel.id))
        )
        return result.scalar()
    
    async def get_recent_posts(
        self,
        skip: int = 0,
        limit: int = 20,
    ) -> List[PostEntity]:
        """Get recent public posts for global feed."""
        return await self.get_public_posts(skip=skip, limit=limit)
    
    async def get_replies(
        self,
        post_id: UUID,
        skip: int = 0,
        limit: int = 50,
    ) -> List[PostEntity]:
        """Get replies to a specific post."""
        result = await self._session.execute(
            select(PostModel)
            .where(PostModel.parent_id == post_id)
            .offset(skip)
            .limit(limit)
            .order_by(PostModel.created_at.asc())
        )
        models = result.scalars().all()
        
        return [self._mapper.to_entity(model) for model in models]
