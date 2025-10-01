"""
Like service for managing likes on posts, videos, and comments.

This module provides comprehensive like management including like/unlike operations,
validation, and engagement tracking.
"""

import logging
from typing import List, Optional
from uuid import UUID

from sqlalchemy import and_, desc, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import NotFoundError, ValidationError
from app.posts.models.like import Like
from app.posts.models.post import Post
from app.posts.models.comment import Comment
from app.posts.schemas.like import LikeCreate, LikeUpdate

logger = logging.getLogger(__name__)


class LikeService:
    """Service for like management and engagement tracking."""

    def __init__(self, db: AsyncSession):
        """Initialize like service."""
        self.db = db

    async def create_like(
        self,
        user_id: UUID,
        like_data: LikeCreate,
    ) -> Like:
        """
        Create a new like.

        Args:
            user_id: ID of the user creating the like
            like_data: Like creation data

        Returns:
            Created like object

        Raises:
            ValidationError: If like target is invalid or user already liked
            NotFoundError: If target entity not found
        """
        # Validate target exists and user hasn't already liked
        if like_data.post_id:
            await self._validate_post_like(user_id, like_data.post_id)
        elif like_data.video_id:
            await self._validate_video_like(user_id, like_data.video_id)
        elif like_data.comment_id:
            await self._validate_comment_like(user_id, like_data.comment_id)
        else:
            raise ValidationError("Like must target a post, video, or comment")

        # Create like
        like = Like(
            user_id=user_id,
            post_id=like_data.post_id,
            video_id=like_data.video_id,
            comment_id=like_data.comment_id,
            like_type=like_data.like_type,
            is_like=like_data.is_like,
        )

        self.db.add(like)

        # Update engagement counts
        await self._increment_engagement_count(like_data)

        await self.db.commit()
        await self.db.refresh(like)

        logger.info(f"Like created: {like.id} by user {user_id} on {like_data.like_type}")

        return like

    async def update_like(
        self,
        user_id: UUID,
        like_data: LikeUpdate,
    ) -> Like:
        """
        Update an existing like (change from like to dislike or vice versa).

        Args:
            user_id: ID of the user updating the like
            like_data: Like update data

        Returns:
            Updated like object

        Raises:
            NotFoundError: If like not found
            ValidationError: If user doesn't own the like
        """
        like = await self._get_user_like(user_id, like_data)

        if not like:
            raise NotFoundError("Like not found")

        # Update the like status
        old_is_like = like.is_like
        like.is_like = like_data.is_like

        # Update engagement counts if status changed
        if old_is_like != like_data.is_like:
            if like_data.is_like:
                await self._increment_engagement_count(like_data)
            else:
                await self._decrement_engagement_count(like_data)

        await self.db.commit()
        await self.db.refresh(like)

        logger.info(f"Like updated: {like.id} by user {user_id}")

        return like

    async def delete_like(
        self,
        user_id: UUID,
        like_data: LikeUpdate,
    ) -> None:
        """
        Delete a like.

        Args:
            user_id: ID of the user deleting the like
            like_data: Like identification data

        Raises:
            NotFoundError: If like not found
        """
        like = await self._get_user_like(user_id, like_data)

        if not like:
            raise NotFoundError("Like not found")

        # Update engagement counts
        await self._decrement_engagement_count(like_data)

        await self.db.delete(like)
        await self.db.commit()

        logger.info(f"Like deleted: {like.id} by user {user_id}")

    async def get_user_likes(
        self,
        user_id: UUID,
        like_type: Optional[str] = None,
        skip: int = 0,
        limit: int = 20,
    ) -> List[Like]:
        """
        Get likes by a specific user.

        Args:
            user_id: User ID
            like_type: Optional filter by like type (post, video, comment)
            skip: Number of likes to skip
            limit: Maximum number of likes to return

        Returns:
            List of like objects
        """
        query = select(Like).where(Like.user_id == user_id)

        if like_type:
            query = query.where(Like.like_type == like_type)

        query = query.order_by(desc(Like.created_at)).offset(skip).limit(limit)

        result = await self.db.execute(query)
        return result.scalars().all()

    async def get_entity_likes(
        self,
        entity_id: UUID,
        entity_type: str,
        skip: int = 0,
        limit: int = 20,
    ) -> List[Like]:
        """
        Get likes for a specific entity (post, video, or comment).

        Args:
            entity_id: Entity ID
            entity_type: Entity type (post, video, comment)
            skip: Number of likes to skip
            limit: Maximum number of likes to return

        Returns:
            List of like objects
        """
        if entity_type == "post":
            query = select(Like).where(Like.post_id == entity_id)
        elif entity_type == "video":
            query = select(Like).where(Like.video_id == entity_id)
        elif entity_type == "comment":
            query = select(Like).where(Like.comment_id == entity_id)
        else:
            raise ValidationError("Invalid entity type")

        query = query.order_by(desc(Like.created_at)).offset(skip).limit(limit)

        result = await self.db.execute(query)
        return result.scalars().all()

    async def check_user_like(
        self,
        user_id: UUID,
        like_data: LikeUpdate,
    ) -> Optional[Like]:
        """
        Check if a user has liked a specific entity.

        Args:
            user_id: User ID
            like_data: Like identification data

        Returns:
            Like object if exists, None otherwise
        """
        return await self._get_user_like(user_id, like_data)

    # Helper methods

    async def _validate_post_like(self, user_id: UUID, post_id: UUID) -> None:
        """Validate post exists and user hasn't liked it."""
        post = await self._get_post(post_id)
        if not post:
            raise NotFoundError(f"Post {post_id} not found")

        existing_like = await self._get_user_post_like(user_id, post_id)
        if existing_like:
            raise ValidationError("You have already liked this post")

    async def _validate_video_like(self, user_id: UUID, video_id: UUID) -> None:
        """Validate video exists and user hasn't liked it."""
        # TODO: Add video validation when video service is available
        existing_like = await self._get_user_video_like(user_id, video_id)
        if existing_like:
            raise ValidationError("You have already liked this video")

    async def _validate_comment_like(self, user_id: UUID, comment_id: UUID) -> None:
        """Validate comment exists and user hasn't liked it."""
        comment = await self._get_comment(comment_id)
        if not comment:
            raise NotFoundError(f"Comment {comment_id} not found")

        existing_like = await self._get_user_comment_like(user_id, comment_id)
        if existing_like:
            raise ValidationError("You have already liked this comment")

    async def _increment_engagement_count(self, like_data: LikeCreate) -> None:
        """Increment engagement count for the liked entity."""
        if like_data.post_id:
            await self._increment_post_likes(like_data.post_id)
        elif like_data.video_id:
            # TODO: Increment video likes when video service is available
            pass
        elif like_data.comment_id:
            await self._increment_comment_likes(like_data.comment_id)

    async def _decrement_engagement_count(self, like_data: LikeUpdate) -> None:
        """Decrement engagement count for the unliked entity."""
        if like_data.post_id:
            await self._decrement_post_likes(like_data.post_id)
        elif like_data.video_id:
            # TODO: Decrement video likes when video service is available
            pass
        elif like_data.comment_id:
            await self._decrement_comment_likes(like_data.comment_id)

    async def _get_user_like(self, user_id: UUID, like_data: LikeUpdate) -> Optional[Like]:
        """Get a user's like for a specific entity."""
        query = select(Like).where(Like.user_id == user_id)

        if like_data.post_id:
            query = query.where(Like.post_id == like_data.post_id)
        elif like_data.video_id:
            query = query.where(Like.video_id == like_data.video_id)
        elif like_data.comment_id:
            query = query.where(Like.comment_id == like_data.comment_id)

        result = await self.db.execute(query)
        return result.scalar_one_or_none()

    async def _get_user_post_like(self, user_id: UUID, post_id: UUID) -> Optional[Like]:
        """Get a user's like for a specific post."""
        query = select(Like).where(
            and_(
                Like.user_id == user_id,
                Like.post_id == post_id,
                Like.like_type == "post"
            )
        )
        result = await self.db.execute(query)
        return result.scalar_one_or_none()

    async def _get_user_video_like(self, user_id: UUID, video_id: UUID) -> Optional[Like]:
        """Get a user's like for a specific video."""
        query = select(Like).where(
            and_(
                Like.user_id == user_id,
                Like.video_id == video_id,
                Like.like_type == "video"
            )
        )
        result = await self.db.execute(query)
        return result.scalar_one_or_none()

    async def _get_user_comment_like(self, user_id: UUID, comment_id: UUID) -> Optional[Like]:
        """Get a user's like for a specific comment."""
        query = select(Like).where(
            and_(
                Like.user_id == user_id,
                Like.comment_id == comment_id,
                Like.like_type == "comment"
            )
        )
        result = await self.db.execute(query)
        return result.scalar_one_or_none()

    async def _get_post(self, post_id: UUID) -> Optional[Post]:
        """Get post by ID."""
        query = select(Post).where(Post.id == post_id)
        result = await self.db.execute(query)
        return result.scalar_one_or_none()

    async def _get_comment(self, comment_id: UUID) -> Optional[Comment]:
        """Get comment by ID."""
        query = select(Comment).where(Comment.id == comment_id)
        result = await self.db.execute(query)
        return result.scalar_one_or_none()

    async def _increment_post_likes(self, post_id: UUID) -> None:
        """Increment likes count for a post."""
        query = select(Post).where(Post.id == post_id)
        result = await self.db.execute(query)
        post = result.scalar_one_or_none()

        if post:
            post.likes_count += 1
            await self.db.commit()

    async def _decrement_post_likes(self, post_id: UUID) -> None:
        """Decrement likes count for a post."""
        query = select(Post).where(Post.id == post_id)
        result = await self.db.execute(query)
        post = result.scalar_one_or_none()

        if post and post.likes_count > 0:
            post.likes_count -= 1
            await self.db.commit()

    async def _increment_comment_likes(self, comment_id: UUID) -> None:
        """Increment likes count for a comment."""
        query = select(Comment).where(Comment.id == comment_id)
        result = await self.db.execute(query)
        comment = result.scalar_one_or_none()

        if comment:
            comment.likes_count += 1
            await self.db.commit()

    async def _decrement_comment_likes(self, comment_id: UUID) -> None:
        """Decrement likes count for a comment."""
        query = select(Comment).where(Comment.id == comment_id)
        result = await self.db.execute(query)
        comment = result.scalar_one_or_none()

        if comment and comment.likes_count > 0:
            comment.likes_count -= 1
            await self.db.commit()