"""
Comment service for managing comments and replies.

This module provides comprehensive comment management including CRUD operations,
reply handling, moderation, and engagement tracking.
"""

import json
import logging
from datetime import datetime
from typing import List, Optional
from uuid import UUID

from sqlalchemy import and_, desc, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.core.exceptions import NotFoundError, ValidationError
from app.posts.models.comment import Comment
from app.posts.models.post import Post
from app.posts.models.like import Like
from app.posts.schemas.comment import CommentCreate, CommentUpdate

logger = logging.getLogger(__name__)


class CommentService:
    """Service for comment management and reply handling."""

    def __init__(self, db: AsyncSession):
        """Initialize comment service."""
        self.db = db

    async def create_comment(
        self,
        user_id: UUID,
        comment_data: CommentCreate,
    ) -> Comment:
        """
        Create a new comment or reply.

        Args:
            user_id: ID of the user creating the comment
            comment_data: Comment creation data

        Returns:
            Created comment object

        Raises:
            ValidationError: If comment content is invalid
            NotFoundError: If parent comment or post/video not found
        """
        # Validate content length
        if len(comment_data.content) < 1:
            raise ValidationError("Comment content cannot be empty")

        if len(comment_data.content) > 1000:
            raise ValidationError("Comment content cannot exceed 1000 characters")

        # Validate parent entities exist
        if comment_data.post_id:
            post = await self._get_post(comment_data.post_id)
            if not post:
                raise NotFoundError(f"Post {comment_data.post_id} not found")
        elif comment_data.video_id:
            # TODO: Add video validation when video service is available
            pass
        else:
            raise ValidationError("Comment must be associated with a post or video")

        # If this is a reply, validate parent comment exists
        if comment_data.parent_comment_id:
            parent_comment = await self.get_comment(comment_data.parent_comment_id)
            if not parent_comment:
                raise NotFoundError(f"Parent comment {comment_data.parent_comment_id} not found")

            # Ensure parent comment is on the same post/video
            if comment_data.post_id and parent_comment.post_id != comment_data.post_id:
                raise ValidationError("Reply must be on the same post as parent comment")
            if comment_data.video_id and parent_comment.video_id != comment_data.video_id:
                raise ValidationError("Reply must be on the same video as parent comment")

        # Extract hashtags and mentions
        hashtags = self._extract_hashtags(comment_data.content)
        mentions = self._extract_mentions(comment_data.content)

        # Create comment
        comment = Comment(
            owner_id=user_id,
            content=comment_data.content,
            hashtags=json.dumps(hashtags),
            mentions=json.dumps(mentions),
            post_id=comment_data.post_id,
            video_id=comment_data.video_id,
            parent_comment_id=comment_data.parent_comment_id,
            is_reply=comment_data.parent_comment_id is not None,
            is_approved=True,  # Auto-approve for now, add moderation later
        )

        self.db.add(comment)
        await self.db.commit()
        await self.db.refresh(comment)

        # Update parent comment reply count if this is a reply
        if comment_data.parent_comment_id:
            await self._increment_reply_count(comment_data.parent_comment_id)

        # Update post/video comment count
        if comment_data.post_id:
            await self._increment_post_comment_count(comment_data.post_id)
        elif comment_data.video_id:
            # TODO: Update video comment count when video service is available
            pass

        # Trigger ML moderation (async)
        from app.ml.ml_tasks import moderate_comment_task
        moderate_comment_task.apply_async(args=[str(comment.id)])

        logger.info(f"Comment created: {comment.id} by user {user_id}")

        return comment

    async def get_comment(self, comment_id: UUID) -> Optional[Comment]:
        """
        Get comment by ID.

        Args:
            comment_id: Comment ID

        Returns:
            Comment object or None if not found
        """
        query = select(Comment).where(Comment.id == comment_id).options(
            selectinload(Comment.owner),
            selectinload(Comment.post),
            selectinload(Comment.likes),
        )

        result = await self.db.execute(query)
        return result.scalar_one_or_none()

    async def update_comment(
        self,
        comment_id: UUID,
        user_id: UUID,
        comment_data: CommentUpdate,
    ) -> Comment:
        """
        Update an existing comment.

        Args:
            comment_id: Comment ID
            user_id: ID of user updating the comment
            comment_data: Comment update data

        Returns:
            Updated comment object

        Raises:
            NotFoundError: If comment not found
            ValidationError: If user doesn't own the comment
        """
        comment = await self.get_comment(comment_id)

        if not comment:
            raise NotFoundError(f"Comment {comment_id} not found")

        if comment.owner_id != user_id:
            raise ValidationError("You can only edit your own comments")

        # Update fields
        if comment_data.content is not None:
            if len(comment_data.content) < 1 or len(comment_data.content) > 1000:
                raise ValidationError("Invalid content length")

            comment.content = comment_data.content
            comment.hashtags = json.dumps(self._extract_hashtags(comment_data.content))
            comment.mentions = json.dumps(self._extract_mentions(comment_data.content))

        comment.updated_at = datetime.utcnow()

        await self.db.commit()
        await self.db.refresh(comment)

        logger.info(f"Comment updated: {comment.id} by user {user_id}")

        return comment

    async def delete_comment(
        self,
        comment_id: UUID,
        user_id: UUID,
    ) -> None:
        """
        Delete a comment.

        Args:
            comment_id: Comment ID
            user_id: ID of user deleting the comment

        Raises:
            NotFoundError: If comment not found
            ValidationError: If user doesn't own the comment
        """
        comment = await self.get_comment(comment_id)

        if not comment:
            raise NotFoundError(f"Comment {comment_id} not found")

        if comment.owner_id != user_id:
            raise ValidationError("You can only delete your own comments")

        # Update counts before deletion
        if comment.post_id:
            await self._decrement_post_comment_count(comment.post_id)
        elif comment.video_id:
            # TODO: Update video comment count when video service is available
            pass

        if comment.parent_comment_id:
            await self._decrement_reply_count(comment.parent_comment_id)

        # Delete the comment (cascade will handle likes)
        await self.db.delete(comment)
        await self.db.commit()

        logger.info(f"Comment deleted: {comment_id} by user {user_id}")

    async def get_post_comments(
        self,
        post_id: UUID,
        skip: int = 0,
        limit: int = 20,
        include_replies: bool = True,
    ) -> List[Comment]:
        """
        Get comments for a specific post.

        Args:
            post_id: Post ID
            skip: Number of comments to skip
            limit: Maximum number of comments to return
            include_replies: Whether to include replies

        Returns:
            List of comment objects
        """
        if include_replies:
            # Get top-level comments with their replies
            query = (
                select(Comment)
                .where(
                    and_(
                        Comment.post_id == post_id,
                        not Comment.is_reply
                    )
                )
                .options(
                    selectinload(Comment.owner),
                    selectinload(Comment.likes),
                    selectinload(Comment.replies).selectinload(Comment.owner),
                    selectinload(Comment.replies).selectinload(Comment.likes),
                )
                .order_by(desc(Comment.created_at))
                .offset(skip)
                .limit(limit)
            )
        else:
            # Get only top-level comments
            query = (
                select(Comment)
                .where(
                    and_(
                        Comment.post_id == post_id,
                        not Comment.is_reply
                    )
                )
                .options(
                    selectinload(Comment.owner),
                    selectinload(Comment.likes),
                )
                .order_by(desc(Comment.created_at))
                .offset(skip)
                .limit(limit)
            )

        result = await self.db.execute(query)
        return result.scalars().all()

    async def get_comment_replies(
        self,
        comment_id: UUID,
        skip: int = 0,
        limit: int = 20,
    ) -> List[Comment]:
        """
        Get replies for a specific comment.

        Args:
            comment_id: Parent comment ID
            skip: Number of replies to skip
            limit: Maximum number of replies to return

        Returns:
            List of reply comment objects
        """
        query = (
            select(Comment)
            .where(Comment.parent_comment_id == comment_id)
            .options(
                selectinload(Comment.owner),
                selectinload(Comment.likes),
            )
            .order_by(Comment.created_at)
            .offset(skip)
            .limit(limit)
        )

        result = await self.db.execute(query)
        return result.scalars().all()

    async def like_comment(
        self,
        user_id: UUID,
        comment_id: UUID,
    ) -> None:
        """
        Like a comment.

        Args:
            user_id: ID of the user liking the comment
            comment_id: Comment ID

        Raises:
            NotFoundError: If comment not found
            ValidationError: If user already liked the comment
        """
        comment = await self.get_comment(comment_id)

        if not comment:
            raise NotFoundError(f"Comment {comment_id} not found")

        # Check if user already liked this comment
        existing_like = await self._get_comment_like(user_id, comment_id)
        if existing_like:
            raise ValidationError("You have already liked this comment")

        # Create like
        like = Like(
            user_id=user_id,
            comment_id=comment_id,
            like_type="comment",
            is_like=True,
        )

        self.db.add(like)
        comment.likes_count += 1

        await self.db.commit()

        logger.info(f"Comment liked: {comment_id} by user {user_id}")

    async def unlike_comment(
        self,
        user_id: UUID,
        comment_id: UUID,
    ) -> None:
        """
        Unlike a comment.

        Args:
            user_id: ID of the user unliking the comment
            comment_id: Comment ID

        Raises:
            NotFoundError: If comment not found or like not found
        """
        comment = await self.get_comment(comment_id)

        if not comment:
            raise NotFoundError(f"Comment {comment_id} not found")

        # Find and delete the like
        like = await self._get_comment_like(user_id, comment_id)
        if not like:
            raise NotFoundError("Like not found")

        await self.db.delete(like)
        comment.likes_count -= 1

        await self.db.commit()

        logger.info(f"Comment unliked: {comment_id} by user {user_id}")

    # Helper methods

    def _extract_hashtags(self, content: str) -> List[str]:
        """Extract hashtags from content."""
        import re
        hashtags = re.findall(r'#(\w+)', content)
        return list(set(hashtags))  # Remove duplicates

    def _extract_mentions(self, content: str) -> List[str]:
        """Extract mentions from content."""
        import re
        mentions = re.findall(r'@(\w+)', content)
        return list(set(mentions))  # Remove duplicates

    async def _get_post(self, post_id: UUID) -> Optional[Post]:
        """Get post by ID."""
        query = select(Post).where(Post.id == post_id)
        result = await self.db.execute(query)
        return result.scalar_one_or_none()

    async def _increment_reply_count(self, comment_id: UUID) -> None:
        """Increment reply count for a comment."""
        query = select(Comment).where(Comment.id == comment_id)
        result = await self.db.execute(query)
        comment = result.scalar_one_or_none()

        if comment:
            comment.replies_count += 1
            await self.db.commit()

    async def _decrement_reply_count(self, comment_id: UUID) -> None:
        """Decrement reply count for a comment."""
        query = select(Comment).where(Comment.id == comment_id)
        result = await self.db.execute(query)
        comment = result.scalar_one_or_none()

        if comment and comment.replies_count > 0:
            comment.replies_count -= 1
            await self.db.commit()

    async def _increment_post_comment_count(self, post_id: UUID) -> None:
        """Increment comment count for a post."""
        query = select(Post).where(Post.id == post_id)
        result = await self.db.execute(query)
        post = result.scalar_one_or_none()

        if post:
            post.comments_count += 1
            await self.db.commit()

    async def _decrement_post_comment_count(self, post_id: UUID) -> None:
        """Decrement comment count for a post."""
        query = select(Post).where(Post.id == post_id)
        result = await self.db.execute(query)
        post = result.scalar_one_or_none()

        if post and post.comments_count > 0:
            post.comments_count -= 1
            await self.db.commit()

    async def _get_comment_like(self, user_id: UUID, comment_id: UUID) -> Optional[Like]:
        """Get a user's like for a comment."""
        query = select(Like).where(
            and_(
                Like.user_id == user_id,
                Like.comment_id == comment_id,
                Like.like_type == "comment"
            )
        )
        result = await self.db.execute(query)
        return result.scalar_one_or_none()