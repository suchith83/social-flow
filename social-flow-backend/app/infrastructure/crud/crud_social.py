"""
CRUD operations for social models (Post, Comment, Like, Follow, Save).
"""

from typing import List, Optional
from uuid import UUID

from sqlalchemy import select, func, and_, or_
from sqlalchemy.ext.asyncio import AsyncSession

from app.infrastructure.crud.base import CRUDBase
from app.models.social import Post, Comment, Like, Follow, Save
from app.schemas.social import (
    PostCreate,
    PostUpdate,
    CommentCreate,
    CommentUpdate,
    LikeCreate,
    FollowCreate,
    SaveCreate,
)


class CRUDPost(CRUDBase[Post, PostCreate, PostUpdate]):
    """CRUD operations for Post model."""

    async def create_with_owner(
        self,
        db: AsyncSession,
        *,
        obj_in: PostCreate,
        owner_id: UUID,
    ) -> Post:
        """
        Create a new post with owner.
        
        Args:
            db: Database session
            obj_in: Post creation schema
            owner_id: Owner user ID
            
        Returns:
            Created post instance
        """
        obj_in_data = obj_in.model_dump(exclude={"allow_comments", "allow_likes"})
        # Add owner_id
        obj_in_data["owner_id"] = owner_id
        
        # Map repost_of_id to original_post_id
        if "repost_of_id" in obj_in_data:
            obj_in_data["original_post_id"] = obj_in_data.pop("repost_of_id")
        
        # Remove fields that don't exist in model
        obj_in_data.pop("status", None)
        
        # Initialize counts
        obj_in_data.setdefault("like_count", 0)
        obj_in_data.setdefault("comment_count", 0)
        obj_in_data.setdefault("view_count", 0)
        obj_in_data.setdefault("repost_count", 0)
        obj_in_data.setdefault("share_count", 0)
        obj_in_data.setdefault("save_count", 0)
        
        # Initialize arrays
        obj_in_data.setdefault("hashtags", [])
        obj_in_data.setdefault("mentions", [])
        obj_in_data.setdefault("media_urls", obj_in_data.pop("images", []))
        
        db_obj = self.model(**obj_in_data)
        db.add(db_obj)
        await db.commit()
        await db.refresh(db_obj)
        return db_obj

    async def get_by_user(
        self,
        db: AsyncSession,
        *,
        user_id: UUID,
        skip: int = 0,
        limit: int = 100,
    ) -> List[Post]:
        """
        Get posts by user.
        
        Args:
            db: Database session
            user_id: User ID
            skip: Number of records to skip
            limit: Maximum number of records to return
            
        Returns:
            List of post instances
        """
        query = (
            select(self.model)
            .where(self.model.owner_id == user_id)
            .order_by(self.model.created_at.desc())
            .offset(skip)
            .limit(limit)
        )
        result = await db.execute(query)
        return list(result.scalars().all())

    async def get_feed(
        self,
        db: AsyncSession,
        *,
        user_id: UUID,
        skip: int = 0,
        limit: int = 100,
    ) -> List[Post]:
        """
        Get posts from followed users (user's feed).
        
        Args:
            db: Database session
            user_id: Current user ID
            skip: Number of records to skip
            limit: Maximum number of records to return
            
        Returns:
            List of post instances
        """
        query = (
            select(self.model)
            .join(Follow, Follow.following_id == self.model.owner_id)
            .where(Follow.follower_id == user_id)
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
    ) -> List[Post]:
        """
        Get trending posts based on engagement.
        
        Args:
            db: Database session
            skip: Number of records to skip
            limit: Maximum number of records to return
            days: Number of days to look back
            
        Returns:
            List of trending post instances
        """
        from datetime import datetime, timedelta, timezone
        
        since = datetime.now(timezone.utc) - timedelta(days=days)
        
        query = (
            select(self.model)
            .where(self.model.created_at >= since)
            .order_by(
                (self.model.like_count + self.model.comment_count * 2).desc()
            )
            .offset(skip)
            .limit(limit)
        )
        result = await db.execute(query)
        return list(result.scalars().all())

    async def increment_like_count(
        self,
        db: AsyncSession,
        *,
        post_id: UUID,
    ) -> Optional[Post]:
        """Increment post like count."""
        post = await self.get(db, post_id)
        if not post:
            return None
        
        post.like_count += 1
        db.add(post)
        await db.commit()
        await db.refresh(post)
        return post

    async def decrement_like_count(
        self,
        db: AsyncSession,
        *,
        post_id: UUID,
    ) -> Optional[Post]:
        """Decrement post like count."""
        post = await self.get(db, post_id)
        if not post:
            return None
        
        post.like_count = max(0, post.like_count - 1)
        db.add(post)
        await db.commit()
        await db.refresh(post)
        return post

    async def increment_comment_count(
        self,
        db: AsyncSession,
        *,
        post_id: UUID,
    ) -> Optional[Post]:
        """Increment post comment count."""
        post = await self.get(db, post_id)
        if not post:
            return None
        
        post.comment_count += 1
        db.add(post)
        await db.commit()
        await db.refresh(post)
        return post

    async def decrement_comment_count(
        self,
        db: AsyncSession,
        *,
        post_id: UUID,
    ) -> Optional[Post]:
        """Decrement post comment count."""
        post = await self.get(db, post_id)
        if not post:
            return None
        
        post.comment_count = max(0, post.comment_count - 1)
        db.add(post)
        await db.commit()
        await db.refresh(post)
        return post


class CRUDComment(CRUDBase[Comment, CommentCreate, CommentUpdate]):
    """CRUD operations for Comment model."""

    async def create_with_user(
        self,
        db: AsyncSession,
        *,
        obj_in: CommentCreate,
        user_id: UUID,
    ) -> Comment:
        """
        Create a new comment with user.
        
        Args:
            db: Database session
            obj_in: Comment creation schema
            user_id: Comment author user ID
            
        Returns:
            Created comment instance
        """
        obj_in_data = obj_in.model_dump()
        # Add user_id
        obj_in_data["user_id"] = user_id
        
        # Initialize counts
        obj_in_data.setdefault("like_count", 0)
        obj_in_data.setdefault("reply_count", 0)
        
        db_obj = self.model(**obj_in_data)
        db.add(db_obj)
        await db.commit()
        await db.refresh(db_obj)
        return db_obj

    async def get_by_post(
        self,
        db: AsyncSession,
        *,
        post_id: UUID,
        skip: int = 0,
        limit: int = 100,
    ) -> List[Comment]:
        """
        Get comments for a post.
        
        Args:
            db: Database session
            post_id: Post ID
            skip: Number of records to skip
            limit: Maximum number of records to return
            
        Returns:
            List of comment instances
        """
        query = (
            select(self.model)
            .where(
                and_(
                    self.model.post_id == post_id,
                    self.model.parent_comment_id.is_(None),
                )
            )
            .order_by(self.model.created_at.desc())
            .offset(skip)
            .limit(limit)
        )
        result = await db.execute(query)
        return list(result.scalars().all())

    async def get_by_video(
        self,
        db: AsyncSession,
        *,
        video_id: UUID,
        skip: int = 0,
        limit: int = 100,
    ) -> List[Comment]:
        """
        Get comments for a video.
        
        Args:
            db: Database session
            video_id: Video ID
            skip: Number of records to skip
            limit: Maximum number of records to return
            
        Returns:
            List of comment instances
        """
        query = (
            select(self.model)
            .where(
                and_(
                    self.model.video_id == video_id,
                    self.model.parent_comment_id.is_(None),
                )
            )
            .order_by(self.model.created_at.desc())
            .offset(skip)
            .limit(limit)
        )
        result = await db.execute(query)
        return list(result.scalars().all())

    async def get_replies(
        self,
        db: AsyncSession,
        *,
        parent_comment_id: UUID,
        skip: int = 0,
        limit: int = 100,
    ) -> List[Comment]:
        """
        Get replies to a comment.
        
        Args:
            db: Database session
            parent_comment_id: Parent comment ID
            skip: Number of records to skip
            limit: Maximum number of records to return
            
        Returns:
            List of reply comment instances
        """
        query = (
            select(self.model)
            .where(self.model.parent_comment_id == parent_comment_id)
            .order_by(self.model.created_at.asc())
            .offset(skip)
            .limit(limit)
        )
        result = await db.execute(query)
        return list(result.scalars().all())

    async def get_comment_count(
        self,
        db: AsyncSession,
        *,
        post_id: Optional[UUID] = None,
        video_id: Optional[UUID] = None,
    ) -> int:
        """Get comment count for post or video."""
        query = select(func.count()).select_from(self.model)
        
        if post_id:
            query = query.where(self.model.post_id == post_id)
        elif video_id:
            query = query.where(self.model.video_id == video_id)
        
        result = await db.execute(query)
        return result.scalar_one()

    async def increment_like_count(
        self,
        db: AsyncSession,
        *,
        comment_id: UUID,
    ) -> Optional[Comment]:
        """Increment comment like count."""
        comment = await self.get(db, comment_id)
        if not comment:
            return None
        
        comment.like_count += 1
        db.add(comment)
        await db.commit()
        await db.refresh(comment)
        return comment

    async def decrement_like_count(
        self,
        db: AsyncSession,
        *,
        comment_id: UUID,
    ) -> Optional[Comment]:
        """Decrement comment like count."""
        comment = await self.get(db, comment_id)
        if not comment:
            return None
        
        comment.like_count = max(0, comment.like_count - 1)
        db.add(comment)
        await db.commit()
        await db.refresh(comment)
        return comment


class CRUDLike(CRUDBase[Like, LikeCreate, LikeCreate]):
    """CRUD operations for Like model."""

    async def get_by_user_and_post(
        self,
        db: AsyncSession,
        *,
        user_id: UUID,
        post_id: UUID,
    ) -> Optional[Like]:
        """Get like for user and post."""
        query = select(self.model).where(
            and_(
                self.model.user_id == user_id,
                self.model.post_id == post_id,
            )
        )
        result = await db.execute(query)
        return result.scalar_one_or_none()

    async def get_by_user_and_video(
        self,
        db: AsyncSession,
        *,
        user_id: UUID,
        video_id: UUID,
    ) -> Optional[Like]:
        """Get like for user and video."""
        query = select(self.model).where(
            and_(
                self.model.user_id == user_id,
                self.model.video_id == video_id,
            )
        )
        result = await db.execute(query)
        return result.scalar_one_or_none()

    async def get_by_user_and_comment(
        self,
        db: AsyncSession,
        *,
        user_id: UUID,
        comment_id: UUID,
    ) -> Optional[Like]:
        """Get like for user and comment."""
        query = select(self.model).where(
            and_(
                self.model.user_id == user_id,
                self.model.comment_id == comment_id,
            )
        )
        result = await db.execute(query)
        return result.scalar_one_or_none()

    async def get_like_count(
        self,
        db: AsyncSession,
        *,
        post_id: Optional[UUID] = None,
        video_id: Optional[UUID] = None,
        comment_id: Optional[UUID] = None,
    ) -> int:
        """Get like count for post, video, or comment."""
        query = select(func.count()).select_from(self.model)
        
        if post_id:
            query = query.where(self.model.post_id == post_id)
        elif video_id:
            query = query.where(self.model.video_id == video_id)
        elif comment_id:
            query = query.where(self.model.comment_id == comment_id)
        
        result = await db.execute(query)
        return result.scalar_one()


class CRUDFollow(CRUDBase[Follow, FollowCreate, FollowCreate]):
    """CRUD operations for Follow model."""

    async def follow(
        self,
        db: AsyncSession,
        *,
        follower_id: UUID,
        following_id: UUID,
    ) -> Follow:
        """
        Create a follow relationship.
        
        Args:
            db: Database session
            follower_id: User who is following
            following_id: User being followed
            
        Returns:
            Created follow relationship
        """
        # Check if already following
        existing = await self.get_by_users(
            db,
            follower_id=follower_id,
            following_id=following_id,
        )
        if existing:
            return existing
        
        # Create new follow
        db_obj = self.model(follower_id=follower_id, following_id=following_id)
        db.add(db_obj)
        await db.commit()
        await db.refresh(db_obj)
        return db_obj

    async def create(
        self,
        db: AsyncSession,
        *,
        obj_in: FollowCreate,
        follower_id: UUID,
        commit: bool = True,
    ) -> Follow:
        """Create a follow relationship with follower_id."""
        from fastapi.encoders import jsonable_encoder
        
        obj_in_data = jsonable_encoder(obj_in)
        obj_in_data["follower_id"] = follower_id
        db_obj = self.model(**obj_in_data)
        db.add(db_obj)
        
        if commit:
            await db.commit()
            await db.refresh(db_obj)
        
        return db_obj

    async def get_by_users(
        self,
        db: AsyncSession,
        *,
        follower_id: UUID,
        following_id: UUID,
    ) -> Optional[Follow]:
        """Get follow relationship between two users."""
        query = select(self.model).where(
            and_(
                self.model.follower_id == follower_id,
                self.model.following_id == following_id,
            )
        )
        result = await db.execute(query)
        return result.scalar_one_or_none()

    async def is_following(
        self,
        db: AsyncSession,
        *,
        follower_id: UUID,
        following_id: UUID,
    ) -> bool:
        """Check if user is following another user."""
        follow = await self.get_by_users(
            db, follower_id=follower_id, following_id=following_id
        )
        return follow is not None


class CRUDSave(CRUDBase[Save, SaveCreate, SaveCreate]):
    """CRUD operations for Save model."""

    async def create_with_user(
        self,
        db: AsyncSession,
        *,
        obj_in: SaveCreate,
        user_id: UUID,
    ) -> Save:
        """Create a save with user_id."""
        obj_in_data = obj_in.model_dump()
        obj_in_data["user_id"] = user_id
        
        db_obj = self.model(**obj_in_data)
        db.add(db_obj)
        await db.commit()
        await db.refresh(db_obj)
        return db_obj

    async def get_by_user_and_post(
        self,
        db: AsyncSession,
        *,
        user_id: UUID,
        post_id: UUID,
    ) -> Optional[Save]:
        """Get save for user and post."""
        query = select(self.model).where(
            and_(
                self.model.user_id == user_id,
                self.model.post_id == post_id,
            )
        )
        result = await db.execute(query)
        return result.scalar_one_or_none()

    async def get_by_user_and_video(
        self,
        db: AsyncSession,
        *,
        user_id: UUID,
        video_id: UUID,
    ) -> Optional[Save]:
        """Get save for user and video."""
        query = select(self.model).where(
            and_(
                self.model.user_id == user_id,
                self.model.video_id == video_id,
            )
        )
        result = await db.execute(query)
        return result.scalar_one_or_none()

    async def get_saved_by_user(
        self,
        db: AsyncSession,
        *,
        user_id: UUID,
        skip: int = 0,
        limit: int = 100,
    ) -> List[Save]:
        """Get all saved items for a user."""
        query = (
            select(self.model)
            .where(self.model.user_id == user_id)
            .order_by(self.model.created_at.desc())
            .offset(skip)
            .limit(limit)
        )
        result = await db.execute(query)
        return list(result.scalars().all())


# Create singleton instances
post = CRUDPost(Post)
comment = CRUDComment(Comment)
like = CRUDLike(Like)
follow = CRUDFollow(Follow)
save = CRUDSave(Save)
