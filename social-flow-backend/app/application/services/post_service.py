"""
Post Application Service

Orchestrates post-related use cases and workflows.
"""

import logging
from typing import List, Optional
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from app.domain.entities.post import PostEntity
from app.domain.repositories.post_repository import IPostRepository
from app.infrastructure.repositories import PostRepository

logger = logging.getLogger(__name__)


class PostApplicationService:
    """
    Post application service for post management use cases.
    
    Handles:
    - Post creation and publishing
    - Feed generation
    - Trending content discovery
    - Social interactions (like, comment, share)
    - Content moderation
    """
    
    def __init__(self, session: AsyncSession):
        self._session = session
        self._post_repo: IPostRepository = PostRepository(session)
    
    # Post Creation
    
    async def create_post(
        self,
        user_id: UUID,
        content: str,
        media_url: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> PostEntity:
        """
        Create new post.
        
        Args:
            user_id: Author user ID
            content: Post content
            media_url: Optional media URL
            tags: Optional list of tags
            
        Returns:
            Created post entity
            
        Raises:
            ValueError: If validation fails
        """
        # Create post entity
        post = PostEntity(
            user_id=user_id,
            content=content,
            media_url=media_url,
        )
        
        # Add tags if provided
        if tags:
            for tag in tags:
                post.add_tag(tag)
        
        # Save
        saved_post = await self._post_repo.add(post)
        await self._session.commit()
        
        logger.info(f"Post created: {saved_post.id} by user {user_id}")
        
        return saved_post
    
    async def create_reply(
        self,
        user_id: UUID,
        parent_post_id: UUID,
        content: str,
        media_url: Optional[str] = None,
    ) -> PostEntity:
        """
        Create reply to another post.
        
        Args:
            user_id: Author user ID
            parent_post_id: Parent post ID
            content: Reply content
            media_url: Optional media URL
            
        Returns:
            Created reply post entity
            
        Raises:
            ValueError: If parent post not found or validation fails
        """
        # Verify parent exists
        parent = await self._post_repo.get_by_id(parent_post_id)
        if parent is None:
            raise ValueError(f"Parent post {parent_post_id} not found")
        
        # Create reply
        reply = PostEntity(
            user_id=user_id,
            content=content,
            media_url=media_url,
            parent_post_id=parent_post_id,
        )
        
        # Save
        saved_reply = await self._post_repo.add(reply)
        
        # Increment parent comment count
        parent.increment_comments()
        await self._post_repo.update(parent)
        
        await self._session.commit()
        
        logger.info(f"Reply created: {saved_reply.id} to post {parent_post_id}")
        
        return saved_reply
    
    # Post Management
    
    async def get_post_by_id(self, post_id: UUID) -> Optional[PostEntity]:
        """Get post by ID."""
        return await self._post_repo.get_by_id(post_id)
    
    async def get_user_posts(
        self,
        user_id: UUID,
        skip: int = 0,
        limit: int = 20,
    ) -> List[PostEntity]:
        """Get posts by user."""
        return await self._post_repo.get_by_owner(user_id, skip, limit)
    
    async def get_post_replies(
        self,
        post_id: UUID,
        skip: int = 0,
        limit: int = 50,
    ) -> List[PostEntity]:
        """Get replies to a post."""
        return await self._post_repo.get_replies(post_id, skip, limit)
    
    async def update_post_content(
        self,
        post_id: UUID,
        content: str,
    ) -> PostEntity:
        """
        Update post content.
        
        Args:
            post_id: Post ID
            content: New content
            
        Returns:
            Updated post entity
            
        Raises:
            ValueError: If post not found or validation fails
        """
        post = await self._post_repo.get_by_id(post_id)
        if post is None:
            raise ValueError(f"Post {post_id} not found")
        
        post.update_content(content)
        
        updated_post = await self._post_repo.update(post)
        await self._session.commit()
        
        logger.info(f"Post content updated: {post_id}")
        
        return updated_post
    
    async def delete_post(self, post_id: UUID) -> bool:
        """
        Delete post.
        
        Args:
            post_id: Post ID
            
        Returns:
            True if deleted
            
        Raises:
            ValueError: If post not found
        """
        post = await self._post_repo.get_by_id(post_id)
        if post is None:
            raise ValueError(f"Post {post_id} not found")
        
        await self._post_repo.delete(post_id)
        await self._session.commit()
        
        logger.info(f"Post deleted: {post_id}")
        
        return True
    
    # Engagement
    
    async def like_post(self, post_id: UUID) -> PostEntity:
        """
        Like post (increment likes).
        
        Args:
            post_id: Post ID
            
        Returns:
            Updated post entity
            
        Raises:
            ValueError: If post not found
        """
        post = await self._post_repo.get_by_id(post_id)
        if post is None:
            raise ValueError(f"Post {post_id} not found")
        
        post.increment_likes()
        
        updated_post = await self._post_repo.update(post)
        await self._session.commit()
        
        return updated_post
    
    async def unlike_post(self, post_id: UUID) -> PostEntity:
        """
        Unlike post (decrement likes).
        
        Args:
            post_id: Post ID
            
        Returns:
            Updated post entity
            
        Raises:
            ValueError: If post not found
        """
        post = await self._post_repo.get_by_id(post_id)
        if post is None:
            raise ValueError(f"Post {post_id} not found")
        
        post.decrement_likes()
        
        updated_post = await self._post_repo.update(post)
        await self._session.commit()
        
        return updated_post
    
    async def share_post(self, post_id: UUID) -> PostEntity:
        """
        Share post (increment shares).
        
        Args:
            post_id: Post ID
            
        Returns:
            Updated post entity
            
        Raises:
            ValueError: If post not found
        """
        post = await self._post_repo.get_by_id(post_id)
        if post is None:
            raise ValueError(f"Post {post_id} not found")
        
        post.increment_shares()
        
        updated_post = await self._post_repo.update(post)
        await self._session.commit()
        
        return updated_post
    
    async def record_post_impression(self, post_id: UUID) -> PostEntity:
        """
        Record post impression (view).
        
        Args:
            post_id: Post ID
            
        Returns:
            Updated post entity
            
        Raises:
            ValueError: If post not found
        """
        post = await self._post_repo.get_by_id(post_id)
        if post is None:
            raise ValueError(f"Post {post_id} not found")
        
        post.record_impression()
        
        updated_post = await self._post_repo.update(post)
        await self._session.commit()
        
        return updated_post
    
    # Feed Generation
    
    async def get_user_feed(
        self,
        user_id: UUID,
        skip: int = 0,
        limit: int = 20,
    ) -> List[PostEntity]:
        """
        Get personalized feed for user (posts from followed users).
        
        Args:
            user_id: User ID
            skip: Number of posts to skip
            limit: Maximum posts to return
            
        Returns:
            List of posts for user's feed
        """
        return await self._post_repo.get_feed_for_user(user_id, skip, limit)
    
    async def get_global_feed(
        self,
        skip: int = 0,
        limit: int = 20,
    ) -> List[PostEntity]:
        """
        Get global feed (recent public posts).
        
        Args:
            skip: Number of posts to skip
            limit: Maximum posts to return
            
        Returns:
            List of recent public posts
        """
        return await self._post_repo.get_recent_posts(skip, limit)
    
    # Discovery
    
    async def get_trending_posts(
        self,
        days: int = 7,
        skip: int = 0,
        limit: int = 20,
    ) -> List[PostEntity]:
        """
        Get trending posts.
        
        Args:
            days: Number of days to look back
            skip: Number of posts to skip
            limit: Maximum posts to return
            
        Returns:
            List of trending posts
        """
        return await self._post_repo.get_trending(days, skip, limit)
    
    async def get_popular_posts(
        self,
        days: int = 30,
        skip: int = 0,
        limit: int = 20,
    ) -> List[PostEntity]:
        """
        Get popular posts by engagement.
        
        Args:
            days: Number of days to look back
            skip: Number of posts to skip
            limit: Maximum posts to return
            
        Returns:
            List of popular posts
        """
        return await self._post_repo.get_popular(days, skip, limit)
    
    async def search_posts(
        self,
        query: str,
        skip: int = 0,
        limit: int = 20,
    ) -> List[PostEntity]:
        """
        Search posts by content.
        
        Args:
            query: Search query
            skip: Number of posts to skip
            limit: Maximum posts to return
            
        Returns:
            List of matching posts
        """
        return await self._post_repo.search_by_content(query, skip, limit)
    
    async def get_posts_by_tag(
        self,
        tag: str,
        skip: int = 0,
        limit: int = 20,
    ) -> List[PostEntity]:
        """
        Get posts by tag.
        
        Args:
            tag: Tag to filter by
            skip: Number of posts to skip
            limit: Maximum posts to return
            
        Returns:
            List of posts with the tag
        """
        return await self._post_repo.get_by_tag(tag, skip, limit)
    
    # Moderation
    
    async def flag_post_for_review(
        self,
        post_id: UUID,
        reason: str,
    ) -> PostEntity:
        """
        Flag post for moderation review.
        
        Args:
            post_id: Post ID
            reason: Flagging reason
            
        Returns:
            Updated post entity
            
        Raises:
            ValueError: If post not found
        """
        post = await self._post_repo.get_by_id(post_id)
        if post is None:
            raise ValueError(f"Post {post_id} not found")
        
        post.flag_for_review(reason)
        
        updated_post = await self._post_repo.update(post)
        await self._session.commit()
        
        logger.warning(f"Post flagged: {post_id} - {reason}")
        
        return updated_post
    
    async def approve_post(self, post_id: UUID) -> PostEntity:
        """
        Approve post after review.
        
        Args:
            post_id: Post ID
            
        Returns:
            Updated post entity
            
        Raises:
            ValueError: If post not found
        """
        post = await self._post_repo.get_by_id(post_id)
        if post is None:
            raise ValueError(f"Post {post_id} not found")
        
        post.approve()
        
        updated_post = await self._post_repo.update(post)
        await self._session.commit()
        
        logger.info(f"Post approved: {post_id}")
        
        return updated_post
    
    async def reject_post(
        self,
        post_id: UUID,
        reason: str,
    ) -> PostEntity:
        """
        Reject post.
        
        Args:
            post_id: Post ID
            reason: Rejection reason
            
        Returns:
            Updated post entity
            
        Raises:
            ValueError: If post not found
        """
        post = await self._post_repo.get_by_id(post_id)
        if post is None:
            raise ValueError(f"Post {post_id} not found")
        
        post.reject(reason)
        
        updated_post = await self._post_repo.update(post)
        await self._session.commit()
        
        logger.warning(f"Post rejected: {post_id} - {reason}")
        
        return updated_post
    
    async def get_flagged_posts(
        self,
        skip: int = 0,
        limit: int = 50,
    ) -> List[PostEntity]:
        """Get posts flagged for review."""
        return await self._post_repo.get_flagged_for_review(skip, limit)
    
    # Statistics
    
    async def get_post_count(self) -> int:
        """Get total post count."""
        return await self._post_repo.count()
    
    async def get_user_post_count(self, user_id: UUID) -> int:
        """Get post count for user."""
        return await self._post_repo.count_by_user(user_id)
