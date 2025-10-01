"""
Post service for managing posts and feed generation.

This module provides comprehensive post management including CRUD operations,
reposting, feed generation with ML-based ranking, and engagement tracking.
"""

import json
import logging
from datetime import datetime, timedelta
from typing import List, Optional
from uuid import UUID

from sqlalchemy import and_, desc, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.core.exceptions import NotFoundError, ValidationError
from app.core.redis import get_redis
from app.users.models.follow import Follow
from app.posts.models.like import Like
from app.posts.models.post import Post
from app.auth.models.user import User
from app.posts.schemas.post import PostCreate, PostUpdate

logger = logging.getLogger(__name__)


class PostService:
    """Service for post management and feed generation."""
    
    def __init__(self, db: AsyncSession):
        """Initialize post service."""
        self.db = db
    
    async def create_post(
        self,
        user_id: UUID,
        post_data: PostCreate,
    ) -> Post:
        """
        Create a new post.
        
        Args:
            user_id: ID of the user creating the post
            post_data: Post creation data
            
        Returns:
            Created post object
            
        Raises:
            ValidationError: If post content is invalid
        """
        # Validate content length
        if len(post_data.content) < 1:
            raise ValidationError("Post content cannot be empty")
        
        if len(post_data.content) > 2000:
            raise ValidationError("Post content cannot exceed 2000 characters")
        
        # Extract hashtags and mentions
        hashtags = self._extract_hashtags(post_data.content)
        mentions = self._extract_mentions(post_data.content)
        
        # Create post
        post = Post(
            owner_id=user_id,
            content=post_data.content,
            media_url=post_data.media_url,
            media_type=post_data.media_type,
            hashtags=json.dumps(hashtags),
            mentions=json.dumps(mentions),
            is_approved=True,  # Auto-approve for now, add moderation later
        )
        
        self.db.add(post)
        await self.db.commit()
        await self.db.refresh(post)
        
        # Trigger ML moderation (async)
        from app.ml.ml_tasks import moderate_post_task
        moderate_post_task.apply_async(args=[str(post.id)])
        
        # Update user post count
        await self._update_user_post_count(user_id, increment=True)
        
        # Propagate to followers' feeds (async task)
        await self._propagate_to_feed(post)
        
        logger.info(f"Post created: {post.id} by user {user_id}")
        
        return post
    
    async def get_post(self, post_id: UUID) -> Optional[Post]:
        """
        Get post by ID.
        
        Args:
            post_id: Post ID
            
        Returns:
            Post object or None if not found
        """
        query = select(Post).where(Post.id == post_id).options(
            selectinload(Post.owner),
            selectinload(Post.comments),
            selectinload(Post.likes),
        )
        
        result = await self.db.execute(query)
        return result.scalar_one_or_none()
    
    async def update_post(
        self,
        post_id: UUID,
        user_id: UUID,
        post_data: PostUpdate,
    ) -> Post:
        """
        Update an existing post.
        
        Args:
            post_id: Post ID
            user_id: ID of user updating the post
            post_data: Post update data
            
        Returns:
            Updated post object
            
        Raises:
            NotFoundError: If post not found
            ValidationError: If user doesn't own the post
        """
        post = await self.get_post(post_id)
        
        if not post:
            raise NotFoundError(f"Post {post_id} not found")
        
        if post.owner_id != user_id:
            raise ValidationError("You can only edit your own posts")
        
        # Update fields
        if post_data.content is not None:
            if len(post_data.content) < 1 or len(post_data.content) > 2000:
                raise ValidationError("Invalid content length")
            
            post.content = post_data.content
            post.hashtags = json.dumps(self._extract_hashtags(post_data.content))
            post.mentions = json.dumps(self._extract_mentions(post_data.content))
        
        if post_data.media_url is not None:
            post.media_url = post_data.media_url
        
        if post_data.media_type is not None:
            post.media_type = post_data.media_type
        
        post.updated_at = datetime.utcnow()
        
        await self.db.commit()
        await self.db.refresh(post)
        
        logger.info(f"Post updated: {post_id} by user {user_id}")
        
        return post
    
    async def delete_post(self, post_id: UUID, user_id: UUID) -> None:
        """
        Delete a post.
        
        Args:
            post_id: Post ID
            user_id: ID of user deleting the post
            
        Raises:
            NotFoundError: If post not found
            ValidationError: If user doesn't own the post
        """
        post = await self.get_post(post_id)
        
        if not post:
            raise NotFoundError(f"Post {post_id} not found")
        
        if post.owner_id != user_id:
            raise ValidationError("You can only delete your own posts")
        
        await self.db.delete(post)
        await self.db.commit()
        
        # Update user post count
        await self._update_user_post_count(user_id, increment=False)
        
        # Remove from Redis feeds
        await self._remove_from_feeds(post_id)
        
        logger.info(f"Post deleted: {post_id} by user {user_id}")
    
    async def repost(
        self,
        user_id: UUID,
        original_post_id: UUID,
        reason: Optional[str] = None,
    ) -> Post:
        """
        Repost an existing post.
        
        Args:
            user_id: ID of user reposting
            original_post_id: ID of original post
            reason: Optional reason/comment for repost
            
        Returns:
            New repost object
            
        Raises:
            NotFoundError: If original post not found
            ValidationError: If trying to repost own post or already reposted
        """
        # Get original post
        original_post = await self.get_post(original_post_id)
        
        if not original_post:
            raise NotFoundError(f"Original post {original_post_id} not found")
        
        if original_post.owner_id == user_id:
            raise ValidationError("You cannot repost your own post")
        
        # Check if already reposted
        existing_repost = await self.db.execute(
            select(Post).where(
                and_(
                    Post.owner_id == user_id,
                    Post.original_post_id == original_post_id,
                    Post.is_repost == True,
                )
            )
        )
        
        if existing_repost.scalar_one_or_none():
            raise ValidationError("You have already reposted this post")
        
        # Create repost
        repost = Post(
            owner_id=user_id,
            content=original_post.content,
            media_url=original_post.media_url,
            media_type=original_post.media_type,
            hashtags=original_post.hashtags,
            mentions=original_post.mentions,
            is_repost=True,
            original_post_id=original_post_id,
            repost_reason=reason,
            is_approved=True,
        )
        
        self.db.add(repost)
        
        # Update original post repost count
        original_post.reposts_count += 1
        
        await self.db.commit()
        await self.db.refresh(repost)
        
        # Update user post count
        await self._update_user_post_count(user_id, increment=True)
        
        # Propagate to followers' feeds
        await self._propagate_to_feed(repost)
        
        logger.info(f"Post reposted: {original_post_id} by user {user_id}")
        
        return repost
    
    async def get_user_posts(
        self,
        user_id: UUID,
        skip: int = 0,
        limit: int = 20,
        include_reposts: bool = True,
    ) -> List[Post]:
        """
        Get posts by a specific user.
        
        Args:
            user_id: User ID
            skip: Number of posts to skip
            limit: Maximum number of posts to return
            include_reposts: Whether to include reposts
            
        Returns:
            List of posts
        """
        query = select(Post).where(Post.owner_id == user_id)
        
        if not include_reposts:
            query = query.where(Post.is_repost == False)
        
        query = query.order_by(desc(Post.created_at)).offset(skip).limit(limit)
        query = query.options(
            selectinload(Post.owner),
            selectinload(Post.original_post),
        )
        
        result = await self.db.execute(query)
        return list(result.scalars().all())
    
    async def get_feed(
        self,
        user_id: UUID,
        skip: int = 0,
        limit: int = 20,
        algorithm: str = "ml_ranked",
    ) -> List[Post]:
        """
        Generate personalized feed for user.
        
        Args:
            user_id: User ID
            skip: Number of posts to skip (for pagination)
            limit: Maximum number of posts to return
            algorithm: Feed algorithm to use ("chronological", "engagement", "ml_ranked")
            
        Returns:
            List of posts for user's feed
        """
        if algorithm == "chronological":
            return await self._get_chronological_feed(user_id, skip, limit)
        elif algorithm == "engagement":
            return await self._get_engagement_feed(user_id, skip, limit)
        else:  # ml_ranked (default)
            return await self._get_ml_ranked_feed(user_id, skip, limit)
    
    async def _get_chronological_feed(
        self,
        user_id: UUID,
        skip: int,
        limit: int,
    ) -> List[Post]:
        """Get chronologically ordered feed."""
        # Get users that current user follows
        following_query = select(Follow.following_id).where(Follow.follower_id == user_id)
        following_result = await self.db.execute(following_query)
        following_ids = [row[0] for row in following_result.all()]
        
        # Include user's own posts
        following_ids.append(user_id)
        
        # Get posts from followed users
        query = (
            select(Post)
            .where(
                and_(
                    Post.owner_id.in_(following_ids),
                    Post.is_approved == True,
                )
            )
            .order_by(desc(Post.created_at))
            .offset(skip)
            .limit(limit)
            .options(
                selectinload(Post.owner),
                selectinload(Post.original_post),
            )
        )
        
        result = await self.db.execute(query)
        return list(result.scalars().all())
    
    async def _get_engagement_feed(
        self,
        user_id: UUID,
        skip: int,
        limit: int,
    ) -> List[Post]:
        """Get feed sorted by engagement metrics."""
        # Get users that current user follows
        following_query = select(Follow.following_id).where(Follow.follower_id == user_id)
        following_result = await self.db.execute(following_query)
        following_ids = [row[0] for row in following_result.all()]
        following_ids.append(user_id)
        
        # Calculate engagement score
        # Score = likes + (comments * 2) + (reposts * 3) + (shares * 1.5)
        engagement_score = (
            Post.likes_count +
            (Post.comments_count * 2) +
            (Post.reposts_count * 3) +
            (Post.shares_count * 1.5)
        )
        
        # Get posts sorted by engagement
        query = (
            select(Post)
            .where(
                and_(
                    Post.owner_id.in_(following_ids),
                    Post.is_approved == True,
                    Post.created_at >= datetime.utcnow() - timedelta(days=7),  # Last 7 days
                )
            )
            .order_by(desc(engagement_score), desc(Post.created_at))
            .offset(skip)
            .limit(limit)
            .options(
                selectinload(Post.owner),
                selectinload(Post.original_post),
            )
        )
        
        result = await self.db.execute(query)
        return list(result.scalars().all())
    
    async def _get_ml_ranked_feed(
        self,
        user_id: UUID,
        skip: int,
        limit: int,
    ) -> List[Post]:
        """
        Get ML-ranked feed using hybrid scoring.
        
        Score = 0.4 * Recency + 0.3 * Engagement + 0.2 * Author Affinity + 0.1 * ML Prediction
        """
        # First, get candidate posts from Redis cache
        redis_client = await get_redis()
        cache_key = f"feed:{user_id}"
        
        # Try to get from cache
        cached_post_ids = await redis_client.zrevrange(
            cache_key,
            skip,
            skip + limit - 1,
            withscores=False,
        )
        
        if cached_post_ids:
            # Get posts from database
            query = (
                select(Post)
                .where(Post.id.in_([UUID(pid.decode()) for pid in cached_post_ids]))
                .options(
                    selectinload(Post.owner),
                    selectinload(Post.original_post),
                )
            )
            
            result = await self.db.execute(query)
            posts = list(result.scalars().all())
            
            # Sort by original order from Redis
            post_dict = {post.id: post for post in posts}
            sorted_posts = [post_dict[UUID(pid.decode())] for pid in cached_post_ids if UUID(pid.decode()) in post_dict]
            
            return sorted_posts
        
        # Cache miss - generate feed and cache it
        posts = await self._generate_ml_ranked_feed(user_id, limit=100)  # Generate larger batch for caching
        
        # Cache the feed in Redis (sorted set with scores)
        if posts:
            post_scores = {}
            for idx, post in enumerate(posts):
                score = len(posts) - idx  # Higher score for earlier posts
                post_scores[str(post.id)] = score
            
            await redis_client.zadd(cache_key, post_scores)
            await redis_client.expire(cache_key, 3600)  # Expire after 1 hour
        
        return posts[skip:skip + limit]
    
    async def _generate_ml_ranked_feed(
        self,
        user_id: UUID,
        limit: int = 100,
    ) -> List[Post]:
        """Generate ML-ranked feed with hybrid scoring."""
        # Get users that current user follows
        following_query = select(Follow.following_id).where(Follow.follower_id == user_id)
        following_result = await self.db.execute(following_query)
        following_ids = [row[0] for row in following_result.all()]
        following_ids.append(user_id)
        
        # Get recent posts (last 48 hours for fresh content)
        recent_cutoff = datetime.utcnow() - timedelta(hours=48)
        
        query = (
            select(Post)
            .where(
                and_(
                    Post.owner_id.in_(following_ids),
                    Post.is_approved == True,
                    Post.created_at >= recent_cutoff,
                )
            )
            .options(
                selectinload(Post.owner),
                selectinload(Post.original_post),
            )
        )
        
        result = await self.db.execute(query)
        posts = list(result.scalars().all())
        
        # Calculate hybrid scores for each post
        scored_posts = []
        now = datetime.utcnow()
        
        for post in posts:
            # Recency score (0-1): Exponential decay
            hours_old = (now - post.created_at).total_seconds() / 3600
            recency_score = 1.0 / (1 + hours_old / 6)  # Half-life of 6 hours
            
            # Engagement score (0-1): Normalized engagement
            engagement = (
                post.likes_count +
                (post.comments_count * 2) +
                (post.reposts_count * 3) +
                (post.shares_count * 1.5)
            )
            max_engagement = 100  # Normalize to 100
            engagement_score = min(engagement / max_engagement, 1.0)
            
            # Author affinity score (0-1): Based on user's interaction history
            # TODO: Calculate from ML model - for now use simple heuristic
            affinity_score = 0.5  # Placeholder
            
            # ML prediction score (0-1): Content relevance prediction
            # TODO: Use ML model - for now use engagement as proxy
            ml_score = engagement_score * 0.8
            
            # Hybrid score
            final_score = (
                0.4 * recency_score +
                0.3 * engagement_score +
                0.2 * affinity_score +
                0.1 * ml_score
            )
            
            scored_posts.append((post, final_score))
        
        # Sort by score
        scored_posts.sort(key=lambda x: x[1], reverse=True)
        
        return [post for post, score in scored_posts[:limit]]
    
    async def like_post(self, user_id: UUID, post_id: UUID) -> None:
        """Like a post."""
        # Check if already liked
        existing_like = await self.db.execute(
            select(Like).where(
                and_(
                    Like.user_id == user_id,
                    Like.post_id == post_id,
                )
            )
        )
        
        if existing_like.scalar_one_or_none():
            raise ValidationError("Post already liked")
        
        # Get post
        post = await self.get_post(post_id)
        if not post:
            raise NotFoundError(f"Post {post_id} not found")
        
        # Create like
        like = Like(
            user_id=user_id,
            post_id=post_id,
        )
        
        self.db.add(like)
        
        # Update post like count
        post.likes_count += 1
        
        await self.db.commit()
        
        logger.info(f"Post liked: {post_id} by user {user_id}")
    
    async def unlike_post(self, user_id: UUID, post_id: UUID) -> None:
        """Unlike a post."""
        # Get like
        like_query = select(Like).where(
            and_(
                Like.user_id == user_id,
                Like.post_id == post_id,
            )
        )
        
        result = await self.db.execute(like_query)
        like = result.scalar_one_or_none()
        
        if not like:
            raise NotFoundError("Like not found")
        
        # Get post
        post = await self.get_post(post_id)
        if post:
            # Update post like count
            post.likes_count = max(0, post.likes_count - 1)
        
        await self.db.delete(like)
        await self.db.commit()
        
        logger.info(f"Post unliked: {post_id} by user {user_id}")
    
    def _extract_hashtags(self, content: str) -> List[str]:
        """Extract hashtags from content."""
        import re
        return re.findall(r'#(\w+)', content)
    
    def _extract_mentions(self, content: str) -> List[str]:
        """Extract mentions from content."""
        import re
        return re.findall(r'@(\w+)', content)
    
    async def _update_user_post_count(self, user_id: UUID, increment: bool = True) -> None:
        """Update user's post count."""
        user_query = select(User).where(User.id == user_id)
        result = await self.db.execute(user_query)
        user = result.scalar_one_or_none()
        
        if user:
            if increment:
                user.posts_count += 1
            else:
                user.posts_count = max(0, user.posts_count - 1)
            
            await self.db.commit()
    
    async def _propagate_to_feed(self, post: Post) -> None:
        """
        Propagate post to followers' Redis feeds.
        
        This is a fan-out write approach for real-time feed updates.
        """
        # Get all followers
        followers_query = select(Follow.follower_id).where(Follow.following_id == post.owner_id)
        result = await self.db.execute(followers_query)
        follower_ids = [row[0] for row in result.all()]
        
        # Add to each follower's feed in Redis
        redis_client = await get_redis()
        
        for follower_id in follower_ids:
            cache_key = f"feed:{follower_id}"
            
            # Calculate initial score (timestamp-based)
            score = post.created_at.timestamp()
            
            # Add to sorted set
            await redis_client.zadd(cache_key, {str(post.id): score})
            
            # Keep only last 1000 posts in cache
            await redis_client.zremrangebyrank(cache_key, 0, -1001)
    
    async def _remove_from_feeds(self, post_id: UUID) -> None:
        """Remove post from all Redis feeds."""
        redis_client = await get_redis()
        
        # Get all user IDs (this is expensive - consider better approach for production)
        # For now, we'll just invalidate the feed cache for the post owner's followers
        pass  # TODO: Implement efficient feed removal


# Service instance
post_service = PostService
