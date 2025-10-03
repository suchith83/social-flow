"""
Recommendation Service with hybrid algorithms and ML integration.

This service provides personalized content recommendations using:
- Collaborative filtering
- Content-based filtering
- Popularity and trending signals
- User interaction history
- ML model predictions
"""

import logging
import json
from typing import Any, Dict, List, Optional
from datetime import datetime, timedelta
from uuid import UUID
import random

from sqlalchemy import select, func, and_, desc, or_
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.video import Video, VideoStatus, VideoVisibility
from app.models.social import Post
from app.models.social import Follow
from app.core.redis import get_redis
from app.ml.services.ml_service import MLService

logger = logging.getLogger(__name__)


class RecommendationService:
    """
    Hybrid recommendation service combining multiple algorithms.
    
    Algorithms:
    1. Collaborative Filtering: "Users like you also liked..."
    2. Content-Based: Similar content based on tags, category
    3. Trending: Popular content in the user's network
    4. Diversity: Ensure content variety
    5. Recency: Prioritize fresh content
    """
    
    def __init__(self, db: AsyncSession):
        self.db = db
        self._redis = None
        self._ml_service = None
    
    async def _get_redis(self):
        """Get Redis client lazily."""
        if self._redis is None:
            self._redis = await get_redis()
        return self._redis
    
    async def _get_ml_service(self) -> MLService:
        """Get ML service lazily."""
        if self._ml_service is None:
            self._ml_service = MLService()
        return self._ml_service
    
    # Video Recommendations
    
    async def get_video_recommendations(
        self,
        user_id: Optional[UUID] = None,
        limit: int = 20,
        algorithm: str = "hybrid",
        exclude_ids: Optional[List[UUID]] = None,
    ) -> Dict[str, Any]:
        """
        Get personalized video recommendations.
        
        Args:
            user_id: User ID for personalization (None for anonymous)
            limit: Number of recommendations
            algorithm: Algorithm to use (hybrid, trending, collaborative, content_based)
            exclude_ids: Video IDs to exclude
            
        Returns:
            Dict with recommended videos and metadata
        """
        # Check cache
        cache_key = f"recommendations:videos:{user_id}:{algorithm}"
        redis = await self._get_redis()
        
        if redis and user_id:
            cached = await redis.get(cache_key)
            if cached:
                cached_data = json.loads(cached)
                # Filter out excluded IDs
                if exclude_ids:
                    cached_data["recommendations"] = [
                        r for r in cached_data["recommendations"]
                        if UUID(r["id"]) not in exclude_ids
                    ]
                return cached_data[:limit]
        
        # Generate recommendations based on algorithm
        if algorithm == "hybrid":
            videos = await self._get_hybrid_video_recommendations(
                user_id, limit * 2, exclude_ids
            )
        elif algorithm == "trending":
            videos = await self._get_trending_videos(limit, exclude_ids)
        elif algorithm == "collaborative":
            videos = await self._get_collaborative_video_recommendations(
                user_id, limit, exclude_ids
            )
        elif algorithm == "content_based":
            videos = await self._get_content_based_video_recommendations(
                user_id, limit, exclude_ids
            )
        else:
            # Default to hybrid
            videos = await self._get_hybrid_video_recommendations(
                user_id, limit * 2, exclude_ids
            )
        
        # Format and rank results
        recommendations = self._rank_and_diversify_videos(videos, limit)
        
        result = {
            "recommendations": [self._format_video(v) for v in recommendations],
            "algorithm": algorithm,
            "generated_at": datetime.utcnow().isoformat(),
            "count": len(recommendations),
        }
        
        # Cache results
        if redis and user_id:
            await redis.setex(cache_key, 900, json.dumps(result, default=str))  # 15 min cache
        
        return result
    
    async def _get_hybrid_video_recommendations(
        self,
        user_id: Optional[UUID],
        limit: int,
        exclude_ids: Optional[List[UUID]] = None,
    ) -> List[Video]:
        """
        Hybrid algorithm combining multiple signals.
        
        Weights:
        - 40% Collaborative filtering (similar users)
        - 30% Content-based (similar content)
        - 20% Trending (popular now)
        - 10% Diversity (explore new content)
        """
        recommendations = []
        
        # Get collaborative recommendations (40%)
        collab_videos = await self._get_collaborative_video_recommendations(
            user_id, int(limit * 0.4), exclude_ids
        )
        recommendations.extend(collab_videos)
        
        # Get content-based recommendations (30%)
        content_videos = await self._get_content_based_video_recommendations(
            user_id, int(limit * 0.3), exclude_ids
        )
        recommendations.extend(content_videos)
        
        # Get trending videos (20%)
        trending_videos = await self._get_trending_videos(
            int(limit * 0.2), exclude_ids
        )
        recommendations.extend(trending_videos)
        
        # Get diverse/exploratory content (10%)
        diverse_videos = await self._get_diverse_videos(
            user_id, int(limit * 0.1), exclude_ids
        )
        recommendations.extend(diverse_videos)
        
        # Remove duplicates
        seen_ids = set()
        unique_videos = []
        for video in recommendations:
            if video.id not in seen_ids:
                seen_ids.add(video.id)
                unique_videos.append(video)
        
        return unique_videos[:limit]
    
    async def _get_collaborative_video_recommendations(
        self,
        user_id: Optional[UUID],
        limit: int,
        exclude_ids: Optional[List[UUID]] = None,
    ) -> List[Video]:
        """
        Collaborative filtering: recommend videos liked by similar users.
        """
        if not user_id:
            # For anonymous users, return popular videos
            return await self._get_trending_videos(limit, exclude_ids)
        
        # Find users that the current user follows
        following_stmt = select(Follow.following_id).where(Follow.follower_id == user_id)
        result = await self.db.execute(following_stmt)
        following_ids = [row[0] for row in result.all()]
        
        if not following_ids:
            # If not following anyone, return trending
            return await self._get_trending_videos(limit, exclude_ids)
        
        # Get videos watched/liked by followed users
        filters = [
            Video.owner_id.in_(following_ids),
            Video.visibility == VideoVisibility.PUBLIC,
            Video.status == VideoStatus.PROCESSED,
            Video.is_approved.is_(True),
        ]
        
        if exclude_ids:
            filters.append(Video.id.notin_(exclude_ids))
        
        stmt = (
            select(Video)
            .where(and_(*filters))
            .order_by(
                desc(Video.likes_count + Video.views_count * 0.1)
            )
            .limit(limit)
        )
        
        result = await self.db.execute(stmt)
        return list(result.scalars().all())
    
    async def _get_content_based_video_recommendations(
        self,
        user_id: Optional[UUID],
        limit: int,
        exclude_ids: Optional[List[UUID]] = None,
    ) -> List[Video]:
        """
        Content-based filtering: recommend similar videos based on tags.
        """
        # Get user's recently watched or liked videos to extract preferences
        # For simplicity, we'll get popular videos with similar tags
        # TODO: Implement tag-based similarity using user history
        
        filters = [
            Video.visibility == VideoVisibility.PUBLIC,
            Video.status == VideoStatus.PROCESSED,
            Video.is_approved.is_(True),
        ]
        
        if exclude_ids:
            filters.append(Video.id.notin_(exclude_ids))
        
        # Prioritize videos with tags (indicating better metadata)
        stmt = (
            select(Video)
            .where(and_(*filters))
            .where(Video.tags.isnot(None))
            .order_by(desc(Video.views_count))
            .limit(limit)
        )
        
        result = await self.db.execute(stmt)
        return list(result.scalars().all())
    
    async def _get_trending_videos(
        self,
        limit: int,
        exclude_ids: Optional[List[UUID]] = None,
    ) -> List[Video]:
        """Get trending videos based on recent engagement."""
        cutoff_date = datetime.utcnow() - timedelta(days=7)
        
        filters = [
            Video.visibility == VideoVisibility.PUBLIC,
            Video.status == VideoStatus.PROCESSED,
            Video.is_approved.is_(True),
            Video.created_at >= cutoff_date,
        ]
        
        if exclude_ids:
            filters.append(Video.id.notin_(exclude_ids))
        
        # Trending score: weighted engagement with recency boost
        trending_score = (
            Video.views_count * 1.0 +
            Video.likes_count * 5.0 +
            Video.comments_count * 10.0 +
            Video.shares_count * 15.0
        )
        
        stmt = (
            select(Video)
            .where(and_(*filters))
            .order_by(desc(trending_score))
            .limit(limit)
        )
        
        result = await self.db.execute(stmt)
        return list(result.scalars().all())
    
    async def _get_diverse_videos(
        self,
        user_id: Optional[UUID],
        limit: int,
        exclude_ids: Optional[List[UUID]] = None,
    ) -> List[Video]:
        """Get diverse content for exploration."""
        filters = [
            Video.visibility == VideoVisibility.PUBLIC,
            Video.status == VideoStatus.PROCESSED,
            Video.is_approved.is_(True),
        ]
        
        if exclude_ids:
            filters.append(Video.id.notin_(exclude_ids))
        
        # Get random sample of quality videos
        stmt = (
            select(Video)
            .where(and_(*filters))
            .where(Video.views_count > 100)  # Quality filter
            .order_by(func.random())
            .limit(limit * 2)
        )
        
        result = await self.db.execute(stmt)
        videos = list(result.scalars().all())
        
        # Return random subset
        return random.sample(videos, min(limit, len(videos)))
    
    # Post/Feed Recommendations
    
    async def get_feed_recommendations(
        self,
        user_id: UUID,
        limit: int = 20,
        algorithm: str = "hybrid",
        exclude_ids: Optional[List[UUID]] = None,
    ) -> Dict[str, Any]:
        """
        Get personalized feed recommendations (posts).
        
        Similar to video recommendations but for posts.
        """
        # Check cache
        cache_key = f"recommendations:feed:{user_id}:{algorithm}"
        redis = await self._get_redis()
        
        if redis:
            cached = await redis.get(cache_key)
            if cached:
                cached_data = json.loads(cached)
                if exclude_ids:
                    cached_data["recommendations"] = [
                        r for r in cached_data["recommendations"]
                        if UUID(r["id"]) not in exclude_ids
                    ]
                return cached_data[:limit]
        
        # Generate recommendations
        if algorithm == "hybrid":
            posts = await self._get_hybrid_post_recommendations(
                user_id, limit * 2, exclude_ids
            )
        elif algorithm == "trending":
            posts = await self._get_trending_posts(limit, exclude_ids)
        elif algorithm == "following":
            posts = await self._get_following_posts(user_id, limit, exclude_ids)
        else:
            posts = await self._get_hybrid_post_recommendations(
                user_id, limit * 2, exclude_ids
            )
        
        # Rank and diversify
        recommendations = self._rank_and_diversify_posts(posts, limit)
        
        result = {
            "recommendations": [self._format_post(p) for p in recommendations],
            "algorithm": algorithm,
            "generated_at": datetime.utcnow().isoformat(),
            "count": len(recommendations),
        }
        
        # Cache
        if redis:
            await redis.setex(cache_key, 600, json.dumps(result, default=str))  # 10 min cache
        
        return result
    
    async def _get_hybrid_post_recommendations(
        self,
        user_id: UUID,
        limit: int,
        exclude_ids: Optional[List[UUID]] = None,
    ) -> List[Post]:
        """Hybrid post recommendations."""
        recommendations = []
        
        # Following posts (50%)
        following_posts = await self._get_following_posts(
            user_id, int(limit * 0.5), exclude_ids
        )
        recommendations.extend(following_posts)
        
        # Trending posts (30%)
        trending_posts = await self._get_trending_posts(
            int(limit * 0.3), exclude_ids
        )
        recommendations.extend(trending_posts)
        
        # Diverse posts (20%)
        diverse_posts = await self._get_diverse_posts(
            user_id, int(limit * 0.2), exclude_ids
        )
        recommendations.extend(diverse_posts)
        
        # Remove duplicates
        seen_ids = set()
        unique_posts = []
        for post in recommendations:
            if post.id not in seen_ids:
                seen_ids.add(post.id)
                unique_posts.append(post)
        
        return unique_posts[:limit]
    
    async def _get_following_posts(
        self,
        user_id: UUID,
        limit: int,
        exclude_ids: Optional[List[UUID]] = None,
    ) -> List[Post]:
        """Get posts from users that current user follows."""
        # Get following IDs
        following_stmt = select(Follow.following_id).where(Follow.follower_id == user_id)
        result = await self.db.execute(following_stmt)
        following_ids = [row[0] for row in result.all()]
        
        if not following_ids:
            return []
        
        filters = [
            Post.owner_id.in_(following_ids),
            Post.is_approved.is_(True),
            Post.is_flagged.is_(False),
        ]
        
        if exclude_ids:
            filters.append(Post.id.notin_(exclude_ids))
        
        stmt = (
            select(Post)
            .where(and_(*filters))
            .order_by(desc(Post.created_at))
            .limit(limit)
        )
        
        result = await self.db.execute(stmt)
        return list(result.scalars().all())
    
    async def _get_trending_posts(
        self,
        limit: int,
        exclude_ids: Optional[List[UUID]] = None,
    ) -> List[Post]:
        """Get trending posts."""
        cutoff_date = datetime.utcnow() - timedelta(days=3)
        
        filters = [
            Post.is_approved.is_(True),
            Post.is_flagged.is_(False),
            Post.created_at >= cutoff_date,
        ]
        
        if exclude_ids:
            filters.append(Post.id.notin_(exclude_ids))
        
        trending_score = (
            Post.likes_count * 1.0 +
            Post.reposts_count * 3.0 +
            Post.comments_count * 2.0 +
            Post.shares_count * 2.5
        )
        
        stmt = (
            select(Post)
            .where(and_(*filters))
            .order_by(desc(trending_score))
            .limit(limit)
        )
        
        result = await self.db.execute(stmt)
        return list(result.scalars().all())
    
    async def _get_diverse_posts(
        self,
        user_id: UUID,
        limit: int,
        exclude_ids: Optional[List[UUID]] = None,
    ) -> List[Post]:
        """Get diverse posts for exploration."""
        filters = [
            Post.is_approved.is_(True),
            Post.is_flagged.is_(False),
        ]
        
        if exclude_ids:
            filters.append(Post.id.notin_(exclude_ids))
        
        stmt = (
            select(Post)
            .where(and_(*filters))
            .where(Post.likes_count > 10)
            .order_by(func.random())
            .limit(limit * 2)
        )
        
        result = await self.db.execute(stmt)
        posts = list(result.scalars().all())
        
        return random.sample(posts, min(limit, len(posts)))
    
    # Ranking and Diversification
    
    def _rank_and_diversify_videos(
        self,
        videos: List[Video],
        limit: int,
    ) -> List[Video]:
        """Rank and diversify video recommendations."""
        # Calculate scores
        scored_videos = []
        for video in videos:
            score = self._calculate_video_score(video)
            scored_videos.append((video, score))
        
        # Sort by score
        scored_videos.sort(key=lambda x: x[1], reverse=True)
        
        # Diversify by owner (avoid too many videos from same creator)
        diversified = []
        owner_counts = {}
        
        for video, score in scored_videos:
            owner_id = str(video.owner_id)
            owner_count = owner_counts.get(owner_id, 0)
            
            # Limit 2 videos per creator in top recommendations
            if owner_count < 2 or len(diversified) >= limit * 0.8:
                diversified.append(video)
                owner_counts[owner_id] = owner_count + 1
            
            if len(diversified) >= limit:
                break
        
        return diversified
    
    def _rank_and_diversify_posts(
        self,
        posts: List[Post],
        limit: int,
    ) -> List[Post]:
        """Rank and diversify post recommendations."""
        scored_posts = []
        for post in posts:
            score = self._calculate_post_score(post)
            scored_posts.append((post, score))
        
        scored_posts.sort(key=lambda x: x[1], reverse=True)
        
        # Diversify
        diversified = []
        owner_counts = {}
        
        for post, score in scored_posts:
            owner_id = str(post.owner_id)
            owner_count = owner_counts.get(owner_id, 0)
            
            if owner_count < 3 or len(diversified) >= limit * 0.8:
                diversified.append(post)
                owner_counts[owner_id] = owner_count + 1
            
            if len(diversified) >= limit:
                break
        
        return diversified
    
    def _calculate_video_score(self, video: Video) -> float:
        """Calculate recommendation score for a video."""
        # Engagement score
        engagement = (
            video.views_count * 0.1 +
            video.likes_count * 5.0 +
            video.comments_count * 10.0 +
            video.shares_count * 15.0
        )
        
        # Recency boost
        hours_old = (datetime.utcnow() - video.created_at).total_seconds() / 3600
        recency_factor = 1.0 / (1 + hours_old / 24)  # Decay over 24 hours
        
        # Quality score (based on retention, if available)
        quality = video.retention_rate / 100.0 if video.retention_rate else 0.5
        
        # Final score
        score = engagement * recency_factor * (0.5 + quality * 0.5)
        
        return score
    
    def _calculate_post_score(self, post: Post) -> float:
        """Calculate recommendation score for a post."""
        engagement = (
            post.likes_count * 1.0 +
            post.reposts_count * 3.0 +
            post.comments_count * 2.0 +
            post.shares_count * 2.5
        )
        
        hours_old = (datetime.utcnow() - post.created_at).total_seconds() / 3600
        recency_factor = 1.0 / (1 + hours_old / 12)  # Faster decay for posts
        
        score = engagement * recency_factor
        
        return score
    
    # Formatting helpers
    
    def _format_video(self, video: Video) -> Dict[str, Any]:
        """Format video for recommendation response."""
        return {
            "id": str(video.id),
            "title": video.title,
            "description": video.description,
            "thumbnail_url": video.thumbnail_url,
            "duration": video.duration,
            "views_count": video.views_count,
            "likes_count": video.likes_count,
            "owner_id": str(video.owner_id),
            "created_at": video.created_at.isoformat(),
            "engagement_rate": video.engagement_rate,
        }
    
    def _format_post(self, post: Post) -> Dict[str, Any]:
        """Format post for recommendation response."""
        return {
            "id": str(post.id),
            "content": post.content,
            "media_url": post.media_url,
            "hashtags": post.hashtags,
            "likes_count": post.likes_count,
            "reposts_count": post.reposts_count,
            "comments_count": post.comments_count,
            "owner_id": str(post.owner_id),
            "created_at": post.created_at.isoformat(),
            "engagement_rate": post.engagement_rate,
        }
    
    # Cache Management
    
    async def invalidate_user_cache(self, user_id: UUID):
        """Invalidate recommendation cache for a user."""
        redis = await self._get_redis()
        if not redis:
            return
        
        patterns = [
            f"recommendations:videos:{user_id}:*",
            f"recommendations:feed:{user_id}:*",
        ]
        
        for pattern in patterns:
            keys = await redis.keys(pattern)
            if keys:
                await redis.delete(*keys)

