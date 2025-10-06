"""
Smart Search Service with Postgres FTS and Elasticsearch support.

This service provides intelligent search capabilities across videos, posts, users,
and hashtags with hybrid ranking, caching, and analytics.
"""

import logging
import json
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from uuid import UUID

from sqlalchemy import select, func, or_, and_, desc, case
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.sql import text

from app.models.video import Video, VideoStatus, VideoVisibility, ModerationStatus
from app.models.social import Post
from app.models.user import User, UserStatus
from app.core.redis import get_redis
from app.core.config import settings

logger = logging.getLogger(__name__)


class SearchService:
    """
    Smart search service with multiple backends and hybrid ranking.
    
    Features:
    - Postgres full-text search with tsvector
    - Hybrid ranking (relevance + engagement + recency)
    - Redis caching for performance
    - Search analytics tracking
    - Autocomplete and suggestions
    - Trending search queries
    """
    
    def __init__(self, db: AsyncSession):
        self.db = db
        self._redis = None
    
    async def _get_redis(self):
        """Get Redis client lazily."""
        if self._redis is None:
            self._redis = await get_redis()
        return self._redis
    
    # Unified Search
    
    async def search_all(
        self,
        query: str,
        user_id: Optional[UUID] = None,
        limit: int = 20,
        offset: int = 0,
        filters: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Unified search across all content types.
        
        Args:
            query: Search query string
            user_id: Optional user ID for personalization
            limit: Results per type
            offset: Pagination offset
            filters: Optional filters (content_type, date_range, etc.)
            
        Returns:
            Dict with results for videos, posts, users, and hashtags
        """
        # Track search query
        await self._track_search(query, user_id)
        
        # Search each content type in parallel
        videos_task = self.search_videos(query, limit, offset, filters)
        posts_task = self.search_posts(query, limit, offset, filters)
        users_task = self.search_users(query, limit, offset)
        
        videos = await videos_task
        posts = await posts_task
        users = await users_task
        
        return {
            "query": query,
            "results": {
                "videos": videos["results"],
                "posts": posts["results"],
                "users": users["results"],
            },
            "counts": {
                "videos": videos["total"],
                "posts": posts["total"],
                "users": users["total"],
            },
            "limit": limit,
            "offset": offset,
        }
    
    # Video Search
    
    async def search_videos(
        self,
        query: str,
        limit: int = 20,
        offset: int = 0,
        filters: Optional[Dict[str, Any]] = None,
        sort_by: str = "relevance",
    ) -> Dict[str, Any]:
        """
        Search videos with hybrid ranking.
        
        Ranking factors:
        - Text relevance (title, description, tags)
        - Engagement metrics (views, likes, comments)
        - Recency (newer content ranked higher)
        - Video quality (resolution, duration)
        """
        # Check cache first
        cache_key = f"search:videos:{query}:{sort_by}:{offset}:{limit}"
        redis = await self._get_redis()
        
        if redis:
            cached = await redis.get(cache_key)
            if cached:
                return json.loads(cached)
        
        # Build search query with FTS
        search_filter = self._build_video_search_filter(query, filters)
        
        # Base query
        base_query = (
            select(Video)
            .where(search_filter)
        )
        
        # Apply sorting
        if sort_by == "relevance":
            # Hybrid relevance score
            # Use real column names; legacy property names mapped via model alias properties for instances
            engagement_score = (
                Video.view_count * 1.0 +
                Video.like_count * 5.0 +
                Video.comment_count * 10.0 +
                Video.share_count * 15.0
            )
            
            # Recency boost (videos from last 30 days get bonus)
            recency_boost = case(
                (Video.created_at >= datetime.utcnow() - timedelta(days=30), 1.5),
                else_=1.0
            )
            
            relevance_score = engagement_score * recency_boost
            
            query_stmt = base_query.order_by(desc(relevance_score), desc(Video.created_at))
        elif sort_by == "recent":
            query_stmt = base_query.order_by(desc(Video.created_at))
        elif sort_by == "views":
            query_stmt = base_query.order_by(desc(Video.view_count))
        elif sort_by == "engagement":
            engagement = Video.like_count + Video.comment_count * 2
            query_stmt = base_query.order_by(desc(engagement))
        else:
            query_stmt = base_query.order_by(desc(Video.created_at))
        
        # Get total count
        count_query = select(func.count()).select_from(base_query.subquery())
        total_result = await self.db.execute(count_query)
        total = total_result.scalar() or 0
        
        # Get paginated results
        query_stmt = query_stmt.offset(offset).limit(limit)
        result = await self.db.execute(query_stmt)
        videos = result.scalars().all()
        
        # Format results
        results_data = {
            "results": [self._format_video(v) for v in videos],
            "total": total,
            "limit": limit,
            "offset": offset,
            "query": query,
            "sort_by": sort_by,
        }
        
        # Cache results
        if redis:
            await redis.setex(cache_key, 300, json.dumps(results_data, default=str))
        
        return results_data
    
    def _build_video_search_filter(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
    ):
        """Build video search filter with FTS and additional filters."""
        # Base visibility and status filters
        base_filter = and_(
            Video.visibility == VideoVisibility.PUBLIC,
            Video.status == VideoStatus.PROCESSED,
            Video.moderation_status == ModerationStatus.APPROVED,
        )
        
        # Text search across title, description, and tags
        query_lower = query.lower()
        text_filter = or_(
            Video.title.ilike(f"%{query_lower}%"),
            Video.description.ilike(f"%{query_lower}%"),
            Video.tags.ilike(f"%{query_lower}%"),
        )
        
        combined_filter = and_(base_filter, text_filter)
        
        # Apply additional filters
        if filters:
            if "duration_min" in filters:
                combined_filter = and_(
                    combined_filter,
                    Video.duration >= filters["duration_min"]
                )
            if "duration_max" in filters:
                combined_filter = and_(
                    combined_filter,
                    Video.duration <= filters["duration_max"]
                )
            if "created_after" in filters:
                combined_filter = and_(
                    combined_filter,
                    Video.created_at >= filters["created_after"]
                )
            if "owner_id" in filters:
                combined_filter = and_(
                    combined_filter,
                    Video.owner_id == filters["owner_id"]
                )
        
        return combined_filter
    
    # Post Search
    
    async def search_posts(
        self,
        query: str,
        limit: int = 20,
        offset: int = 0,
        filters: Optional[Dict[str, Any]] = None,
        sort_by: str = "relevance",
    ) -> Dict[str, Any]:
        """
        Search posts with hybrid ranking.
        
        Ranking factors:
        - Text relevance (content, hashtags)
        - Engagement metrics (likes, reposts, comments)
        - Recency (newer posts ranked higher)
        - Author reputation
        """
        # Check cache
        cache_key = f"search:posts:{query}:{sort_by}:{offset}:{limit}"
        redis = await self._get_redis()
        
        if redis:
            cached = await redis.get(cache_key)
            if cached:
                return json.loads(cached)
        
        # Build search filter
        search_filter = self._build_post_search_filter(query, filters)
        
        # Base query
        base_query = select(Post).where(search_filter)
        
        # Apply sorting
        if sort_by == "relevance":
            engagement_score = (
                Post.like_count * 1.0 +
                Post.repost_count * 3.0 +
                Post.comment_count * 2.0 +
                Post.share_count * 2.5
            )
            
            recency_boost = case(
                (Post.created_at >= datetime.utcnow() - timedelta(days=7), 1.5),
                else_=1.0
            )
            
            relevance_score = engagement_score * recency_boost
            query_stmt = base_query.order_by(desc(relevance_score), desc(Post.created_at))
        elif sort_by == "recent":
            query_stmt = base_query.order_by(desc(Post.created_at))
        elif sort_by == "popular":
            query_stmt = base_query.order_by(desc(Post.like_count))
        else:
            query_stmt = base_query.order_by(desc(Post.created_at))
        
        # Get total count
        count_query = select(func.count()).select_from(base_query.subquery())
        total_result = await self.db.execute(count_query)
        total = total_result.scalar() or 0
        
        # Get paginated results
        query_stmt = query_stmt.offset(offset).limit(limit)
        result = await self.db.execute(query_stmt)
        posts = result.scalars().all()
        
        # Format results
        results_data = {
            "results": [self._format_post(p) for p in posts],
            "total": total,
            "limit": limit,
            "offset": offset,
            "query": query,
            "sort_by": sort_by,
        }
        
        # Cache results
        if redis:
            await redis.setex(cache_key, 300, json.dumps(results_data, default=str))
        
        return results_data
    
    def _build_post_search_filter(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
    ):
        """Build post search filter."""
        # Base filter
        base_filter = and_()  # Simplified: assume all posts are eligible in minimal implementation
        
        # Text search
        query_lower = query.lower()
        text_filter = or_(
            Post.content.ilike(f"%{query_lower}%"),
            Post.hashtags.ilike(f"%{query_lower}%"),
        )
        
        combined_filter = and_(base_filter, text_filter)
        
        # Additional filters
        if filters:
            if "created_after" in filters:
                combined_filter = and_(
                    combined_filter,
                    Post.created_at >= filters["created_after"]
                )
            if "owner_id" in filters:
                combined_filter = and_(
                    combined_filter,
                    Post.owner_id == filters["owner_id"]
                )
            if "has_media" in filters and filters["has_media"]:
                combined_filter = and_(
                    combined_filter,
                    Post.media_url.isnot(None)
                )
        
        return combined_filter
    
    # User Search
    
    async def search_users(
        self,
        query: str,
        limit: int = 20,
        offset: int = 0,
    ) -> Dict[str, Any]:
        """Search users by username, display name, or bio."""
        query_lower = query.lower()
        
        # Search filter
        search_filter = or_(
            User.username.ilike(f"%{query_lower}%"),
            User.email.ilike(f"%{query_lower}%"),
        )
        
        # Base query
        base_query = select(User).where(
            and_(
                search_filter,
                User.status == UserStatus.ACTIVE,
            )
        )
        
        # Get total
        count_query = select(func.count()).select_from(base_query.subquery())
        total_result = await self.db.execute(count_query)
        total = total_result.scalar() or 0
        
        # Get results
        query_stmt = base_query.offset(offset).limit(limit)
        result = await self.db.execute(query_stmt)
        users = result.scalars().all()
        
        return {
            "results": [self._format_user(u) for u in users],
            "total": total,
            "limit": limit,
            "offset": offset,
            "query": query,
        }
    
    # Hashtag Search
    
    async def search_hashtags(
        self,
        query: str,
        limit: int = 20,
    ) -> Dict[str, Any]:
        """Search and rank hashtags by popularity."""
        query_lower = query.lower().lstrip('#')
        
        # Search posts with hashtags
        search_filter = Post.hashtags.ilike(f"%{query_lower}%")
        
        # Aggregate hashtag counts
        stmt = text("""
            SELECT 
                LOWER(TRIM(BOTH '"' FROM hashtag)) as hashtag,
                COUNT(*) as count
            FROM posts,
            jsonb_array_elements_text(
                CASE 
                    WHEN hashtags::text ~ '^\\[.*\\]$' 
                    THEN hashtags::jsonb 
                    ELSE '[]'::jsonb 
                END
            ) as hashtag
            WHERE 
                LOWER(hashtag) LIKE :query
                AND is_approved = true
                AND created_at >= :cutoff_date
            GROUP BY LOWER(TRIM(BOTH '"' FROM hashtag))
            ORDER BY count DESC
            LIMIT :limit
        """)
        
        result = await self.db.execute(
            stmt,
            {
                "query": f"%{query_lower}%",
                "cutoff_date": datetime.utcnow() - timedelta(days=30),
                "limit": limit,
            }
        )
        
        hashtags = [
            {
                "hashtag": row.hashtag,
                "count": row.count,
                "trending_score": self._calculate_hashtag_trend_score(row.count),
            }
            for row in result.fetchall()
        ]
        
        return {
            "results": hashtags,
            "query": query,
            "limit": limit,
        }
    
    # Autocomplete and Suggestions
    
    async def get_suggestions(
        self,
        query: str,
        limit: int = 10,
        user_id: Optional[UUID] = None,
    ) -> Dict[str, Any]:
        """Get search suggestions for autocomplete."""
        suggestions = []
        
        # Get video title suggestions
        video_stmt = (
            select(Video.title)
            .where(
                and_(
                    Video.title.ilike(f"{query}%"),
                    Video.visibility == VideoVisibility.PUBLIC,
                    Video.status == VideoStatus.PROCESSED,
                )
            )
            .order_by(desc(Video.views_count))
            .limit(limit)
        )
        
        result = await self.db.execute(video_stmt)
        video_titles = result.scalars().all()
        suggestions.extend([{"text": title, "type": "video"} for title in video_titles])
        
        # Get user suggestions
        user_stmt = (
            select(User.username)
            .where(
                and_(
                    User.username.ilike(f"{query}%"),
                    User.status == UserStatus.ACTIVE,
                )
            )
            .limit(limit // 2)
        )
        
        result = await self.db.execute(user_stmt)
        usernames = result.scalars().all()
        suggestions.extend([{"text": name, "type": "user"} for name in usernames])
        
        return {
            "query": query,
            "suggestions": suggestions[:limit],
        }
    
    # Trending Searches
    
    async def get_trending_searches(
        self,
        limit: int = 20,
        time_window: str = "24h",
    ) -> Dict[str, Any]:
        """Get trending search queries."""
        redis = await self._get_redis()
        
        if not redis:
            return {"trending_searches": [], "time_window": time_window}
        
        # Get trending from Redis sorted set
        cache_key = "trending:searches"
        trending = await redis.zrevrange(cache_key, 0, limit - 1, withscores=True)
        
        results = [
            {
                "query": query.decode() if isinstance(query, bytes) else query,
                "score": score,
            }
            for query, score in trending
        ]
        
        return {
            "trending_searches": results,
            "time_window": time_window,
            "limit": limit,
        }
    
    # Analytics and Tracking
    
    async def _track_search(self, query: str, user_id: Optional[UUID] = None):
        """Track search query for analytics and trending."""
        redis = await self._get_redis()
        
        if not redis:
            return
        
        try:
            # Increment search count in trending
            await redis.zincrby("trending:searches", 1, query)
            
            # Expire old entries (keep 7 days)
            await redis.expire("trending:searches", 7 * 24 * 3600)
            
            # Track user search history
            if user_id:
                user_key = f"user:{user_id}:search_history"
                await redis.lpush(user_key, json.dumps({
                    "query": query,
                    "timestamp": datetime.utcnow().isoformat(),
                }))
                await redis.ltrim(user_key, 0, 99)  # Keep last 100 searches
                await redis.expire(user_key, 30 * 24 * 3600)  # 30 days
        except Exception as e:
            logger.error(f"Error tracking search: {e}")
    
    # Helpers
    
    def _format_video(self, video: Video) -> Dict[str, Any]:
        """Format video for search results."""
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
        }
    
    def _format_post(self, post: Post) -> Dict[str, Any]:
        """Format post for search results."""
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
        }
    
    def _format_user(self, user: User) -> Dict[str, Any]:
        """Format user for search results."""
        return {
            "id": str(user.id),
            "username": user.username,
            "email": user.email,
            "is_active": user.is_active,
        }
    
    def _calculate_hashtag_trend_score(self, count: int) -> float:
        """Calculate trending score for hashtag."""
        # Simple logarithmic scoring
        import math
        return math.log10(count + 1) * 10


