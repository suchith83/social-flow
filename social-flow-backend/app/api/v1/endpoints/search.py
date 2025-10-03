"""
Search endpoints.

This module contains all search-related API endpoints with smart search
and recommendation capabilities.
"""

from typing import Any, Optional, Dict
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.models.user import User
from app.auth.api.auth import get_current_active_user
from app.analytics.services.analytics_service import analytics_service
from app.services.search_service import SearchService
from app.services.recommendation_service import RecommendationService

router = APIRouter()


def get_search_service(db: AsyncSession = Depends(get_db)) -> SearchService:
    """Dependency to get search service."""
    return SearchService(db)


def get_recommendation_service(db: AsyncSession = Depends(get_db)) -> RecommendationService:
    """Dependency to get recommendation service."""
    return RecommendationService(db)


@router.get("/")
async def search(
    q: str,
    content_type: str = "all",
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    filters: Optional[str] = Query(None),
    sort: str = Query("relevance"),
    current_user: Optional[User] = Depends(get_current_active_user),
    search_service: SearchService = Depends(get_search_service),
) -> Any:
    """
    Unified search across all content types with smart ranking.
    
    - **q**: Search query
    - **content_type**: Type of content (all, videos, posts, users)
    - **limit**: Maximum results per type
    - **offset**: Pagination offset
    - **filters**: JSON string with additional filters
    - **sort**: Sort order (relevance, recent, popular)
    """
    try:
        # Parse filters if provided
        filter_dict = {}
        if filters:
            import json
            try:
                filter_dict = json.loads(filters)
            except json.JSONDecodeError:
                pass
        
        # Track search analytics
        if current_user:
            await analytics_service.track_event(
                event_type="search_query",
                user_id=str(current_user.id),
                data={
                    "query": q,
                    "content_type": content_type,
                    "filters": filter_dict,
                    "sort": sort
                }
            )
        
        # Route to appropriate search
        if content_type == "videos":
            result = await search_service.search_videos(
                query=q,
                limit=limit,
                offset=offset,
                filters=filter_dict,
                sort_by=sort,
            )
        elif content_type == "posts":
            result = await search_service.search_posts(
                query=q,
                limit=limit,
                offset=offset,
                filters=filter_dict,
                sort_by=sort,
            )
        elif content_type == "users":
            result = await search_service.search_users(
                query=q,
                limit=limit,
                offset=offset,
            )
        else:  # all
            result = await search_service.search_all(
                query=q,
                user_id=current_user.id if current_user else None,
                limit=limit,
                offset=offset,
                filters=filter_dict,
            )
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@router.get("/videos")
async def search_videos(
    q: str,
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    filters: Optional[str] = Query(None),
    sort: str = Query("relevance"),
    current_user: Optional[User] = Depends(get_current_active_user),
    search_service: SearchService = Depends(get_search_service),
) -> Any:
    """
    Search for videos with smart ranking.
    
    - **q**: Search query
    - **limit**: Maximum results
    - **offset**: Pagination offset
    - **filters**: JSON filters (duration, date_range, etc.)
    - **sort**: Sort order (relevance, recent, views, engagement)
    """
    try:
        # Parse filters
        filter_dict = {}
        if filters:
            import json
            try:
                filter_dict = json.loads(filters)
            except json.JSONDecodeError:
                pass
        
        # Track analytics
        if current_user:
            await analytics_service.track_event(
                event_type="video_search",
                user_id=str(current_user.id),
                data={
                    "query": q,
                    "filters": filter_dict,
                    "sort": sort
                }
            )
        
        result = await search_service.search_videos(
            query=q,
            limit=limit,
            offset=offset,
            filters=filter_dict,
            sort_by=sort,
        )
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Video search failed: {str(e)}")


@router.get("/users")
async def search_users(
    q: str,
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    filters: Optional[str] = Query(None),
    sort: str = Query("relevance"),
    current_user: Optional[User] = Depends(get_current_active_user),
    search_service: SearchService = Depends(get_search_service),
) -> Any:
    """
    Search for users by username or email.
    
    - **q**: Search query
    - **limit**: Maximum results
    - **offset**: Pagination offset
    """
    try:
        # Track analytics
        if current_user:
            await analytics_service.track_event(
                event_type="user_search",
                user_id=str(current_user.id),
                data={
                    "query": q,
                }
            )
        
        result = await search_service.search_users(
            query=q,
            limit=limit,
            offset=offset,
        )
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"User search failed: {str(e)}")


@router.get("/suggestions")
async def get_suggestions(
    q: str,
    limit: int = Query(10, ge=1, le=50),
    current_user: Optional[User] = Depends(get_current_active_user),
    search_service: SearchService = Depends(get_search_service),
) -> Any:
    """
    Get search suggestions and autocomplete.
    
    - **q**: Partial search query
    - **limit**: Maximum suggestions
    """
    try:
        result = await search_service.get_suggestions(
            query=q,
            limit=limit,
            user_id=current_user.id if current_user else None,
        )
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get suggestions: {str(e)}")


@router.get("/trending")
async def get_trending_searches(
    time_window: str = Query("24h"),
    limit: int = Query(20, ge=1, le=100),
    current_user: Optional[User] = Depends(get_current_active_user),
    search_service: SearchService = Depends(get_search_service),
) -> Any:
    """
    Get trending search queries.
    
    - **time_window**: Time window (1h, 24h, 7d)
    - **limit**: Maximum results
    """
    try:
        result = await search_service.get_trending_searches(
            limit=limit,
            time_window=time_window,
        )
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get trending searches: {str(e)}")


@router.post("/interaction")
async def record_search_interaction(
    query: str,
    result_id: str,
    interaction_type: str,  # click, view, like, etc.
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
) -> Any:
    """Record search interaction for learning and analytics."""
    try:
        # TODO: Store search interaction data
        # This would help improve search ranking and recommendations
        
        # Track analytics
        await analytics_service.track_event(
            event_type="search_interaction",
            user_id=str(current_user.id),
            data={
                "query": query,
                "result_id": result_id,
                "interaction_type": interaction_type
            }
        )
        
        return {
            "message": "Search interaction recorded successfully",
            "query": query,
            "result_id": result_id,
            "interaction_type": interaction_type
        }
    except Exception:
        raise HTTPException(status_code=500, detail="Failed to record search interaction")


# Hashtag search endpoints

@router.get("/hashtags/{hashtag}/videos")
async def get_videos_by_hashtag(
    hashtag: str,
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    sort: str = Query("recent"),
    current_user: Optional[User] = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
) -> Any:
    """Get videos by hashtag."""
    try:
        # TODO: Implement hashtag video search
        # This would search for videos containing the specific hashtag
        
        return {
            "hashtag": hashtag,
            "videos": [],
            "total": 0,
            "limit": limit,
            "offset": offset,
            "sort": sort
        }
    except Exception:
        raise HTTPException(status_code=500, detail="Failed to get videos by hashtag")


@router.get("/hashtags/trending")
async def get_trending_hashtags(
    time_window: str = Query("24h"),
    limit: int = Query(20, ge=1, le=100),
    current_user: Optional[User] = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
) -> Any:
    """Get trending hashtags."""
    try:
        # TODO: Implement trending hashtags functionality
        # This would analyze hashtag usage patterns over time
        
        return {
            "time_window": time_window,
            "trending_hashtags": [],
            "limit": limit
        }
    except Exception:
        raise HTTPException(status_code=500, detail="Failed to get trending hashtags")


@router.get("/hashtags/{hashtag}/related")
async def get_related_hashtags(
    hashtag: str,
    limit: int = Query(10, ge=1, le=50),
    current_user: Optional[User] = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
) -> Any:
    """Get related hashtags."""
    try:
        # TODO: Implement related hashtags functionality
        # This would find hashtags that are commonly used together
        
        return {
            "hashtag": hashtag,
            "related_hashtags": [],
            "limit": limit
        }
    except Exception:
        raise HTTPException(status_code=500, detail="Failed to get related hashtags")


@router.get("/hashtags/{hashtag}/analytics")
async def get_hashtag_analytics(
    hashtag: str,
    time_window: str = Query("7d"),
    current_user: User = Depends(get_current_active_user),
    search_service: SearchService = Depends(get_search_service),
) -> Any:
    """Get hashtag analytics and performance metrics."""
    try:
        # TODO: Implement detailed hashtag analytics
        
        return {
            "hashtag": hashtag,
            "time_window": time_window,
            "analytics": {
                "usage_count": 0,
                "growth_rate": 0.0,
                "top_videos": [],
                "demographics": {},
                "engagement_metrics": {}
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get hashtag analytics: {str(e)}")


# Recommendation Endpoints

@router.get("/recommendations/videos")
async def get_video_recommendations(
    limit: int = Query(20, ge=1, le=100),
    algorithm: str = Query("hybrid", pattern="^(hybrid|trending|collaborative|content_based)$"),
    current_user: Optional[User] = Depends(get_current_active_user),
    recommendation_service: RecommendationService = Depends(get_recommendation_service),
) -> Any:
    """
    Get personalized video recommendations.
    
    - **limit**: Number of recommendations
    - **algorithm**: Algorithm to use
        - `hybrid`: Combines multiple signals (default)
        - `trending`: Popular videos now
        - `collaborative`: Based on similar users
        - `content_based`: Based on viewing history
    """
    try:
        result = await recommendation_service.get_video_recommendations(
            user_id=current_user.id if current_user else None,
            limit=limit,
            algorithm=algorithm,
        )
        
        # Track analytics
        if current_user:
            await analytics_service.track_event(
                event_type="video_recommendations_viewed",
                user_id=str(current_user.id),
                data={
                    "algorithm": algorithm,
                    "count": result["count"],
                }
            )
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get video recommendations: {str(e)}")


@router.get("/recommendations/feed")
async def get_feed_recommendations(
    limit: int = Query(20, ge=1, le=100),
    algorithm: str = Query("hybrid", pattern="^(hybrid|trending|following)$"),
    current_user: User = Depends(get_current_active_user),
    recommendation_service: RecommendationService = Depends(get_recommendation_service),
) -> Any:
    """
    Get personalized feed recommendations (posts).
    
    - **limit**: Number of recommendations
    - **algorithm**: Algorithm to use
        - `hybrid`: Combines following + trending + discovery
        - `trending`: Trending posts
        - `following`: Posts from followed users
    """
    try:
        result = await recommendation_service.get_feed_recommendations(
            user_id=current_user.id,
            limit=limit,
            algorithm=algorithm,
        )
        
        # Track analytics
        await analytics_service.track_event(
            event_type="feed_recommendations_viewed",
            user_id=str(current_user.id),
            data={
                "algorithm": algorithm,
                "count": result["count"],
            }
        )
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get feed recommendations: {str(e)}")


@router.get("/recommendations/posts")
async def get_post_recommendations(
    limit: int = Query(20, ge=1, le=100),
    algorithm: str = Query("hybrid"),
    current_user: Optional[User] = Depends(get_current_active_user),
    recommendation_service: RecommendationService = Depends(get_recommendation_service),
) -> Any:
    """
    Get personalized post recommendations.
    
    Alias for /recommendations/feed with flexible authentication.
    """
    try:
        if not current_user:
            # For anonymous users, return trending posts
            algorithm = "trending"
        
        result = await recommendation_service.get_feed_recommendations(
            user_id=current_user.id if current_user else None,
            limit=limit,
            algorithm=algorithm,
        )
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get post recommendations: {str(e)}")

