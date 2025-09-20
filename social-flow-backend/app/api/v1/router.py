"""
Main API router for v1 endpoints.

This module defines the main router that includes all v1 API endpoints.
"""

from fastapi import APIRouter

from app.api.v1.endpoints import (
    auth,
    users,
    videos,
    posts,
    comments,
    likes,
    follows,
    ads,
    payments,
    subscriptions,
    notifications,
    analytics,
    search,
    admin,
    moderation,
    ml,
    live_streaming,
)

api_router = APIRouter()

# Include all endpoint routers
api_router.include_router(auth.router, prefix="/auth", tags=["authentication"])
api_router.include_router(users.router, prefix="/users", tags=["users"])
api_router.include_router(videos.router, prefix="/videos", tags=["videos"])
api_router.include_router(posts.router, prefix="/posts", tags=["posts"])
api_router.include_router(comments.router, prefix="/comments", tags=["comments"])
api_router.include_router(likes.router, prefix="/likes", tags=["likes"])
api_router.include_router(follows.router, prefix="/follows", tags=["follows"])
api_router.include_router(ads.router, prefix="/ads", tags=["ads"])
api_router.include_router(payments.router, prefix="/payments", tags=["payments"])
api_router.include_router(subscriptions.router, prefix="/subscriptions", tags=["subscriptions"])
api_router.include_router(notifications.router, prefix="/notifications", tags=["notifications"])
api_router.include_router(analytics.router, prefix="/analytics", tags=["analytics"])
api_router.include_router(search.router, prefix="/search", tags=["search"])
api_router.include_router(admin.router, prefix="/admin", tags=["admin"])
api_router.include_router(moderation.router, prefix="/moderation", tags=["moderation"])
api_router.include_router(ml.router, prefix="/ml", tags=["ml-ai"])
api_router.include_router(live_streaming.router, prefix="/live", tags=["live-streaming"])
