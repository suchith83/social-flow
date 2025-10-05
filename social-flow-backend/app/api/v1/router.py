"""
Main API router for v1 endpoints.

This module defines the main router that includes all v1 API endpoints.
"""

from fastapi import APIRouter

from app.api.v1.endpoints import (
    auth as auth_endpoints,
    users as users_endpoints,
    videos as videos_endpoints,
    social as social_endpoints,
    payments as payments_endpoints,
    notifications as notifications_endpoints,
    search,
    admin,
    moderation,
    health,
    users_new,
    videos_new,
    posts_new,
    ai_pipelines,
)

# Import from modules
from app.auth.api import auth, subscriptions, stripe_connect
from app.users.api import users, follows
from app.videos.api import videos
from app.posts.api import posts, comments, likes
# Use the newer livestream routes instead of legacy live module
from app.livestream.routes import livestream_routes as livestream
from app.ads.api import ads
from app.payments.api import stripe_payments, stripe_subscriptions, stripe_webhooks
from app.notifications.api import notifications
from app.ml.api import ml
from app.analytics.api import analytics
from app.analytics.routes import analytics_routes as analytics_enhanced

api_router = APIRouter()

# Core authentication and user management
api_router.include_router(auth_endpoints.router, prefix="/auth", tags=["authentication"])
api_router.include_router(users_endpoints.router, prefix="/users", tags=["users"])
api_router.include_router(videos_endpoints.router, prefix="/videos", tags=["videos"])
api_router.include_router(social_endpoints.router, prefix="/social", tags=["social"])
# Payment endpoints already include /payments prefix in their paths
api_router.include_router(payments_endpoints.router, tags=["payments"])
api_router.include_router(notifications_endpoints.router, tags=["notifications"])

# Include all endpoint routers
# Legacy auth router (keeping for backward compatibility)
api_router.include_router(auth.router, prefix="/auth/legacy", tags=["authentication-legacy"])

# NEW: Clean Architecture Endpoints (v2 - coexisting with legacy)
api_router.include_router(users_new.router, prefix="/v2/users", tags=["users-v2"])
api_router.include_router(videos_new.router, prefix="/v2/videos", tags=["videos-v2"])
api_router.include_router(posts_new.router, prefix="/v2/posts", tags=["posts-v2"])

# Legacy endpoints (will be deprecated)
api_router.include_router(users.router, prefix="/users/legacy", tags=["users-legacy"])
api_router.include_router(videos.router, prefix="/videos/legacy", tags=["videos-legacy"])
api_router.include_router(posts.router, prefix="/posts", tags=["posts"])
api_router.include_router(comments.router, prefix="/comments", tags=["comments"])
api_router.include_router(likes.router, prefix="/likes", tags=["likes"])
api_router.include_router(follows.router, prefix="/follows", tags=["follows"])
api_router.include_router(ads.router, prefix="/ads", tags=["ads"])
# Commented out: Conflicts with payments_endpoints.router which is more comprehensive
# api_router.include_router(payments.router, prefix="/payments", tags=["payments"])
api_router.include_router(subscriptions.router, prefix="/subscriptions", tags=["subscriptions"])
api_router.include_router(notifications.router, prefix="/notifications", tags=["notifications"])
api_router.include_router(analytics.router, prefix="/analytics", tags=["analytics"])
api_router.include_router(analytics_enhanced.router, prefix="/analytics", tags=["analytics-enhanced"])
api_router.include_router(search.router, prefix="/search", tags=["search"])
api_router.include_router(admin.router, prefix="/admin", tags=["admin"])
api_router.include_router(moderation.router, prefix="/moderation", tags=["moderation"])
api_router.include_router(ml.router, prefix="/ml", tags=["ml-ai"])
# AI Pipelines & Advanced Recommendations
api_router.include_router(ai_pipelines.router, prefix="/ai", tags=["ai-pipelines"])
# Livestream API (mounted without extra prefix; router provides '/livestream')
api_router.include_router(livestream.router)

# Stripe Payment Integration
api_router.include_router(stripe_payments.router, tags=["Payments"])
api_router.include_router(stripe_subscriptions.router, tags=["Subscriptions"])
api_router.include_router(stripe_connect.router, tags=["Creator Payouts"])
api_router.include_router(stripe_webhooks.router, tags=["Webhooks"])

# Health & Monitoring
api_router.include_router(health.router, tags=["Health"])
