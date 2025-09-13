# Package initializer for offline-analytics
"""
Offline Analytics package initializer.

Exposes router and Celery app so they can be mounted by the main app.
"""
from .routes import router as offline_analytics_router
from .tasks import celery_app

__all__ = ["offline_analytics_router", "celery_app"]
