# Package initializer for content-caching
"""
Offline Content Caching package initializer.

Exports router and Celery app where appropriate.
"""
from .routes import router as content_caching_router
from .tasks import celery_app

__all__ = ["content_caching_router", "celery_app"]
