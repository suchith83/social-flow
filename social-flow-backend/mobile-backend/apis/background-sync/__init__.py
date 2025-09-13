# Package initializer or entrypoint for background sync
"""
Background Sync API package initializer.
Ensures Celery worker discovery and lazy loading of sync services.
"""

from .routes import router as background_sync_router
from .tasks import celery_app

__all__ = [
    "background_sync_router",
    "celery_app",
]
