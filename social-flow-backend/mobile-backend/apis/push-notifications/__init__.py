# Package initializer for push-notifications
"""
Push Notifications package initializer.

Exports router and celery app so they can be mounted/inspected by the main app.
"""
from .routes import router as push_router
from .tasks import celery_app

__all__ = ["push_router", "celery_app"]
