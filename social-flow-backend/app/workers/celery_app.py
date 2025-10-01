"""
Celery application configuration.

This module configures Celery for background task processing.
"""

from celery import Celery
from app.core.config import settings

# Create Celery instance
celery_app = Celery(
    "social_flow",
    broker=settings.REDIS_URL,
    backend=settings.REDIS_URL,
    include=[
        "app.modules.videos.video_processing",
        "app.modules.ml.ai_processing",
        "app.modules.analytics.analytics_processing",
        "app.modules.notifications.notification_processing",
        "app.modules.notifications.email_processing",
    ]
)

# Celery configuration
celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_time_limit=30 * 60,  # 30 minutes
    task_soft_time_limit=25 * 60,  # 25 minutes
    worker_prefetch_multiplier=1,
    worker_max_tasks_per_child=1000,
    result_expires=3600,  # 1 hour
    task_routes={
        "app.workers.video_processing.*": {"queue": "video_processing"},
        "app.workers.ai_processing.*": {"queue": "ai_processing"},
        "app.workers.analytics_processing.*": {"queue": "analytics_processing"},
        "app.workers.notification_processing.*": {"queue": "notifications"},
        "app.workers.email_processing.*": {"queue": "email"},
    },
    task_annotations={
        "*": {"rate_limit": "100/m"},
        "app.workers.video_processing.*": {"rate_limit": "10/m"},
        "app.workers.ai_processing.*": {"rate_limit": "20/m"},
    },
)

# Periodic tasks
celery_app.conf.beat_schedule = {
    "cleanup-expired-sessions": {
        "task": "app.workers.analytics_processing.cleanup_expired_sessions",
        "schedule": 3600.0,  # Every hour
    },
    "generate-daily-reports": {
        "task": "app.workers.analytics_processing.generate_daily_reports",
        "schedule": 86400.0,  # Every day at midnight
    },
    "update-trending-content": {
        "task": "app.workers.ai_processing.update_trending_content",
        "schedule": 1800.0,  # Every 30 minutes
    },
    "process-pending-videos": {
        "task": "app.workers.video_processing.process_pending_videos",
        "schedule": 300.0,  # Every 5 minutes
    },
}

if __name__ == "__main__":
    celery_app.start()
