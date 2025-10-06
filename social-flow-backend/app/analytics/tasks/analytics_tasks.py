"""
Analytics Background Tasks.

Celery tasks for periodic metric calculation, aggregation,
and data cleanup.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, Any

from celery import Task
from sqlalchemy import select

try:
    from app.core.celery_app import celery_app  # type: ignore
except Exception:  # pragma: no cover - fallback for test environment
    class _DummyCelery:
        def task(self, *dargs, **dkwargs):  # noqa: D401
            def decorator(fn):
                return fn
            return decorator
    celery_app = _DummyCelery()
from app.core.database import async_session_maker
from app.analytics.services.enhanced_service import EnhancedAnalyticsService
from app.videos.models.video import Video
from app.models.user import User

logger = logging.getLogger(__name__)


@celery_app.task(bind=True, name="analytics.calculate_video_metrics")
def calculate_video_metrics_task(self: Task, video_id: str) -> Dict[str, Any]:
    """
    Calculate comprehensive metrics for a video.
    
    This task is triggered when:
    - A video receives new views
    - Periodic recalculation is needed
    - Manual refresh is requested
    """
    try:
        import asyncio
        
        async def _calculate():
            async with async_session_maker() as db:
                service = EnhancedAnalyticsService(db)
                metrics = await service.calculate_video_metrics(video_id)
                
                return {
                    "video_id": str(metrics.video_id),
                    "total_views": metrics.total_views,
                    "engagement_score": metrics.engagement_score,
                    "quality_score": metrics.quality_score
                }
        
        result = asyncio.run(_calculate())
        logger.info(f"Video metrics calculated: {video_id}")
        return result
        
    except Exception as e:
        logger.error(f"Error calculating video metrics: {e}")
        self.retry(exc=e, countdown=60, max_retries=3)


@celery_app.task(bind=True, name="analytics.calculate_user_metrics")
def calculate_user_metrics_task(self: Task, user_id: str) -> Dict[str, Any]:
    """
    Calculate comprehensive behavior metrics for a user.
    
    This task is triggered when:
    - User activity occurs
    - Periodic recalculation is needed
    - Dashboard is accessed
    """
    try:
        import asyncio
        
        async def _calculate():
            async with async_session_maker() as db:
                service = EnhancedAnalyticsService(db)
                metrics = await service.calculate_user_metrics(user_id)
                
                return {
                    "user_id": str(metrics.user_id),
                    "total_videos_watched": metrics.total_videos_watched,
                    "creator_status": metrics.creator_status,
                    "activity_score": metrics.activity_score
                }
        
        result = asyncio.run(_calculate())
        logger.info(f"User metrics calculated: {user_id}")
        return result
        
    except Exception as e:
        logger.error(f"Error calculating user metrics: {e}")
        self.retry(exc=e, countdown=60, max_retries=3)


@celery_app.task(bind=True, name="analytics.calculate_revenue_metrics")
def calculate_revenue_metrics_task(
    self: Task,
    date_str: str,
    period_type: str = "daily",
    user_id: str = None
) -> Dict[str, Any]:
    """
    Calculate revenue metrics for a specific period.
    
    This task is scheduled to run:
    - Daily at midnight for previous day
    - Weekly on Mondays
    - Monthly on 1st of month
    """
    try:
        import asyncio
        
        async def _calculate():
            async with async_session_maker() as db:
                service = EnhancedAnalyticsService(db)
                date = datetime.fromisoformat(date_str)
                metrics = await service.calculate_revenue_metrics(date, period_type, user_id)
                
                return {
                    "date": metrics.date.isoformat(),
                    "period_type": metrics.period_type,
                    "total_revenue": metrics.total_revenue,
                    "subscription_revenue": metrics.subscription_revenue,
                    "donation_revenue": metrics.donation_revenue
                }
        
        result = asyncio.run(_calculate())
        logger.info(f"Revenue metrics calculated: {date_str} ({period_type})")
        return result
        
    except Exception as e:
        logger.error(f"Error calculating revenue metrics: {e}")
        self.retry(exc=e, countdown=60, max_retries=3)


@celery_app.task(bind=True, name="analytics.recalculate_all_video_metrics")
def recalculate_all_video_metrics_task(self: Task) -> Dict[str, Any]:
    """
    Recalculate metrics for all videos.
    
    This task runs periodically (e.g., every 6 hours) to ensure
    all video metrics are up to date. Only recalculates videos
    with recent activity or stale metrics.
    """
    try:
        import asyncio
        
        async def _recalculate():
            async with async_session_maker() as db:
                # Get videos with views in last 24 hours
                # Or videos with stale metrics (>24 hours old)
                stmt = select(Video).where(Video.views_count > 0).limit(1000)
                result = await db.execute(stmt)
                videos = result.scalars().all()
                
                service = EnhancedAnalyticsService(db)
                calculated_count = 0
                
                for video in videos:
                    try:
                        await service.calculate_video_metrics(str(video.id))
                        calculated_count += 1
                    except Exception as e:
                        logger.error(f"Error calculating metrics for video {video.id}: {e}")
                        continue
                
                return {
                    "total_videos": len(videos),
                    "calculated": calculated_count,
                    "timestamp": datetime.utcnow().isoformat()
                }
        
        result = asyncio.run(_recalculate())
        logger.info(f"Bulk video metrics recalculation complete: {result['calculated']} videos")
        return result
        
    except Exception as e:
        logger.error(f"Error in bulk video metrics recalculation: {e}")
        self.retry(exc=e, countdown=300, max_retries=2)


@celery_app.task(bind=True, name="analytics.recalculate_all_user_metrics")
def recalculate_all_user_metrics_task(self: Task) -> Dict[str, Any]:
    """
    Recalculate metrics for all active users.
    
    This task runs periodically (e.g., daily) to ensure
    all user behavior metrics are up to date. Focuses on
    users with recent activity.
    """
    try:
        import asyncio
        
        async def _recalculate():
            async with async_session_maker() as db:
                # Get users active in last 30 days
                stmt = select(User).where(User.is_active).limit(10000)
                result = await db.execute(stmt)
                users = result.scalars().all()
                
                service = EnhancedAnalyticsService(db)
                calculated_count = 0
                
                for user in users:
                    try:
                        await service.calculate_user_metrics(str(user.id))
                        calculated_count += 1
                    except Exception as e:
                        logger.error(f"Error calculating metrics for user {user.id}: {e}")
                        continue
                
                return {
                    "total_users": len(users),
                    "calculated": calculated_count,
                    "timestamp": datetime.utcnow().isoformat()
                }
        
        result = asyncio.run(_recalculate())
        logger.info(f"Bulk user metrics recalculation complete: {result['calculated']} users")
        return result
        
    except Exception as e:
        logger.error(f"Error in bulk user metrics recalculation: {e}")
        self.retry(exc=e, countdown=300, max_retries=2)


@celery_app.task(bind=True, name="analytics.calculate_daily_revenue")
def calculate_daily_revenue_task(self: Task) -> Dict[str, Any]:
    """
    Calculate revenue metrics for yesterday.
    
    This task runs daily at midnight to calculate the previous day's
    revenue metrics. Results are stored for fast dashboard access.
    """
    try:
        yesterday = (datetime.utcnow() - timedelta(days=1)).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        
        # Calculate platform-wide revenue
        result = calculate_revenue_metrics_task.delay(
            date_str=yesterday.isoformat(),
            period_type="daily",
            user_id=None
        )
        
        logger.info(f"Daily revenue calculation triggered for {yesterday.date()}")
        return {
            "date": yesterday.isoformat(),
            "task_id": result.id
        }
        
    except Exception as e:
        logger.error(f"Error triggering daily revenue calculation: {e}")
        self.retry(exc=e, countdown=300, max_retries=3)


@celery_app.task(bind=True, name="analytics.calculate_weekly_revenue")
def calculate_weekly_revenue_task(self: Task) -> Dict[str, Any]:
    """
    Calculate revenue metrics for last week.
    
    This task runs weekly on Mondays to calculate the previous week's
    revenue metrics.
    """
    try:
        # Get start of last week (Monday)
        today = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
        days_since_monday = today.weekday()
        last_monday = today - timedelta(days=days_since_monday + 7)
        
        # Calculate platform-wide revenue
        result = calculate_revenue_metrics_task.delay(
            date_str=last_monday.isoformat(),
            period_type="weekly",
            user_id=None
        )
        
        logger.info(f"Weekly revenue calculation triggered for week of {last_monday.date()}")
        return {
            "week_start": last_monday.isoformat(),
            "task_id": result.id
        }
        
    except Exception as e:
        logger.error(f"Error triggering weekly revenue calculation: {e}")
        self.retry(exc=e, countdown=300, max_retries=3)


@celery_app.task(bind=True, name="analytics.cleanup_old_view_sessions")
def cleanup_old_view_sessions_task(self: Task, days: int = 90) -> Dict[str, Any]:
    """
    Clean up old view session records.
    
    This task runs monthly to delete view session records older than
    the specified number of days. Metrics are preserved, only raw
    session data is deleted.
    """
    try:
        import asyncio
        
        async def _cleanup():
            from app.analytics.models.extended import ViewSession
            
            async with async_session_maker() as db:
                cutoff_date = datetime.utcnow() - timedelta(days=days)
                
                # Delete old sessions
                stmt = select(ViewSession).where(ViewSession.created_at < cutoff_date)
                result = await db.execute(stmt)
                old_sessions = result.scalars().all()
                
                deleted_count = 0
                for session in old_sessions:
                    await db.delete(session)
                    deleted_count += 1
                
                await db.commit()
                
                return {
                    "deleted_count": deleted_count,
                    "cutoff_date": cutoff_date.isoformat(),
                    "timestamp": datetime.utcnow().isoformat()
                }
        
        result = asyncio.run(_cleanup())
        logger.info(f"Old view sessions cleaned up: {result['deleted_count']} records deleted")
        return result
        
    except Exception as e:
        logger.error(f"Error cleaning up old view sessions: {e}")
        self.retry(exc=e, countdown=600, max_retries=2)


# ========== Periodic Task Setup ==========

def setup_analytics_periodic_tasks():
    """
    Configure periodic analytics tasks.
    
    Call this function during application startup to register
    all periodic analytics tasks with Celery Beat.
    """
    from celery.schedules import crontab
    
    celery_app.conf.beat_schedule.update({
        # Recalculate all video metrics every 6 hours
        'recalculate-video-metrics': {
            'task': 'analytics.recalculate_all_video_metrics',
            'schedule': crontab(hour='*/6', minute='0'),
        },
        
        # Recalculate all user metrics daily at 2 AM
        'recalculate-user-metrics': {
            'task': 'analytics.recalculate_all_user_metrics',
            'schedule': crontab(hour='2', minute='0'),
        },
        
        # Calculate daily revenue at midnight
        'calculate-daily-revenue': {
            'task': 'analytics.calculate_daily_revenue',
            'schedule': crontab(hour='0', minute='5'),  # 5 minutes past midnight
        },
        
        # Calculate weekly revenue on Mondays at 1 AM
        'calculate-weekly-revenue': {
            'task': 'analytics.calculate_weekly_revenue',
            'schedule': crontab(day_of_week='monday', hour='1', minute='0'),
        },
        
        # Cleanup old view sessions monthly on 1st at 3 AM
        'cleanup-old-view-sessions': {
            'task': 'analytics.cleanup_old_view_sessions',
            'schedule': crontab(day_of_month='1', hour='3', minute='0'),
            'args': (90,)  # Delete sessions older than 90 days
        },
    })
    
    logger.info("Analytics periodic tasks configured")


# Note: Call setup_analytics_periodic_tasks() in your app initialization
# Example: from app.analytics.tasks.analytics_tasks import setup_analytics_periodic_tasks
#          setup_analytics_periodic_tasks()


