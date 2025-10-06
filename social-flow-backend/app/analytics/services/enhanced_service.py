"""
Enhanced Analytics Service for Comprehensive Metrics and Reporting.

This service provides video metrics, user behavior tracking, revenue reporting,
and dashboard data aggregation.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from decimal import Decimal

from sqlalchemy import select, func, and_, or_, desc
from sqlalchemy.ext.asyncio import AsyncSession

from app.analytics.models.extended import (
    VideoMetrics, UserBehaviorMetrics, RevenueMetrics,
    AggregatedMetrics, ViewSession
)
from app.videos.models.video import Video
from app.auth.models.user import User
# Payments models are optional in some test scenarios; guard imports
try:
    from app.payments.models.subscription import Subscription  # type: ignore
except Exception:  # pragma: no cover
    class Subscription:  # minimal stub
        id = None
        created_at = None
        status = None
        creator_id = None

try:
    from app.payments.models.transaction import Transaction  # type: ignore
except Exception:  # pragma: no cover
    class Transaction:  # minimal stub
        id = None
        created_at = None
        status = None
        user_id = None
        transaction_type = None
        amount = 0

logger = logging.getLogger(__name__)


class EnhancedAnalyticsService:
    """Enhanced analytics service with comprehensive metric calculation."""
    
    def __init__(self, db: AsyncSession):
        self.db = db
    
    # ========== Video Metrics ==========
    
    async def calculate_video_metrics(self, video_id: str) -> VideoMetrics:
        """Calculate comprehensive metrics for a video."""
        try:
            import uuid as uuid_module
            video_uuid = uuid_module.UUID(video_id) if isinstance(video_id, str) else video_id
            
            # Check if metrics exist
            stmt = select(VideoMetrics).where(VideoMetrics.video_id == video_uuid)
            result = await self.db.execute(stmt)
            metrics = result.scalar_one_or_none()
            
            if not metrics:
                metrics = VideoMetrics(video_id=video_uuid)
                self.db.add(metrics)
            
            # Get all view sessions for this video
            stmt = select(ViewSession).where(ViewSession.video_id == video_uuid)
            result = await self.db.execute(stmt)
            sessions = result.scalars().all()
            
            if not sessions:
                await self.db.commit()
                return metrics
            
            # Calculate view metrics
            metrics.total_views = len(sessions)
            metrics.unique_views = len(set(s.user_id for s in sessions if s.user_id))
            
            # Time-based views
            now = datetime.utcnow()
            metrics.views_24h = len([s for s in sessions if s.created_at > now - timedelta(days=1)])
            metrics.views_7d = len([s for s in sessions if s.created_at > now - timedelta(days=7)])
            metrics.views_30d = len([s for s in sessions if s.created_at > now - timedelta(days=30)])
            
            # Watch time metrics
            metrics.total_watch_time = sum(s.duration for s in sessions)
            metrics.avg_watch_time = metrics.total_watch_time / len(sessions) if sessions else 0
            
            # Watch percentage
            watch_percentages = [s.watch_percentage for s in sessions]
            metrics.avg_watch_percentage = sum(watch_percentages) / len(watch_percentages) if watch_percentages else 0
            
            # Completion rate
            completed_sessions = [s for s in sessions if s.completed]
            metrics.completion_rate = (len(completed_sessions) / len(sessions) * 100) if sessions else 0
            
            # Get video for engagement metrics
            stmt = select(Video).where(Video.id == video_uuid)
            result = await self.db.execute(stmt)
            video = result.scalar_one_or_none()
            
            if video:
                # Support multiple possible attribute names across schema variants
                metrics.total_likes = (
                    getattr(video, 'likes_count', None) or
                    getattr(video, 'like_count', None) or 0
                )
                metrics.total_comments = (
                    getattr(video, 'comments_count', None) or
                    getattr(video, 'comment_count', None) or 0
                )
                metrics.total_shares = (
                    getattr(video, 'shares_count', None) or
                    getattr(video, 'share_count', None) or 0
                )
                
                # Calculate engagement rates (per 100 views)
                if metrics.total_views > 0:
                    metrics.like_rate = (metrics.total_likes / metrics.total_views) * 100
                    metrics.comment_rate = (metrics.total_comments / metrics.total_views) * 100
                    metrics.share_rate = (metrics.total_shares / metrics.total_views) * 100
            
            # Calculate retention curve (10% intervals)
            retention_curve = []
            video_duration = sessions[0].video_duration if sessions else 0
            if video_duration > 0:
                for i in range(0, 101, 10):
                    time_point = (video_duration * i) / 100
                    viewers_at_point = len([s for s in sessions if s.duration >= time_point])
                    retention = (viewers_at_point / len(sessions) * 100) if sessions else 0
                    retention_curve.append({"time": i, "retention": round(retention, 2)})
            metrics.retention_curve = retention_curve
            
            # Device breakdown
            device_counts = {}
            for session in sessions:
                device = session.device_type or 'unknown'
                device_counts[device] = device_counts.get(device, 0) + 1
            metrics.device_breakdown = device_counts
            
            # OS breakdown
            os_counts = {}
            for session in sessions:
                os = session.os or 'unknown'
                os_counts[os] = os_counts.get(os, 0) + 1
            metrics.os_breakdown = os_counts
            
            # Country breakdown
            country_counts = {}
            for session in sessions:
                country = session.country or 'unknown'
                country_counts[country] = country_counts.get(country, 0) + 1
            metrics.country_breakdown = country_counts
            
            # Top countries (sorted)
            top_countries = sorted(
                [{"country": k, "views": v} for k, v in country_counts.items()],
                key=lambda x: x["views"],
                reverse=True
            )[:10]
            metrics.top_countries = top_countries
            
            # Traffic sources
            traffic_counts = {}
            for session in sessions:
                source = session.traffic_source or 'direct'
                traffic_counts[source] = traffic_counts.get(source, 0) + 1
            metrics.traffic_sources = traffic_counts
            
            # Calculate engagement score (0-100)
            engagement_score = 0
            if metrics.total_views > 0:
                # 40% - Like rate (normalized to 0-40)
                engagement_score += min(metrics.like_rate * 4, 40)
                # 30% - Comment rate (normalized to 0-30)
                engagement_score += min(metrics.comment_rate * 6, 30)
                # 30% - Share rate (normalized to 0-30)
                engagement_score += min(metrics.share_rate * 10, 30)
            metrics.engagement_score = round(engagement_score, 2)
            
            # Calculate virality score (0-100)
            virality_score = 0
            if metrics.total_views > 0:
                # Based on share rate and view growth
                virality_score = min(metrics.share_rate * 20, 100)
            metrics.virality_score = round(virality_score, 2)
            
            # Calculate quality score (0-100)
            quality_score = (
                (metrics.avg_watch_percentage * 0.4) +  # 40% weight
                (metrics.completion_rate * 0.3) +        # 30% weight
                (min(metrics.engagement_score, 100) * 0.3)  # 30% weight
            )
            metrics.quality_score = round(quality_score, 2)
            
            metrics.last_calculated_at = datetime.utcnow()
            
            await self.db.commit()
            await self.db.refresh(metrics)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating video metrics: {e}")
            await self.db.rollback()
            raise
    
    async def get_video_metrics(self, video_id: str) -> Optional[VideoMetrics]:
        """Get video metrics, calculating if necessary."""
        import uuid as uuid_module
        video_uuid = uuid_module.UUID(video_id) if isinstance(video_id, str) else video_id
        
        stmt = select(VideoMetrics).where(VideoMetrics.video_id == video_uuid)
        result = await self.db.execute(stmt)
        metrics = result.scalar_one_or_none()
        
        if not metrics or (datetime.utcnow() - metrics.last_calculated_at).seconds > 3600:
            # Recalculate if older than 1 hour
            metrics = await self.calculate_video_metrics(video_id)
        
        return metrics
    
    # ========== User Behavior Metrics ==========
    
    async def calculate_user_metrics(self, user_id: str) -> UserBehaviorMetrics:
        """Calculate comprehensive metrics for a user."""
        try:
            import uuid as uuid_module
            user_uuid = uuid_module.UUID(user_id) if isinstance(user_id, str) else user_id
            
            # Check if metrics exist
            stmt = select(UserBehaviorMetrics).where(UserBehaviorMetrics.user_id == user_uuid)
            result = await self.db.execute(stmt)
            metrics = result.scalar_one_or_none()
            
            if not metrics:
                metrics = UserBehaviorMetrics(user_id=user_uuid)
                self.db.add(metrics)
            
            # Get user
            stmt = select(User).where(User.id == user_uuid)
            result = await self.db.execute(stmt)
            user = result.scalar_one_or_none()
            
            if not user:
                await self.db.commit()
                return metrics
            
            # Get view sessions for this user
            stmt = select(ViewSession).where(ViewSession.user_id == user_uuid)
            result = await self.db.execute(stmt)
            view_sessions = result.scalars().all()
            
            # Content consumption metrics
            metrics.total_videos_watched = len(view_sessions)
            metrics.total_watch_time = sum(s.duration for s in view_sessions)
            
            now = datetime.utcnow()
            metrics.videos_watched_30d = len([s for s in view_sessions if s.created_at > now - timedelta(days=30)])
            
            if metrics.videos_watched_30d > 0:
                metrics.avg_daily_watch_time = metrics.total_watch_time / 30
            
            # Get user's videos if they are a creator.
            # Unified model uses owner_id. Allow fallback to user_id if present.
            if hasattr(Video, 'owner_id'):
                stmt = select(Video).where(Video.owner_id == user_uuid)
            elif hasattr(Video, 'user_id'):
                stmt = select(Video).where(Video.user_id == user_uuid)
            else:
                stmt = select(Video).where(False)
            result = await self.db.execute(stmt)
            user_videos = result.scalars().all()
            
            metrics.total_videos_uploaded = len(user_videos)
            metrics.videos_uploaded_30d = len([v for v in user_videos if v.created_at > now - timedelta(days=30)])
            
            if user_videos:
                metrics.creator_status = True
                
                # Calculate creator metrics
                metrics.total_video_views = 0
                metrics.total_video_likes = 0
                for v in user_videos:
                    metrics.total_video_views += (
                        getattr(v, 'views_count', None) or
                        getattr(v, 'view_count', None) or 0
                    )
                    metrics.total_video_likes += (
                        getattr(v, 'likes_count', None) or
                        getattr(v, 'like_count', None) or 0
                    )
                
                if len(user_videos) > 0:
                    metrics.avg_video_performance = metrics.total_video_views / len(user_videos)
            
            # Social metrics (assuming follower relationships exist)
            # Note: This would need to be adjusted based on actual follower model
            metrics.followers_count = (
                getattr(user, 'followers_count', None) or
                getattr(user, 'follower_count', None) or 0
            )
            metrics.following_count = (
                getattr(user, 'following_count', None) or
                getattr(user, 'follow_count', None) or 0
            )
            
            # Calculate engagement metrics
            if metrics.total_video_views > 0:
                metrics.engagement_rate = (metrics.total_video_likes / metrics.total_video_views) * 100
            
            transactions = []
            try:
                if hasattr(Transaction, '__table__') and getattr(Transaction, '__table__') is not None:
                    stmt = select(Transaction).where(
                        and_(
                            Transaction.user_id == user_uuid,
                            Transaction.status == 'completed'
                        )
                    )
                    result = await self.db.execute(stmt)
                    transactions = result.scalars().all()
            except Exception:
                # Gracefully ignore if transaction model not available in test context
                transactions = []
            
            # Revenue as creator (earnings)
            creator_transactions = [t for t in transactions if t.transaction_type == 'payout']
            metrics.total_earnings = sum(float(t.amount) for t in creator_transactions)
            
            recent_earnings = [t for t in creator_transactions if t.created_at > now - timedelta(days=30)]
            metrics.earnings_30d = sum(float(t.amount) for t in recent_earnings)
            
            # Spending as user
            spending_transactions = [t for t in transactions if t.transaction_type in ('subscription', 'donation')]
            metrics.total_spent = sum(float(t.amount) for t in spending_transactions)
            
            recent_spending = [t for t in spending_transactions if t.created_at > now - timedelta(days=30)]
            metrics.spent_30d = sum(float(t.amount) for t in recent_spending)
            
            # Calculate user scores
            # Activity score (0-100) based on session frequency
            if metrics.sessions_30d > 0:
                metrics.activity_score = min((metrics.sessions_30d / 30) * 100, 100)
            
            # Creator score (0-100) based on content and views
            if metrics.creator_status:
                video_score = min((metrics.total_videos_uploaded / 10) * 50, 50)
                view_score = min((metrics.total_video_views / 10000) * 50, 50)
                metrics.creator_score = video_score + view_score
            
            # Engagement score (0-100)
            metrics.engagement_score = min(metrics.engagement_rate * 10, 100)
            
            # Loyalty score (0-100) based on activity and spending
            loyalty_score = 0
            if metrics.sessions_30d > 15:
                loyalty_score += 50
            elif metrics.sessions_30d > 5:
                loyalty_score += 30
            
            if metrics.spent_30d > 0:
                loyalty_score += 50
            
            metrics.loyalty_score = min(loyalty_score, 100)
            
            metrics.last_calculated_at = datetime.utcnow()
            
            await self.db.commit()
            await self.db.refresh(metrics)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating user metrics: {e}")
            await self.db.rollback()
            raise
    
    async def get_user_metrics(self, user_id: str) -> Optional[UserBehaviorMetrics]:
        """Get user behavior metrics, calculating if necessary."""
        import uuid as uuid_module
        user_uuid = uuid_module.UUID(user_id) if isinstance(user_id, str) else user_id
        stmt = select(UserBehaviorMetrics).where(UserBehaviorMetrics.user_id == user_uuid)
        result = await self.db.execute(stmt)
        metrics = result.scalar_one_or_none()
        
        if not metrics or (datetime.utcnow() - metrics.last_calculated_at).seconds > 3600:
            # Recalculate if older than 1 hour
            metrics = await self.calculate_user_metrics(str(user_uuid))
        
        return metrics
    
    # ========== Revenue Metrics ==========
    
    async def calculate_revenue_metrics(
        self,
        date: datetime,
        period_type: str = "daily",
        user_id: Optional[str] = None
    ) -> RevenueMetrics:
        """Calculate revenue metrics for a specific period."""
        try:
            # Check if metrics exist
            stmt = select(RevenueMetrics).where(
                and_(
                    RevenueMetrics.date == date,
                    RevenueMetrics.period_type == period_type,
                    RevenueMetrics.user_id == user_id
                )
            )
            result = await self.db.execute(stmt)
            metrics = result.scalar_one_or_none()
            
            if not metrics:
                metrics = RevenueMetrics(
                    date=date,
                    period_type=period_type,
                    user_id=user_id
                )
                self.db.add(metrics)
            
            # Calculate date range based on period type
            if period_type == "daily":
                start_date = date
                end_date = date + timedelta(days=1)
            elif period_type == "weekly":
                start_date = date
                end_date = date + timedelta(days=7)
            elif period_type == "monthly":
                start_date = date
                # Approximate month
                end_date = date + timedelta(days=30)
            else:
                start_date = date
                end_date = datetime.utcnow()
            
            # Build transaction query
            transaction_filters = [
                Transaction.created_at >= start_date,
                Transaction.created_at < end_date,
                Transaction.status == 'completed'
            ]
            
            if user_id:
                transaction_filters.append(Transaction.user_id == user_id)
            
            stmt = select(Transaction).where(and_(*transaction_filters))
            result = await self.db.execute(stmt)
            transactions = result.scalars().all()
            
            # Calculate subscription revenue
            subscription_txns = [t for t in transactions if t.transaction_type == 'subscription']
            metrics.subscription_revenue = sum(float(t.amount) for t in subscription_txns)
            
            # Calculate donation revenue
            donation_txns = [t for t in transactions if t.transaction_type == 'donation']
            metrics.donation_revenue = sum(float(t.amount) for t in donation_txns)
            metrics.total_donations = len(donation_txns)
            if donation_txns:
                metrics.avg_donation_amount = metrics.donation_revenue / len(donation_txns)
            
            # Calculate total revenue
            metrics.total_revenue = sum(float(t.amount) for t in transactions)
            metrics.gross_revenue = metrics.total_revenue
            
            # Platform fee (assume 10%)
            metrics.platform_fee = metrics.gross_revenue * 0.1
            metrics.net_revenue = metrics.gross_revenue - metrics.platform_fee
            
            # Transaction metrics
            metrics.total_transactions = len(transactions)
            metrics.successful_transactions = len(transactions)
            
            # Get subscription data
            subscription_filters = [
                Subscription.created_at >= start_date,
                Subscription.created_at < end_date
            ]
            
            if user_id:
                subscription_filters.append(Subscription.creator_id == user_id)
            
            stmt = select(Subscription).where(and_(*subscription_filters))
            result = await self.db.execute(stmt)
            subscriptions = result.scalars().all()
            
            metrics.new_subscriptions = len([s for s in subscriptions if s.status == 'active'])
            metrics.canceled_subscriptions = len([s for s in subscriptions if s.status == 'canceled'])
            
            # Calculate active subscriptions (current total)
            stmt = select(func.count(Subscription.id)).where(
                Subscription.status == 'active'
            )
            if user_id:
                stmt = stmt.where(Subscription.creator_id == user_id)
            
            result = await self.db.execute(stmt)
            metrics.active_subscriptions = result.scalar() or 0
            
            # Calculate MRR (Monthly Recurring Revenue)
            if period_type == "monthly":
                metrics.subscription_mrr = metrics.subscription_revenue
            else:
                # Extrapolate to monthly
                metrics.subscription_mrr = metrics.subscription_revenue * 30
            
            # Calculate ARPU
            stmt = select(func.count(func.distinct(Transaction.user_id))).where(
                and_(*transaction_filters)
            )
            result = await self.db.execute(stmt)
            paying_users = result.scalar() or 0
            metrics.paying_users = paying_users
            
            if paying_users > 0:
                metrics.arppu = metrics.total_revenue / paying_users
            
            metrics.updated_at = datetime.utcnow()
            
            await self.db.commit()
            await self.db.refresh(metrics)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating revenue metrics: {e}")
            await self.db.rollback()
            raise
    
    async def get_revenue_report(
        self,
        start_date: datetime,
        end_date: datetime,
        user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get comprehensive revenue report for date range."""
        try:
            import uuid as uuid_module
            user_uuid: Optional[uuid_module.UUID] = None
            if user_id:
                try:
                    user_uuid = uuid_module.UUID(user_id) if isinstance(user_id, str) else user_id
                except Exception:
                    # If invalid UUID string, ignore filter entirely
                    user_uuid = None

            # Build base filters
            filters = [
                RevenueMetrics.date >= start_date,
                RevenueMetrics.date < end_date,
                RevenueMetrics.period_type == "daily"
            ]

            # Apply user scoping:
            # - If user_uuid provided: filter to that user
            # - If explicit user_id None passed: platform-wide metrics (user_id IS NULL)
            if user_id is not None:  # caller intended a user scope (could be empty string)
                if user_uuid is not None:
                    filters.append(RevenueMetrics.user_id == user_uuid)
                else:
                    # If invalid or empty user id, fall back to IS NULL for safety
                    filters.append(RevenueMetrics.user_id.is_(None))
            else:
                # No user_id argument supplied -> platform scope
                filters.append(RevenueMetrics.user_id.is_(None))

            stmt = select(RevenueMetrics).where(and_(*filters)).order_by(RevenueMetrics.date)
            
            result = await self.db.execute(stmt)
            daily_metrics = result.scalars().all()
            
            # Aggregate totals
            total_revenue = sum(m.total_revenue for m in daily_metrics)
            total_subscriptions = sum(m.subscription_revenue for m in daily_metrics)
            total_donations = sum(m.donation_revenue for m in daily_metrics)
            total_transactions = sum(m.total_transactions for m in daily_metrics)
            
            # Build time series data
            time_series = [
                {
                    "date": m.date.isoformat(),
                    "revenue": m.total_revenue,
                    "subscriptions": m.subscription_revenue,
                    "donations": m.donation_revenue,
                    "transactions": m.total_transactions
                }
                for m in daily_metrics
            ]
            
            return {
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "total_revenue": total_revenue,
                "subscription_revenue": total_subscriptions,
                "donation_revenue": total_donations,
                "total_transactions": total_transactions,
                "avg_daily_revenue": total_revenue / len(daily_metrics) if daily_metrics else 0,
                "time_series": time_series
            }
            
        except Exception as e:
            logger.error(f"Error generating revenue report: {e}")
            raise
    
    # ========== Dashboard Data ==========
    
    async def get_platform_overview(self) -> Dict[str, Any]:
        """Get platform-wide overview metrics."""
        try:
            now = datetime.utcnow()
            
            # Total users
            stmt = select(func.count(User.id))
            result = await self.db.execute(stmt)
            total_users = result.scalar() or 0
            
            # Total videos
            stmt = select(func.count(Video.id))
            result = await self.db.execute(stmt)
            total_videos = result.scalar() or 0
            
            # Total views (sum of all video views)
            stmt = select(func.sum(Video.views_count))
            result = await self.db.execute(stmt)
            total_views = result.scalar() or 0
            
            # Active users (last 30 days)
            stmt = select(func.count(func.distinct(ViewSession.user_id))).where(
                ViewSession.created_at > now - timedelta(days=30)
            )
            result = await self.db.execute(stmt)
            active_users = result.scalar() or 0
            
            # Total revenue (last 30 days)
            stmt = select(func.sum(RevenueMetrics.total_revenue)).where(
                and_(
                    RevenueMetrics.date > now - timedelta(days=30),
                    RevenueMetrics.user_id.is_(None)
                )
            )
            result = await self.db.execute(stmt)
            total_revenue_30d = result.scalar() or 0
            
            # Active subscriptions
            stmt = select(func.count(Subscription.id)).where(
                Subscription.status == 'active'
            )
            result = await self.db.execute(stmt)
            active_subscriptions = result.scalar() or 0
            
            return {
                "total_users": total_users,
                "total_videos": total_videos,
                "total_views": int(total_views),
                "active_users_30d": active_users,
                "total_revenue_30d": float(total_revenue_30d),
                "active_subscriptions": active_subscriptions,
                "timestamp": now.isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting platform overview: {e}")
            raise
    
    async def get_top_videos(self, limit: int = 10, metric: str = "views") -> List[Dict[str, Any]]:
        """Get top performing videos."""
        try:
            # Order by specified metric
            order_column = VideoMetrics.total_views  # default
            
            if metric == "engagement":
                order_column = VideoMetrics.engagement_score
            elif metric == "revenue":
                order_column = VideoMetrics.total_revenue
            elif metric == "quality":
                order_column = VideoMetrics.quality_score
            
            stmt = select(VideoMetrics).order_by(desc(order_column)).limit(limit)
            result = await self.db.execute(stmt)
            metrics = result.scalars().all()
            
            # Get video details
            top_videos = []
            for m in metrics:
                stmt = select(Video).where(Video.id == m.video_id)
                result = await self.db.execute(stmt)
                video = result.scalar_one_or_none()
                
                if video:
                    top_videos.append({
                        "video_id": str(video.id),
                        "title": video.title,
                        "views": m.total_views,
                        "engagement_score": m.engagement_score,
                        "quality_score": m.quality_score,
                        "revenue": m.total_revenue
                    })
            
            return top_videos
            
        except Exception as e:
            logger.error(f"Error getting top videos: {e}")
            raise
    
    # ========== View Session Recording ==========
    
    async def record_view_session(
        self,
        video_id: str,
        user_id: Optional[str],
        session_id: str,
        duration: int,
        video_duration: int,
        **kwargs
    ) -> ViewSession:
        """Record a video view session."""
        try:
            watch_percentage = (duration / video_duration * 100) if video_duration > 0 else 0
            completed = watch_percentage >= 90  # Consider 90%+ as completed
            
            # Convert string IDs to UUID objects
            import uuid as uuid_module
            video_uuid = uuid_module.UUID(video_id) if isinstance(video_id, str) else video_id
            user_uuid = uuid_module.UUID(user_id) if user_id and isinstance(user_id, str) else user_id
            
            session = ViewSession(
                video_id=video_uuid,
                user_id=user_uuid,
                session_id=session_id,
                started_at=kwargs.get('started_at', datetime.utcnow()),
                ended_at=kwargs.get('ended_at'),
                duration=duration,
                video_duration=video_duration,
                watch_percentage=watch_percentage,
                completed=completed,
                quality_level=kwargs.get('quality_level'),
                buffering_count=kwargs.get('buffering_count', 0),
                buffering_duration=kwargs.get('buffering_duration', 0),
                device_type=kwargs.get('device_type'),
                os=kwargs.get('os'),
                browser=kwargs.get('browser'),
                ip_address=kwargs.get('ip_address'),
                country=kwargs.get('country'),
                referrer=kwargs.get('referrer'),
                traffic_source=kwargs.get('traffic_source'),
                liked=kwargs.get('liked', False),
                commented=kwargs.get('commented', False),
                shared=kwargs.get('shared', False)
            )
            
            self.db.add(session)
            await self.db.commit()
            await self.db.refresh(session)
            
            # Trigger async metric recalculation
            # This would typically be done via Celery task
            logger.info(f"View session recorded for video {video_id}")
            
            return session
            
        except Exception as e:
            logger.error(f"Error recording view session: {e}")
            await self.db.rollback()
            raise
