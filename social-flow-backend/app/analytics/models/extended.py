"""
Extended Analytics Models for Comprehensive Tracking.

This module defines extended analytics models for video metrics, user behavior,
revenue tracking, and aggregated statistics.
"""

import uuid
from datetime import datetime
from enum import Enum
from typing import Optional

from sqlalchemy import (
    Column, DateTime, Float, Integer, String, Text, 
    ForeignKey, Boolean, Index, BigInteger
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy import JSON
from sqlalchemy.orm import relationship

from app.core.database import Base


class VideoMetrics(Base):
    """Video performance metrics model."""
    
    __tablename__ = "video_metrics"
    
    # Primary key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Video reference
    video_id = Column(UUID(as_uuid=True), ForeignKey('videos.id', ondelete='CASCADE'), nullable=False, index=True)
    
    # View metrics
    total_views = Column(BigInteger, default=0, nullable=False)
    unique_views = Column(BigInteger, default=0, nullable=False)
    views_24h = Column(BigInteger, default=0, nullable=False)
    views_7d = Column(BigInteger, default=0, nullable=False)
    views_30d = Column(BigInteger, default=0, nullable=False)
    
    # Watch time metrics (in seconds)
    total_watch_time = Column(BigInteger, default=0, nullable=False)  # Total seconds watched
    avg_watch_time = Column(Float, default=0.0, nullable=False)  # Average seconds per view
    avg_watch_percentage = Column(Float, default=0.0, nullable=False)  # % of video watched (0-100)
    completion_rate = Column(Float, default=0.0, nullable=False)  # % who watched to end (0-100)
    
    # Engagement metrics
    total_likes = Column(Integer, default=0, nullable=False)
    total_dislikes = Column(Integer, default=0, nullable=False)
    total_comments = Column(Integer, default=0, nullable=False)
    total_shares = Column(Integer, default=0, nullable=False)
    total_saves = Column(Integer, default=0, nullable=False)
    
    # Engagement rates (0-100)
    like_rate = Column(Float, default=0.0, nullable=False)  # Likes per 100 views
    comment_rate = Column(Float, default=0.0, nullable=False)
    share_rate = Column(Float, default=0.0, nullable=False)
    
    # Audience retention
    retention_curve = Column(JSON, nullable=True)  # [{time: 0, retention: 100}, ...]
    peak_concurrent_viewers = Column(Integer, default=0, nullable=False)
    
    # Traffic sources
    traffic_sources = Column(JSON, nullable=True)  # {source_name: count, ...}
    referrer_breakdown = Column(JSON, nullable=True)  # {domain: count, ...}
    
    # Device and platform breakdown
    device_breakdown = Column(JSON, nullable=True)  # {mobile: X, desktop: Y, tablet: Z}
    os_breakdown = Column(JSON, nullable=True)  # {ios: X, android: Y, ...}
    browser_breakdown = Column(JSON, nullable=True)
    
    # Geographic breakdown
    country_breakdown = Column(JSON, nullable=True)  # {country_code: count, ...}
    top_countries = Column(JSON, nullable=True)  # [{country, views}, ...]
    
    # Revenue metrics (if monetized)
    total_revenue = Column(Float, default=0.0, nullable=False)
    ad_revenue = Column(Float, default=0.0, nullable=False)
    subscription_revenue = Column(Float, default=0.0, nullable=False)
    donation_revenue = Column(Float, default=0.0, nullable=False)
    
    # Performance score (0-100)
    engagement_score = Column(Float, default=0.0, nullable=False)
    virality_score = Column(Float, default=0.0, nullable=False)
    quality_score = Column(Float, default=0.0, nullable=False)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    last_calculated_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Indexes
    __table_args__ = (
        Index('idx_video_metrics_video_id', 'video_id'),
        Index('idx_video_metrics_updated_at', 'updated_at'),
    )
    
    def __repr__(self) -> str:
        return f"<VideoMetrics(video_id={self.video_id}, views={self.total_views})>"


class UserBehaviorMetrics(Base):
    """User behavior and activity metrics model."""
    
    __tablename__ = "user_behavior_metrics"
    
    # Primary key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # User reference
    user_id = Column(UUID(as_uuid=True), ForeignKey('users.id', ondelete='CASCADE'), nullable=False, index=True)
    
    # Activity metrics
    total_sessions = Column(Integer, default=0, nullable=False)
    total_session_duration = Column(BigInteger, default=0, nullable=False)  # Total seconds
    avg_session_duration = Column(Float, default=0.0, nullable=False)  # Average seconds
    sessions_24h = Column(Integer, default=0, nullable=False)
    sessions_7d = Column(Integer, default=0, nullable=False)
    sessions_30d = Column(Integer, default=0, nullable=False)
    
    # Content creation metrics
    total_videos_uploaded = Column(Integer, default=0, nullable=False)
    total_posts_created = Column(Integer, default=0, nullable=False)
    total_comments_posted = Column(Integer, default=0, nullable=False)
    videos_uploaded_30d = Column(Integer, default=0, nullable=False)
    
    # Content consumption metrics
    total_videos_watched = Column(Integer, default=0, nullable=False)
    total_watch_time = Column(BigInteger, default=0, nullable=False)  # Seconds
    videos_watched_30d = Column(Integer, default=0, nullable=False)
    avg_daily_watch_time = Column(Float, default=0.0, nullable=False)  # Seconds
    
    # Engagement metrics
    total_likes_given = Column(Integer, default=0, nullable=False)
    total_comments_given = Column(Integer, default=0, nullable=False)
    total_shares = Column(Integer, default=0, nullable=False)
    total_follows = Column(Integer, default=0, nullable=False)
    total_followers = Column(Integer, default=0, nullable=False)
    
    # Social metrics
    following_count = Column(Integer, default=0, nullable=False)
    followers_count = Column(Integer, default=0, nullable=False)
    followers_growth_30d = Column(Integer, default=0, nullable=False)
    engagement_rate = Column(Float, default=0.0, nullable=False)  # % engagement on content
    
    # Creator metrics (if applicable)
    creator_status = Column(Boolean, default=False, nullable=False)
    total_video_views = Column(BigInteger, default=0, nullable=False)  # Views on user's videos
    total_video_likes = Column(BigInteger, default=0, nullable=False)  # Likes on user's videos
    avg_video_performance = Column(Float, default=0.0, nullable=False)  # Avg views per video
    
    # Revenue metrics (if applicable)
    total_earnings = Column(Float, default=0.0, nullable=False)
    earnings_30d = Column(Float, default=0.0, nullable=False)
    subscription_revenue = Column(Float, default=0.0, nullable=False)
    ad_revenue = Column(Float, default=0.0, nullable=False)
    donation_revenue = Column(Float, default=0.0, nullable=False)
    
    # Spending metrics
    total_spent = Column(Float, default=0.0, nullable=False)
    spent_30d = Column(Float, default=0.0, nullable=False)
    subscriptions_purchased = Column(Integer, default=0, nullable=False)
    donations_sent = Column(Float, default=0.0, nullable=False)
    
    # Device preferences
    primary_device = Column(String(50), nullable=True)  # mobile, desktop, tablet
    device_usage = Column(JSON, nullable=True)  # {device: percentage, ...}
    
    # Time preferences
    most_active_hours = Column(JSON, nullable=True)  # [hour_of_day, ...]
    most_active_days = Column(JSON, nullable=True)  # [day_of_week, ...]
    
    # Content preferences
    favorite_categories = Column(JSON, nullable=True)  # [{category, count}, ...]
    watch_history_summary = Column(JSON, nullable=True)
    
    # User scores (0-100)
    activity_score = Column(Float, default=0.0, nullable=False)
    creator_score = Column(Float, default=0.0, nullable=False)
    engagement_score = Column(Float, default=0.0, nullable=False)
    loyalty_score = Column(Float, default=0.0, nullable=False)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    last_calculated_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Indexes
    __table_args__ = (
        Index('idx_user_behavior_user_id', 'user_id'),
        Index('idx_user_behavior_updated_at', 'updated_at'),
        Index('idx_user_behavior_creator_status', 'creator_status'),
    )
    
    def __repr__(self) -> str:
        return f"<UserBehaviorMetrics(user_id={self.user_id}, sessions={self.total_sessions})>"

    def __init__(self, *args, **kwargs):
        # Map test-friendly aliases to actual column names
        alias_map = {
            "videos_watched": "total_videos_watched",
            "videos_uploaded": "total_videos_uploaded",
        }
        for a, r in alias_map.items():
            if a in kwargs and r not in kwargs:
                kwargs[r] = kwargs.pop(a)
        super().__init__(*args, **kwargs)


class RevenueMetrics(Base):
    """Revenue and monetization metrics model."""
    
    __tablename__ = "revenue_metrics"
    
    # Primary key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Time period
    date = Column(DateTime, nullable=False, index=True)  # Date of the metrics
    period_type = Column(String(20), nullable=False)  # daily, weekly, monthly
    
    # User reference (optional - for per-user metrics)
    user_id = Column(UUID(as_uuid=True), ForeignKey('users.id', ondelete='CASCADE'), nullable=True, index=True)
    
    # Subscription revenue
    subscription_revenue = Column(Float, default=0.0, nullable=False)
    new_subscriptions = Column(Integer, default=0, nullable=False)
    renewed_subscriptions = Column(Integer, default=0, nullable=False)
    canceled_subscriptions = Column(Integer, default=0, nullable=False)
    active_subscriptions = Column(Integer, default=0, nullable=False)
    subscription_mrr = Column(Float, default=0.0, nullable=False)  # Monthly recurring revenue
    
    # Donation revenue
    donation_revenue = Column(Float, default=0.0, nullable=False)
    total_donations = Column(Integer, default=0, nullable=False)
    avg_donation_amount = Column(Float, default=0.0, nullable=False)
    unique_donors = Column(Integer, default=0, nullable=False)
    
    # Ad revenue
    ad_revenue = Column(Float, default=0.0, nullable=False)
    ad_impressions = Column(BigInteger, default=0, nullable=False)
    ad_clicks = Column(BigInteger, default=0, nullable=False)
    ad_ctr = Column(Float, default=0.0, nullable=False)  # Click-through rate
    ad_cpm = Column(Float, default=0.0, nullable=False)  # Cost per mille
    
    # Total revenue
    total_revenue = Column(Float, default=0.0, nullable=False)
    gross_revenue = Column(Float, default=0.0, nullable=False)
    net_revenue = Column(Float, default=0.0, nullable=False)
    platform_fee = Column(Float, default=0.0, nullable=False)
    
    # Payouts
    total_payouts = Column(Float, default=0.0, nullable=False)
    pending_payouts = Column(Float, default=0.0, nullable=False)
    completed_payouts = Column(Float, default=0.0, nullable=False)
    
    # Transaction metrics
    total_transactions = Column(Integer, default=0, nullable=False)
    successful_transactions = Column(Integer, default=0, nullable=False)
    failed_transactions = Column(Integer, default=0, nullable=False)
    refunded_transactions = Column(Integer, default=0, nullable=False)
    
    # User metrics
    paying_users = Column(Integer, default=0, nullable=False)
    new_paying_users = Column(Integer, default=0, nullable=False)
    churned_users = Column(Integer, default=0, nullable=False)
    
    # ARPU metrics (Average Revenue Per User)
    arpu = Column(Float, default=0.0, nullable=False)
    arppu = Column(Float, default=0.0, nullable=False)  # Average revenue per paying user
    
    # Breakdown by revenue type
    revenue_breakdown = Column(JSON, nullable=True)  # {type: amount, ...}
    
    # Geographic breakdown
    revenue_by_country = Column(JSON, nullable=True)  # {country: amount, ...}
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    
    # Indexes
    __table_args__ = (
        Index('idx_revenue_metrics_date', 'date'),
        Index('idx_revenue_metrics_user_date', 'user_id', 'date'),
        Index('idx_revenue_metrics_period_type', 'period_type'),
    )
    
    def __repr__(self) -> str:
        return f"<RevenueMetrics(date={self.date}, revenue=${self.total_revenue})>"


class AggregatedMetrics(Base):
    """Pre-aggregated metrics for fast dashboard loading."""
    
    __tablename__ = "aggregated_metrics"
    
    # Primary key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Aggregation info
    metric_type = Column(String(50), nullable=False, index=True)  # platform, user, video, revenue
    aggregation_period = Column(String(20), nullable=False)  # hourly, daily, weekly, monthly
    start_date = Column(DateTime, nullable=False, index=True)
    end_date = Column(DateTime, nullable=False)
    
    # Entity reference (optional)
    entity_id = Column(UUID(as_uuid=True), nullable=True, index=True)
    entity_type = Column(String(50), nullable=True)  # user, video, etc.
    
    # Aggregated data
    metrics_data = Column(JSON, nullable=False)  # All metrics in JSON format
    
    # Summary fields for quick access
    total_count = Column(BigInteger, default=0, nullable=False)
    total_value = Column(Float, default=0.0, nullable=False)
    avg_value = Column(Float, default=0.0, nullable=False)
    min_value = Column(Float, nullable=True)
    max_value = Column(Float, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    
    # Indexes
    __table_args__ = (
        Index('idx_aggregated_metrics_type_period', 'metric_type', 'aggregation_period'),
        Index('idx_aggregated_metrics_entity', 'entity_id', 'entity_type'),
        Index('idx_aggregated_metrics_start_date', 'start_date'),
    )
    
    def __repr__(self) -> str:
        return f"<AggregatedMetrics(type={self.metric_type}, period={self.aggregation_period})>"


class ViewSession(Base):
    """Individual video viewing session for granular analytics."""
    
    __tablename__ = "view_sessions"
    
    # Primary key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Video and user reference
    video_id = Column(UUID(as_uuid=True), ForeignKey('videos.id', ondelete='CASCADE'), nullable=False, index=True)
    user_id = Column(UUID(as_uuid=True), ForeignKey('users.id', ondelete='SET NULL'), nullable=True, index=True)
    session_id = Column(String(255), nullable=False, index=True)  # Browser session
    
    # View details
    started_at = Column(DateTime, nullable=False)
    ended_at = Column(DateTime, nullable=True)
    duration = Column(Integer, nullable=False, default=0)  # Seconds watched
    video_duration = Column(Integer, nullable=False)  # Total video length
    watch_percentage = Column(Float, nullable=False, default=0.0)  # % watched
    completed = Column(Boolean, default=False, nullable=False)  # Watched to end
    
    # Playback quality
    quality_level = Column(String(20), nullable=True)  # 720p, 1080p, etc.
    buffering_count = Column(Integer, default=0, nullable=False)
    buffering_duration = Column(Integer, default=0, nullable=False)  # Total buffering time
    
    # Device and context
    device_type = Column(String(50), nullable=True)
    os = Column(String(50), nullable=True)
    browser = Column(String(50), nullable=True)
    ip_address = Column(String(45), nullable=True)
    country = Column(String(100), nullable=True)
    
    # Traffic source
    referrer = Column(String(500), nullable=True)
    traffic_source = Column(String(100), nullable=True)  # direct, search, social, etc.
    
    # Engagement during session
    liked = Column(Boolean, default=False, nullable=False)
    commented = Column(Boolean, default=False, nullable=False)
    shared = Column(Boolean, default=False, nullable=False)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Indexes
    __table_args__ = (
        Index('idx_view_sessions_video_id', 'video_id'),
        Index('idx_view_sessions_user_id', 'user_id'),
        Index('idx_view_sessions_created_at', 'created_at'),
        Index('idx_view_sessions_session_id', 'session_id'),
    )
    
    def __repr__(self) -> str:
        return f"<ViewSession(video_id={self.video_id}, user_id={self.user_id}, duration={self.duration}s)>"
