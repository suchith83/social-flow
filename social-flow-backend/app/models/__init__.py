"""
Database models package.

This package contains all SQLAlchemy models for the Social Flow backend.
Comprehensive production-ready models with proper relationships, indexes,
and partitioning support.
"""

# ==================== Base Models ====================
from app.models.base import (
    Base,
    UUIDMixin,
    TimestampMixin,
    SoftDeleteMixin,
    AuditMixin,
    MetadataMixin,
    CommonBase,
    AuditedBase,
    FlexibleBase,
)

# ==================== User Models ====================
from app.models.user import (
    User,
    UserRole,
    UserStatus,
    EmailVerificationToken,
    PasswordResetToken,
)

# ==================== Video Models ====================
from app.models.video import (
    Video,
    VideoView,
    VideoStatus,
    VideoVisibility,
    ModerationStatus,
)

# ==================== Social Models ====================
from app.models.social import (
    Post,
    Comment,
    Like,
    Follow,
    Save,
    PostVisibility,
    MediaType,
)

# ==================== Payment Models ====================
from app.models.payment import (
    Payment,
    Subscription,
    Payout,
    Transaction,
    PaymentType,
    PaymentStatus,
    PaymentProvider,
    SubscriptionTier,
    SubscriptionStatus,
    PayoutStatus,
)

# ==================== Ad Models ====================
from app.models.ad import (
    AdCampaign,
    Ad,
    AdImpression,
    AdClick,
    AdType,
    AdPlacement,
    CampaignStatus,
    CampaignObjective,
    BidStrategy,
)

# ==================== LiveStream Models ====================
from app.models.livestream import (
    LiveStream,
    StreamChat,
    StreamDonation,
    StreamViewer,
    StreamStatus,
    StreamVisibility,
    StreamQuality,
    DonationStatus,
    ChatMessageType,
)

# ==================== Notification Models ====================
from app.models.notification import (
    Notification,
    NotificationSettings,
    PushToken,
    NotificationType,
    NotificationChannel,
    NotificationStatus,
    PushPlatform,
)

# ==================== Analytics Models ====================
from app.analytics.models.extended import (
    VideoMetrics,
    UserBehaviorMetrics,
    RevenueMetrics,
    AggregatedMetrics,
    ViewSession,
)

from app.analytics.models.analytics import (
    Analytics,
    AnalyticsType,
    AnalyticsCategory,
)


__all__ = [
    # Base
    "Base",
    "UUIDMixin",
    "TimestampMixin",
    "SoftDeleteMixin",
    "AuditMixin",
    "MetadataMixin",
    "CommonBase",
    "AuditedBase",
    "FlexibleBase",
    
    # User
    "User",
    "UserRole",
    "UserStatus",
    "EmailVerificationToken",
    "PasswordResetToken",
    
    # Video
    "Video",
    "VideoView",
    "VideoStatus",
    "VideoVisibility",
    "ModerationStatus",
    
    # Social
    "Post",
    "Comment",
    "Like",
    "Follow",
    "Save",
    "PostVisibility",
    "MediaType",
    
    # Payment
    "Payment",
    "Subscription",
    "Payout",
    "Transaction",
    "PaymentType",
    "PaymentStatus",
    "PaymentProvider",
    "SubscriptionTier",
    "SubscriptionStatus",
    "PayoutStatus",
    
    # Ad
    "AdCampaign",
    "Ad",
    "AdImpression",
    "AdClick",
    "AdType",
    "AdPlacement",
    "CampaignStatus",
    "CampaignObjective",
    "BidStrategy",
    
    # LiveStream
    "LiveStream",
    "StreamChat",
    "StreamDonation",
    "StreamViewer",
    "StreamStatus",
    "StreamVisibility",
    "StreamQuality",
    "DonationStatus",
    "ChatMessageType",
    
    # Notification
    "Notification",
    "NotificationSettings",
    "PushToken",
    "NotificationType",
    "NotificationChannel",
    "NotificationStatus",
    "PushPlatform",
    
    # Analytics
    "VideoMetrics",
    "UserBehaviorMetrics",
    "RevenueMetrics",
    "AggregatedMetrics",
    "ViewSession",
    "Analytics",
    "AnalyticsType",
    "AnalyticsCategory",
]


# Model registry for Alembic migrations
MODEL_REGISTRY = [
    # User models
    User,
    EmailVerificationToken,
    PasswordResetToken,
    
    # Video models
    Video,
    VideoView,
    
    # Social models
    Post,
    Comment,
    Like,
    Follow,
    Save,
    
    # Payment models
    Payment,
    Subscription,
    Payout,
    Transaction,
    
    # Ad models
    AdCampaign,
    Ad,
    AdImpression,
    AdClick,
    
    # LiveStream models
    LiveStream,
    StreamChat,
    StreamDonation,
    StreamViewer,
    
    # Notification models
    Notification,
    NotificationSettings,
    PushToken,
    
    # Analytics models
    VideoMetrics,
    UserBehaviorMetrics,
    RevenueMetrics,
    AggregatedMetrics,
    ViewSession,
    Analytics,
]

