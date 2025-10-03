"""
CRUD operations for database models.

This package provides CRUD (Create, Read, Update, Delete) operations
for all models in the application using SQLAlchemy 2.0 async patterns.
"""

from app.infrastructure.crud.base import CRUDBase

# User CRUD
from app.infrastructure.crud.crud_user import user

# Video CRUD
from app.infrastructure.crud.crud_video import video

# Social CRUD
from app.infrastructure.crud.crud_social import (
    post,
    comment,
    like,
    follow,
    save,
)

# Payment CRUD
from app.infrastructure.crud.crud_payment import (
    payment,
    subscription,
    payout,
    transaction,
)

# Ad CRUD
from app.infrastructure.crud.crud_ad import (
    ad_campaign,
    ad,
    ad_impression,
    ad_click,
)

# LiveStream CRUD
from app.infrastructure.crud.crud_livestream import (
    livestream,
    stream_chat,
    stream_donation,
    stream_viewer,
)

# Notification CRUD
from app.infrastructure.crud.crud_notification import (
    notification,
    notification_settings,
    push_token,
)

__all__ = [
    # Base
    "CRUDBase",
    # User
    "user",
    # Video
    "video",
    # Social
    "post",
    "comment",
    "like",
    "follow",
    "save",
    # Payment
    "payment",
    "subscription",
    "payout",
    "transaction",
    # Ad
    "ad_campaign",
    "ad",
    "ad_impression",
    "ad_click",
    # LiveStream
    "livestream",
    "stream_chat",
    "stream_donation",
    "stream_viewer",
    # Notification
    "notification",
    "notification_settings",
    "push_token",
]
