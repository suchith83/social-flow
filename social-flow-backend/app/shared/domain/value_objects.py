"""
Shared Value Objects - Shared Kernel

Value objects used across multiple bounded contexts.
These represent concepts that are part of the ubiquitous language
and need to be consistent across the entire system.
"""

from enum import Enum


class UserRole(str, Enum):
    """
    User role in the system.
    
    Part of shared kernel as it's used by multiple bounded contexts:
    - Auth: For authorization and user management
    - Videos: For determining upload and moderation permissions
    - Posts: For determining posting and moderation permissions
    - Ads: For advertiser and admin capabilities
    - Livestream: For streaming permissions
    """
    USER = "user"
    CREATOR = "creator"
    MODERATOR = "moderator"
    ADMIN = "admin"
    SUPER_ADMIN = "super_admin"


class VideoStatus(str, Enum):
    """
    Video processing status.
    
    Part of shared kernel as it's referenced by multiple contexts.
    """
    UPLOADING = "uploading"
    PROCESSING = "processing"
    PROCESSED = "processed"
    FAILED = "failed"
    DELETED = "deleted"


class VideoVisibility(str, Enum):
    """
    Video visibility level.
    
    Part of shared kernel for consistent visibility rules.
    """
    PUBLIC = "public"
    UNLISTED = "unlisted"
    PRIVATE = "private"


class PostVisibility(str, Enum):
    """
    Post visibility level.
    
    Part of shared kernel for consistent visibility rules.
    """
    PUBLIC = "public"
    FRIENDS = "friends"
    PRIVATE = "private"
