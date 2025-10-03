"""
Value Objects for the Domain Layer

Value objects are immutable objects that are defined by their attributes
rather than a unique identity. They encapsulate domain concepts and ensure
validation rules.
"""

import re
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class VideoStatus(str, Enum):
    """Video processing status."""
    UPLOADING = "uploading"
    PROCESSING = "processing"
    PROCESSED = "processed"
    FAILED = "failed"
    DELETED = "deleted"


class VideoVisibility(str, Enum):
    """Video visibility level."""
    PUBLIC = "public"
    UNLISTED = "unlisted"
    PRIVATE = "private"


class PostVisibility(str, Enum):
    """Post visibility level."""
    PUBLIC = "public"
    FRIENDS = "friends"
    PRIVATE = "private"


class UserRole(str, Enum):
    """User role in the system."""
    USER = "user"
    CREATOR = "creator"
    MODERATOR = "moderator"
    ADMIN = "admin"
    SUPER_ADMIN = "super_admin"


@dataclass(frozen=True)
class Email:
    """Email value object with validation."""
    
    value: str
    
    def __post_init__(self):
        """Validate email format."""
        if not self._is_valid_email(self.value):
            raise ValueError(f"Invalid email format: {self.value}")
    
    @staticmethod
    def _is_valid_email(email: str) -> bool:
        """Validate email format using regex."""
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return bool(re.match(pattern, email))
    
    def __str__(self) -> str:
        return self.value
    
    @property
    def domain(self) -> str:
        """Extract domain from email."""
        return self.value.split('@')[1]


@dataclass(frozen=True)
class Username:
    """Username value object with validation."""
    
    value: str
    
    def __post_init__(self):
        """Validate username format."""
        if not self._is_valid_username(self.value):
            raise ValueError(
                f"Invalid username: {self.value}. "
                "Username must be 3-50 characters, alphanumeric with underscores/hyphens only."
            )
    
    @staticmethod
    def _is_valid_username(username: str) -> bool:
        """Validate username format."""
        if not username or len(username) < 3 or len(username) > 50:
            return False
        # Allow alphanumeric, underscores, and hyphens
        pattern = r'^[a-zA-Z0-9_-]+$'
        return bool(re.match(pattern, username))
    
    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True)
class Password:
    """Password value object with validation."""
    
    value: str
    
    def __post_init__(self):
        """Validate password strength."""
        if not self._is_strong_password(self.value):
            raise ValueError(
                "Weak password. Password must be at least 8 characters, "
                "contain uppercase, lowercase, digit, and special character."
            )
    
    @staticmethod
    def _is_strong_password(password: str) -> bool:
        """Validate password strength."""
        if len(password) < 8:
            return False
        
        has_upper = any(c.isupper() for c in password)
        has_lower = any(c.islower() for c in password)
        has_digit = any(c.isdigit() for c in password)
        has_special = any(c in "!@#$%^&*()_+-=[]{}|;:,.<>?" for c in password)
        
        return has_upper and has_lower and has_digit and has_special
    
    def __str__(self) -> str:
        return "********"  # Never expose password


@dataclass(frozen=True)
class VideoMetadata:
    """Video metadata value object."""
    
    duration: float  # in seconds
    resolution: str  # e.g., "1920x1080"
    bitrate: int  # in kbps
    codec: str  # e.g., "h264"
    file_size: int  # in bytes
    
    def __post_init__(self):
        """Validate metadata."""
        if self.duration <= 0:
            raise ValueError("Duration must be positive")
        if self.file_size <= 0:
            raise ValueError("File size must be positive")
        if self.bitrate <= 0:
            raise ValueError("Bitrate must be positive")
    
    @property
    def width(self) -> int:
        """Extract width from resolution."""
        return int(self.resolution.split('x')[0])
    
    @property
    def height(self) -> int:
        """Extract height from resolution."""
        return int(self.resolution.split('x')[1])
    
    @property
    def aspect_ratio(self) -> str:
        """Calculate aspect ratio."""
        from math import gcd
        w, h = self.width, self.height
        divisor = gcd(w, h)
        return f"{w//divisor}:{h//divisor}"
    
    @property
    def size_mb(self) -> float:
        """Get file size in MB."""
        return self.file_size / (1024 * 1024)


@dataclass(frozen=True)
class EngagementMetrics:
    """Engagement metrics value object."""
    
    views: int = 0
    likes: int = 0
    dislikes: int = 0
    comments: int = 0
    shares: int = 0
    
    def __post_init__(self):
        """Validate metrics."""
        if any(v < 0 for v in [self.views, self.likes, self.dislikes, self.comments, self.shares]):
            raise ValueError("Metrics cannot be negative")
    
    @property
    def total_engagement(self) -> int:
        """Calculate total engagement."""
        return self.likes + self.comments + self.shares
    
    @property
    def engagement_rate(self) -> float:
        """Calculate engagement rate as percentage."""
        if self.views == 0:
            return 0.0
        return (self.total_engagement / self.views) * 100
    
    @property
    def like_ratio(self) -> float:
        """Calculate like to dislike ratio."""
        total_reactions = self.likes + self.dislikes
        if total_reactions == 0:
            return 0.0
        return (self.likes / total_reactions) * 100
    
    def with_view(self) -> 'EngagementMetrics':
        """Return new instance with incremented views."""
        return EngagementMetrics(
            views=self.views + 1,
            likes=self.likes,
            dislikes=self.dislikes,
            comments=self.comments,
            shares=self.shares
        )
    
    def with_like(self) -> 'EngagementMetrics':
        """Return new instance with incremented likes."""
        return EngagementMetrics(
            views=self.views,
            likes=self.likes + 1,
            dislikes=self.dislikes,
            comments=self.comments,
            shares=self.shares
        )
    
    def with_comment(self) -> 'EngagementMetrics':
        """Return new instance with incremented comments."""
        return EngagementMetrics(
            views=self.views,
            likes=self.likes,
            dislikes=self.dislikes,
            comments=self.comments + 1,
            shares=self.shares
        )


@dataclass(frozen=True)
class StorageLocation:
    """Storage location value object."""
    
    bucket: str
    key: str
    region: Optional[str] = None
    
    def __post_init__(self):
        """Validate storage location."""
        if not self.bucket:
            raise ValueError("Bucket cannot be empty")
        if not self.key:
            raise ValueError("Key cannot be empty")
    
    @property
    def url(self) -> str:
        """Generate S3 URL."""
        if self.region:
            return f"https://{self.bucket}.s3.{self.region}.amazonaws.com/{self.key}"
        return f"https://{self.bucket}.s3.amazonaws.com/{self.key}"
    
    def __str__(self) -> str:
        return f"s3://{self.bucket}/{self.key}"
