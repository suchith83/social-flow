"""
Application configuration management.

This module handles all configuration settings using Pydantic BaseSettings
for type validation and environment variable management.
"""

import secrets
from typing import Any, Dict, List, Optional, Union

from pydantic import AnyHttpUrl, BaseSettings, EmailStr, HttpUrl, PostgresDsn, validator


class Settings(BaseSettings):
    """Application settings with environment variable support."""
    
    # Project Information
    PROJECT_NAME: str = "Social Flow Backend"
    PROJECT_DESCRIPTION: str = "Social media backend with video streaming and micro-posts"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"
    
    # Security
    SECRET_KEY: str = secrets.token_urlsafe(32)
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7
    ALGORITHM: str = "HS256"
    
    # CORS
    BACKEND_CORS_ORIGINS: List[AnyHttpUrl] = []
    
    @validator("BACKEND_CORS_ORIGINS", pre=True)
    def assemble_cors_origins(cls, v: Union[str, List[str]]) -> Union[List[str], str]:
        if isinstance(v, str) and not v.startswith("["):
            return [i.strip() for i in v.split(",")]
        elif isinstance(v, (list, str)):
            return v
        raise ValueError(v)
    
    # Database
    POSTGRES_SERVER: str = "localhost"
    POSTGRES_USER: str = "postgres"
    POSTGRES_PASSWORD: str = "password"
    POSTGRES_DB: str = "social_flow"
    POSTGRES_PORT: str = "5432"
    DATABASE_URL: Optional[PostgresDsn] = None
    
    @validator("DATABASE_URL", pre=True)
    def assemble_db_connection(cls, v: Optional[str], values: Dict[str, Any]) -> Any:
        if isinstance(v, str):
            return v
        return PostgresDsn.build(
            scheme="postgresql+asyncpg",
            user=values.get("POSTGRES_USER"),
            password=values.get("POSTGRES_PASSWORD"),
            host=values.get("POSTGRES_SERVER"),
            port=values.get("POSTGRES_PORT"),
            path=f"/{values.get('POSTGRES_DB') or ''}",
        )
    
    # Redis
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_PASSWORD: Optional[str] = None
    REDIS_DB: int = 0
    REDIS_URL: Optional[str] = None
    
    @validator("REDIS_URL", pre=True)
    def assemble_redis_connection(cls, v: Optional[str], values: Dict[str, Any]) -> Any:
        if isinstance(v, str):
            return v
        password = values.get("REDIS_PASSWORD")
        auth = f":{password}@" if password else ""
        return f"redis://{auth}{values.get('REDIS_HOST')}:{values.get('REDIS_PORT')}/{values.get('REDIS_DB')}"
    
    # AWS Configuration
    AWS_ACCESS_KEY_ID: Optional[str] = None
    AWS_SECRET_ACCESS_KEY: Optional[str] = None
    AWS_REGION: str = "us-east-1"
    AWS_S3_BUCKET: str = "social-flow-videos"
    AWS_S3_REGION: str = "us-east-1"
    AWS_CLOUDFRONT_DOMAIN: Optional[str] = None
    
    # Media Processing
    MEDIA_UPLOAD_MAX_SIZE: int = 100 * 1024 * 1024  # 100MB
    MEDIA_ALLOWED_EXTENSIONS: List[str] = ["mp4", "avi", "mov", "wmv", "flv", "webm"]
    MEDIA_MAX_DURATION: int = 3600  # 1 hour in seconds
    
    # Video Processing
    VIDEO_ENCODING_QUEUE: str = "video_encoding"
    VIDEO_THUMBNAIL_QUEUE: str = "video_thumbnails"
    VIDEO_PROCESSING_TIMEOUT: int = 3600  # 1 hour
    
    # Live Streaming
    RTMP_INGEST_URL: str = "rtmp://localhost:1935/live"
    RTMP_PLAYBACK_URL: str = "http://localhost:8080/hls"
    STREAM_KEY_LENGTH: int = 32
    
    # Social Features
    POST_MAX_LENGTH: int = 280
    COMMENT_MAX_LENGTH: int = 500
    HASHTAG_MAX_LENGTH: int = 50
    MENTION_MAX_LENGTH: int = 20
    
    # Feed Algorithm
    FEED_PAGE_SIZE: int = 20
    FEED_CACHE_TTL: int = 300  # 5 minutes
    FEED_ALGORITHM: str = "hybrid"  # hybrid, chronological, personalized
    
    # Notifications
    NOTIFICATION_QUEUE: str = "notifications"
    EMAIL_FROM: Optional[EmailStr] = None
    PUSH_NOTIFICATION_ENABLED: bool = True
    
    # Payments
    STRIPE_SECRET_KEY: Optional[str] = None
    STRIPE_PUBLISHABLE_KEY: Optional[str] = None
    STRIPE_WEBHOOK_SECRET: Optional[str] = None
    
    # Analytics
    ANALYTICS_QUEUE: str = "analytics"
    ANALYTICS_BATCH_SIZE: int = 1000
    ANALYTICS_FLUSH_INTERVAL: int = 60  # 1 minute
    
    # ML/AI
    ML_MODEL_CACHE_TTL: int = 3600  # 1 hour
    RECOMMENDATION_MODEL_URL: Optional[str] = None
    CONTENT_MODERATION_MODEL_URL: Optional[str] = None
    
    # Monitoring
    ENABLE_METRICS: bool = True
    LOG_LEVEL: str = "INFO"
    SENTRY_DSN: Optional[HttpUrl] = None
    
    # Development
    DEBUG: bool = False
    TESTING: bool = False
    
    # Rate Limiting
    RATE_LIMIT_ENABLED: bool = True
    RATE_LIMIT_REQUESTS: int = 100
    RATE_LIMIT_WINDOW: int = 60  # 1 minute
    
    # Allowed Hosts
    ALLOWED_HOSTS: List[str] = ["*"]
    
    class Config:
        env_file = ".env"
        case_sensitive = True


# Create settings instance
settings = Settings()
