"""
Application configuration management.

This module handles all configuration settings using Pydantic Settings
for type validation and environment variable management.
"""

import secrets
import os
from typing import Any, List, Optional, Union

from pydantic import AnyHttpUrl, EmailStr, HttpUrl, PostgresDsn, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings with environment variable support."""
    
    # Project Information
    PROJECT_NAME: str = "Social Flow Backend"
    PROJECT_DESCRIPTION: str = "Social media backend with video streaming and micro-posts"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"
    
    # Security
    SECRET_KEY: str = os.getenv("SECRET_KEY", secrets.token_urlsafe(32))
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7
    ALGORITHM: str = "HS256"
    
    # CORS
    BACKEND_CORS_ORIGINS: List[AnyHttpUrl] = []
    
    @field_validator("BACKEND_CORS_ORIGINS", mode="before")
    @classmethod
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
    # Allow any database URL (PostgreSQL or SQLite for development)
    DATABASE_URL: Optional[str] = None
    
    @field_validator("DATABASE_URL", mode="before")
    @classmethod
    def assemble_db_connection(cls, v: Optional[str], info) -> Any:
        if isinstance(v, str):
            # Return as-is if already a string (allows SQLite URLs for development)
            return v
        # In Pydantic V2, use info.data to access other fields
        postgres_user = info.data.get("POSTGRES_USER")
        postgres_password = info.data.get("POSTGRES_PASSWORD")
        postgres_server = info.data.get("POSTGRES_SERVER")
        postgres_port = info.data.get("POSTGRES_PORT", "5432")
        postgres_db = info.data.get("POSTGRES_DB")
        
        # Build PostgreSQL URL by default
        return f"postgresql+asyncpg://{postgres_user}:{postgres_password}@{postgres_server}:{postgres_port}/{postgres_db}"
    
    # Redis
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_PASSWORD: Optional[str] = None
    REDIS_DB: int = 0
    REDIS_URL: Optional[str] = None
    
    @field_validator("REDIS_URL", mode="before")
    @classmethod
    def assemble_redis_connection(cls, v: Optional[str], info) -> Any:
        if isinstance(v, str):
            return v
        # In Pydantic V2, use info.data to access other fields
        redis_password = info.data.get("REDIS_PASSWORD")
        redis_host = info.data.get("REDIS_HOST")
        redis_port = info.data.get("REDIS_PORT")
        redis_db = info.data.get("REDIS_DB")
        
        auth = f":{redis_password}@" if redis_password else ""
        return f"redis://{auth}{redis_host}:{redis_port}/{redis_db}"
    
    # AWS Configuration
    AWS_ACCESS_KEY_ID: Optional[str] = None
    AWS_SECRET_ACCESS_KEY: Optional[str] = None
    AWS_REGION: str = "us-east-1"
    AWS_S3_BUCKET: str = "social-flow-videos"
    S3_BUCKET_NAME: str = "social-flow-videos"  # Alias for compatibility
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
    
    # Email / SMTP Configuration
    SMTP_HOST: Optional[str] = None
    SMTP_PORT: int = 587
    SMTP_USER: Optional[str] = None
    SMTP_PASSWORD: Optional[str] = None
    SMTP_TLS: bool = True
    SMTP_SSL: bool = False
    
    # Firebase Cloud Messaging (FCM)
    FCM_CREDENTIALS_FILE: Optional[str] = None
    FCM_PROJECT_ID: Optional[str] = None
    
    # Twilio SMS
    TWILIO_ACCOUNT_SID: Optional[str] = None
    TWILIO_AUTH_TOKEN: Optional[str] = None
    TWILIO_PHONE_NUMBER: Optional[str] = None
    
    # Payments
    STRIPE_SECRET_KEY: Optional[str] = None
    STRIPE_PUBLISHABLE_KEY: Optional[str] = None
    STRIPE_WEBHOOK_SECRET: Optional[str] = None
    STRIPE_BASIC_PRICE_ID: Optional[str] = None
    STRIPE_PREMIUM_PRICE_ID: Optional[str] = None
    STRIPE_PRO_PRICE_ID: Optional[str] = None
    FRONTEND_URL: str = "http://localhost:3000"
    
    # OAuth Social Login
    GOOGLE_CLIENT_ID: Optional[str] = None
    GOOGLE_CLIENT_SECRET: Optional[str] = None
    FACEBOOK_CLIENT_ID: Optional[str] = None
    FACEBOOK_CLIENT_SECRET: Optional[str] = None
    APPLE_CLIENT_ID: Optional[str] = None
    APPLE_CLIENT_SECRET: Optional[str] = None
    OAUTH_REDIRECT_URI: str = "http://localhost:3000/auth/callback"
    
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
    ENVIRONMENT: str = "development"  # development, staging, production
    
    # Rate Limiting
    RATE_LIMIT_ENABLED: bool = True
    RATE_LIMIT_REQUESTS: int = 100
    RATE_LIMIT_WINDOW: int = 60  # 1 minute
    
    # Allowed Hosts
    ALLOWED_HOSTS: List[str] = ["*"]
    
    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=True,
        extra="allow"
    )


# Create settings instance
settings = Settings()
