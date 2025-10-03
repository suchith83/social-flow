"""
Enhanced Configuration Management with AWS Integration

This module provides comprehensive configuration management with:
- Environment-based settings
- AWS service configurations
- Database sharding support
- Advanced caching strategies
- Security hardening
- Feature flags
"""

import secrets
from typing import Any, Dict, List, Optional, Union
from enum import Enum

from pydantic import (
    AnyHttpUrl,
    EmailStr,
    Field,
    PostgresDsn,
    RedisDsn,
    field_validator,
    model_validator,
)
from pydantic_settings import BaseSettings, SettingsConfigDict


class Environment(str, Enum):
    """Application environment types."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TESTING = "testing"


class LogLevel(str, Enum):
    """Logging levels."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class EnhancedSettings(BaseSettings):
    """
    Enhanced application settings with comprehensive AWS and advanced features support.
    
    Supports:
    - Multi-environment configuration
    - AWS service integration (S3, MediaConvert, IVS, SageMaker, etc.)
    - Database sharding and read replicas
    - Advanced caching with Redis Cluster
    - Security hardening (rate limiting, encryption, etc.)
    - Feature flags for gradual rollout
    - Observability (logging, metrics, tracing)
    """
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )
    
    # ============================================================================
    # APPLICATION SETTINGS
    # ============================================================================
    
    ENVIRONMENT: Environment = Environment.DEVELOPMENT
    DEBUG: bool = False
    TESTING: bool = False
    
    PROJECT_NAME: str = "Social Flow"
    PROJECT_DESCRIPTION: str = "Advanced social media platform with video streaming"
    VERSION: str = "2.0.0"
    API_V1_STR: str = "/api/v1"
    API_V2_STR: str = "/api/v2"
    
    # Server configuration
    SERVER_HOST: str = "0.0.0.0"
    SERVER_PORT: int = 8000
    SERVER_WORKERS: int = 4
    
    # CORS
    BACKEND_CORS_ORIGINS: List[str] = Field(default_factory=lambda: [
        "http://localhost:3000",
        "http://localhost:8080",
    ])
    ALLOWED_HOSTS: List[str] = Field(default_factory=lambda: ["*"])
    
    @field_validator("BACKEND_CORS_ORIGINS", mode="before")
    @classmethod
    def assemble_cors_origins(cls, v: Union[str, List[str]]) -> List[str]:
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",")]
        return v
    
    # ============================================================================
    # SECURITY SETTINGS
    # ============================================================================
    
    SECRET_KEY: str = Field(default_factory=lambda: secrets.token_urlsafe(32))
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7
    PASSWORD_RESET_TOKEN_EXPIRE_HOURS: int = 1
    EMAIL_VERIFICATION_TOKEN_EXPIRE_HOURS: int = 24
    
    # Password policy
    PASSWORD_MIN_LENGTH: int = 8
    PASSWORD_REQUIRE_UPPERCASE: bool = True
    PASSWORD_REQUIRE_LOWERCASE: bool = True
    PASSWORD_REQUIRE_DIGIT: bool = True
    PASSWORD_REQUIRE_SPECIAL: bool = True
    
    # Rate limiting
    RATE_LIMIT_ENABLED: bool = True
    RATE_LIMIT_PER_MINUTE: int = 60
    RATE_LIMIT_PER_HOUR: int = 1000
    RATE_LIMIT_PER_DAY: int = 10000
    
    # Session management
    SESSION_COOKIE_NAME: str = "session_id"
    SESSION_EXPIRE_SECONDS: int = 86400  # 24 hours
    MAX_SESSIONS_PER_USER: int = 5
    
    # 2FA
    TWO_FACTOR_ISSUER_NAME: str = "Social Flow"
    TWO_FACTOR_ENABLED: bool = True
    
    # ============================================================================
    # DATABASE SETTINGS
    # ============================================================================
    
    # Primary database
    POSTGRES_SERVER: str = "localhost"
    POSTGRES_USER: str = "postgres"
    POSTGRES_PASSWORD: str = "password"
    POSTGRES_DB: str = "social_flow"
    POSTGRES_PORT: int = 5432
    DATABASE_URL: Optional[str] = None
    
    # Read replicas for scaling reads
    DATABASE_READ_REPLICAS: List[str] = Field(default_factory=list)
    
    # Connection pool settings
    DB_POOL_SIZE: int = 20
    DB_MAX_OVERFLOW: int = 40
    DB_POOL_TIMEOUT: int = 30
    DB_POOL_RECYCLE: int = 3600
    DB_ECHO: bool = False
    
    # Database sharding configuration
    DB_SHARDING_ENABLED: bool = False
    DB_SHARD_COUNT: int = 4
    DB_SHARDS: Dict[str, str] = Field(default_factory=dict)
    
    @field_validator("DATABASE_URL", mode="before")
    @classmethod
    def assemble_db_connection(cls, v: Optional[str], info) -> str:
        if v:
            return v
        
        postgres_user = info.data.get("POSTGRES_USER")
        postgres_password = info.data.get("POSTGRES_PASSWORD")
        postgres_server = info.data.get("POSTGRES_SERVER")
        postgres_port = info.data.get("POSTGRES_PORT", 5432)
        postgres_db = info.data.get("POSTGRES_DB")
        
        return f"postgresql+asyncpg://{postgres_user}:{postgres_password}@{postgres_server}:{postgres_port}/{postgres_db}"
    
    # ============================================================================
    # REDIS SETTINGS
    # ============================================================================
    
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_PASSWORD: Optional[str] = None
    REDIS_DB: int = 0
    REDIS_URL: Optional[str] = None
    
    # Redis Cluster support
    REDIS_CLUSTER_ENABLED: bool = False
    REDIS_CLUSTER_NODES: List[str] = Field(default_factory=list)
    
    # Redis connection pool
    REDIS_MAX_CONNECTIONS: int = 50
    REDIS_SOCKET_TIMEOUT: int = 5
    REDIS_SOCKET_CONNECT_TIMEOUT: int = 5
    
    # Caching configuration
    CACHE_TTL_DEFAULT: int = 300  # 5 minutes
    CACHE_TTL_USER_PROFILE: int = 600  # 10 minutes
    CACHE_TTL_VIDEO_METADATA: int = 1800  # 30 minutes
    CACHE_TTL_FEED: int = 60  # 1 minute
    
    @field_validator("REDIS_URL", mode="before")
    @classmethod
    def assemble_redis_connection(cls, v: Optional[str], info) -> str:
        if v:
            return v
        
        redis_password = info.data.get("REDIS_PASSWORD")
        redis_host = info.data.get("REDIS_HOST")
        redis_port = info.data.get("REDIS_PORT", 6379)
        redis_db = info.data.get("REDIS_DB", 0)
        
        auth = f":{redis_password}@" if redis_password else ""
        return f"redis://{auth}{redis_host}:{redis_port}/{redis_db}"
    
    # ============================================================================
    # AWS SETTINGS
    # ============================================================================
    
    # AWS Credentials
    AWS_ACCESS_KEY_ID: Optional[str] = None
    AWS_SECRET_ACCESS_KEY: Optional[str] = None
    AWS_REGION: str = "us-east-1"
    AWS_ACCOUNT_ID: Optional[str] = None
    
    # S3 Configuration
    AWS_S3_BUCKET_VIDEOS: str = "social-flow-videos"
    AWS_S3_BUCKET_IMAGES: str = "social-flow-images"
    AWS_S3_BUCKET_THUMBNAILS: str = "social-flow-thumbnails"
    AWS_S3_BUCKET_UPLOADS: str = "social-flow-uploads"
    
    # CloudFront CDN
    AWS_CLOUDFRONT_DOMAIN: Optional[str] = None
    AWS_CLOUDFRONT_KEY_PAIR_ID: Optional[str] = None
    AWS_CLOUDFRONT_PRIVATE_KEY_PATH: Optional[str] = None
    
    # MediaConvert for video encoding
    AWS_MEDIACONVERT_ENDPOINT: Optional[str] = None
    AWS_MEDIACONVERT_ROLE_ARN: Optional[str] = None
    AWS_MEDIACONVERT_QUEUE_ARN: Optional[str] = None
    
    # IVS for live streaming
    AWS_IVS_ENABLED: bool = True
    AWS_IVS_CHANNEL_TYPE: str = "STANDARD"  # STANDARD or BASIC
    
    # SageMaker for ML
    AWS_SAGEMAKER_ENDPOINT_MODERATION: Optional[str] = None
    AWS_SAGEMAKER_ENDPOINT_RECOMMENDATIONS: Optional[str] = None
    AWS_SAGEMAKER_ENDPOINT_COPYRIGHT: Optional[str] = None
    
    # SQS for message queuing
    AWS_SQS_QUEUE_VIDEO_PROCESSING: Optional[str] = None
    AWS_SQS_QUEUE_NOTIFICATIONS: Optional[str] = None
    AWS_SQS_QUEUE_ANALYTICS: Optional[str] = None
    
    # SNS for notifications
    AWS_SNS_TOPIC_ALERTS: Optional[str] = None
    AWS_SNS_TOPIC_NOTIFICATIONS: Optional[str] = None
    
    # Secrets Manager
    AWS_SECRETS_MANAGER_ENABLED: bool = True
    AWS_SECRETS_MANAGER_SECRET_NAME: str = "social-flow/backend"
    
    # KMS for encryption
    AWS_KMS_KEY_ID: Optional[str] = None
    
    # X-Ray for distributed tracing
    AWS_XRAY_ENABLED: bool = False
    AWS_XRAY_SAMPLING_RATE: float = 0.1
    
    # ============================================================================
    # VIDEO PROCESSING SETTINGS
    # ============================================================================
    
    # Upload configuration
    VIDEO_MAX_FILE_SIZE_MB: int = 5000  # 5GB
    VIDEO_CHUNK_SIZE_MB: int = 10  # 10MB chunks for multipart upload
    VIDEO_ALLOWED_FORMATS: List[str] = Field(default_factory=lambda: [
        "mp4", "mov", "avi", "mkv", "webm", "flv"
    ])
    
    # Encoding configuration
    VIDEO_ENCODING_FORMATS: List[str] = Field(default_factory=lambda: ["HLS", "DASH"])
    VIDEO_ENCODING_QUALITIES: List[str] = Field(default_factory=lambda: [
        "240p", "360p", "480p", "720p", "1080p", "1440p", "2160p"
    ])
    VIDEO_ENCODING_BITRATES: Dict[str, int] = Field(default_factory=lambda: {
        "240p": 400000,
        "360p": 800000,
        "480p": 1200000,
        "720p": 2500000,
        "1080p": 5000000,
        "1440p": 8000000,
        "2160p": 16000000,
    })
    
    # Thumbnail generation
    THUMBNAIL_COUNT: int = 5
    THUMBNAIL_WIDTH: int = 1280
    THUMBNAIL_HEIGHT: int = 720
    THUMBNAIL_FORMAT: str = "jpg"
    THUMBNAIL_QUALITY: int = 85
    
    # View counting
    VIEW_COUNT_BATCH_SIZE: int = 100
    VIEW_COUNT_FLUSH_INTERVAL_SECONDS: int = 60
    VIEW_MINIMUM_WATCH_TIME_SECONDS: int = 3
    
    # ============================================================================
    # LIVE STREAMING SETTINGS
    # ============================================================================
    
    LIVE_STREAM_MAX_DURATION_HOURS: int = 24
    LIVE_STREAM_MAX_BITRATE: int = 6000000  # 6 Mbps
    LIVE_STREAM_LATENCY_MODE: str = "LOW"  # LOW or NORMAL
    LIVE_STREAM_RECORDING_ENABLED: bool = True
    LIVE_STREAM_CHAT_ENABLED: bool = True
    LIVE_STREAM_CHAT_MESSAGE_LIMIT: int = 200  # per second
    
    # ============================================================================
    # AI/ML SETTINGS
    # ============================================================================
    
    # Content moderation
    ML_MODERATION_ENABLED: bool = True
    ML_MODERATION_THRESHOLD: float = 0.85
    ML_MODERATION_AUTO_REJECT: bool = True
    
    # Recommendations
    ML_RECOMMENDATIONS_ENABLED: bool = True
    ML_RECOMMENDATIONS_MODEL: str = "collaborative_filtering"
    ML_RECOMMENDATIONS_RETRAIN_DAYS: int = 7
    
    # Copyright detection
    ML_COPYRIGHT_ENABLED: bool = True
    ML_COPYRIGHT_MATCH_THRESHOLD_SECONDS: int = 7
    ML_COPYRIGHT_FINGERPRINT_ENABLED: bool = True
    
    # Sentiment analysis
    ML_SENTIMENT_ENABLED: bool = True
    ML_SENTIMENT_LANGUAGES: List[str] = Field(default_factory=lambda: [
        "en", "es", "fr", "de", "ja", "ko", "zh"
    ])
    
    # ============================================================================
    # ADVERTISEMENT SETTINGS
    # ============================================================================
    
    ADS_ENABLED: bool = True
    ADS_VIDEO_DURATION_SECONDS: int = 7
    ADS_MAX_PER_VIDEO: int = 3
    ADS_SKIP_AFTER_SECONDS: int = 5
    ADS_REVENUE_SHARE_PERCENTAGE: int = 55  # Creator gets 55%
    
    # Ad targeting
    ADS_TARGETING_GEO_ENABLED: bool = True
    ADS_TARGETING_DEMOGRAPHIC_ENABLED: bool = True
    ADS_TARGETING_INTEREST_ENABLED: bool = True
    ADS_TARGETING_ML_ENABLED: bool = True
    
    # ============================================================================
    # PAYMENT SETTINGS
    # ============================================================================
    
    # Stripe
    STRIPE_SECRET_KEY: Optional[str] = None
    STRIPE_PUBLISHABLE_KEY: Optional[str] = None
    STRIPE_WEBHOOK_SECRET: Optional[str] = None
    STRIPE_CONNECT_ENABLED: bool = True
    
    # Payment processing
    PAYMENT_CURRENCY: str = "USD"
    PAYMENT_MIN_PAYOUT_AMOUNT: int = 5000  # $50 in cents
    PAYMENT_PAYOUT_SCHEDULE_DAYS: int = 7
    
    # Watch time revenue calculation
    PAYMENT_REVENUE_PER_MINUTE_CENTS: int = 1  # $0.01 per minute watched
    PAYMENT_PREMIUM_MULTIPLIER: float = 2.0  # 2x for premium subscribers
    
    # ============================================================================
    # NOTIFICATION SETTINGS
    # ============================================================================
    
    # Firebase Cloud Messaging
    FCM_SERVER_KEY: Optional[str] = None
    FCM_ENABLED: bool = True
    
    # Email (SendGrid)
    SENDGRID_API_KEY: Optional[str] = None
    SENDGRID_FROM_EMAIL: EmailStr = "noreply@socialflow.com"  # type: ignore
    SENDGRID_FROM_NAME: str = "Social Flow"
    
    # SMS (Twilio)
    TWILIO_ACCOUNT_SID: Optional[str] = None
    TWILIO_AUTH_TOKEN: Optional[str] = None
    TWILIO_PHONE_NUMBER: Optional[str] = None
    
    # WebSocket
    WEBSOCKET_ENABLED: bool = True
    WEBSOCKET_MAX_CONNECTIONS: int = 10000
    WEBSOCKET_HEARTBEAT_INTERVAL: int = 30
    
    # ============================================================================
    # BACKGROUND JOBS SETTINGS
    # ============================================================================
    
    # Celery
    CELERY_BROKER_URL: Optional[str] = None
    CELERY_RESULT_BACKEND: Optional[str] = None
    CELERY_TASK_ALWAYS_EAGER: bool = False  # Set True for testing
    
    @field_validator("CELERY_BROKER_URL", mode="before")
    @classmethod
    def assemble_celery_broker(cls, v: Optional[str], info) -> str:
        if v:
            return v
        return info.data.get("REDIS_URL", "redis://localhost:6379/0")
    
    # Task configuration
    TASK_VIDEO_ENCODING_PRIORITY: int = 5
    TASK_NOTIFICATION_PRIORITY: int = 10
    TASK_ANALYTICS_PRIORITY: int = 3
    
    # ============================================================================
    # OBSERVABILITY SETTINGS
    # ============================================================================
    
    # Logging
    LOG_LEVEL: LogLevel = LogLevel.INFO
    LOG_FORMAT: str = "json"  # json or text
    LOG_TO_FILE: bool = True
    LOG_FILE_PATH: str = "logs/app.log"
    LOG_FILE_MAX_BYTES: int = 10485760  # 10MB
    LOG_FILE_BACKUP_COUNT: int = 5
    
    # Metrics (Prometheus)
    METRICS_ENABLED: bool = True
    METRICS_PORT: int = 9090
    
    # Tracing (Jaeger/X-Ray)
    TRACING_ENABLED: bool = False
    TRACING_SAMPLE_RATE: float = 0.1
    JAEGER_AGENT_HOST: str = "localhost"
    JAEGER_AGENT_PORT: int = 6831
    
    # APM (Application Performance Monitoring)
    SENTRY_DSN: Optional[str] = None
    SENTRY_ENABLED: bool = False
    SENTRY_ENVIRONMENT: Optional[str] = None
    SENTRY_TRACES_SAMPLE_RATE: float = 0.1
    
    # ============================================================================
    # FEATURE FLAGS
    # ============================================================================
    
    FEATURE_SOCIAL_LOGIN: bool = True
    FEATURE_TWO_FACTOR_AUTH: bool = True
    FEATURE_LIVE_STREAMING: bool = True
    FEATURE_STORIES: bool = True
    FEATURE_REELS: bool = True
    FEATURE_POLLS: bool = True
    FEATURE_DONATIONS: bool = True
    FEATURE_SUBSCRIPTIONS: bool = True
    FEATURE_MARKETPLACE: bool = False  # Future feature
    FEATURE_MESSAGING: bool = True
    FEATURE_GROUPS: bool = False  # Future feature
    
    # ============================================================================
    # SEARCH SETTINGS
    # ============================================================================
    
    # Elasticsearch
    ELASTICSEARCH_ENABLED: bool = True
    ELASTICSEARCH_HOST: str = "localhost"
    ELASTICSEARCH_PORT: int = 9200
    ELASTICSEARCH_INDEX_VIDEOS: str = "videos"
    ELASTICSEARCH_INDEX_POSTS: str = "posts"
    ELASTICSEARCH_INDEX_USERS: str = "users"
    
    # OpenSearch (AWS alternative)
    OPENSEARCH_ENABLED: bool = False
    OPENSEARCH_ENDPOINT: Optional[str] = None
    
    # ============================================================================
    # CONTENT MODERATION & SAFETY
    # ============================================================================
    
    MODERATION_AUTO_ENABLED: bool = True
    MODERATION_MANUAL_REVIEW_THRESHOLD: float = 0.7
    MODERATION_QUEUE_MAX_SIZE: int = 1000
    
    # Content restrictions
    CONTENT_MAX_TAG_COUNT: int = 30
    CONTENT_MAX_HASHTAG_COUNT: int = 30
    CONTENT_MAX_MENTION_COUNT: int = 20
    CONTENT_MAX_DESCRIPTION_LENGTH: int = 5000
    
    # Geofencing
    GEOFENCING_ENABLED: bool = True
    GEOFENCING_DEFAULT_COUNTRIES: List[str] = Field(default_factory=lambda: ["US"])
    
    # Age restrictions
    AGE_RESTRICTION_ENABLED: bool = True
    AGE_MINIMUM_YEARS: int = 13
    
    # ============================================================================
    # ANALYTICS SETTINGS
    # ============================================================================
    
    ANALYTICS_ENABLED: bool = True
    ANALYTICS_RETENTION_DAYS: int = 90
    ANALYTICS_REALTIME_ENABLED: bool = True
    ANALYTICS_EXPORT_ENABLED: bool = True
    
    # ============================================================================
    # TESTING & DEBUG
    # ============================================================================
    
    ENABLE_METRICS: bool = True
    ENABLE_SWAGGER_UI: bool = True
    ENABLE_REDOC: bool = True
    
    # Mock external services in testing
    MOCK_AWS_SERVICES: bool = False
    MOCK_STRIPE: bool = False
    MOCK_EMAIL: bool = False
    
    @model_validator(mode="after")
    def validate_production_settings(self) -> "EnhancedSettings":
        """Validate critical production settings."""
        if self.ENVIRONMENT == Environment.PRODUCTION:
            if self.DEBUG:
                raise ValueError("DEBUG must be False in production")
            if self.SECRET_KEY == "changeme":
                raise ValueError("SECRET_KEY must be changed in production")
            if not self.AWS_CLOUDFRONT_DOMAIN:
                print("WARNING: AWS_CLOUDFRONT_DOMAIN not set in production")
        
        return self


# Singleton instance
settings = EnhancedSettings()


# Export for backward compatibility
__all__ = ["settings", "EnhancedSettings", "Environment", "LogLevel"]
