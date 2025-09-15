from pydantic import BaseSettings, Field


class EventProcessingSettings(BaseSettings):
    """Environment-driven configuration for event processing."""

    kafka_bootstrap: str = Field("localhost:9092", env="KAFKA_BOOTSTRAP")
    kafka_topic: str = Field("raw-events", env="KAFKA_TOPIC")

    redis_url: str = Field("redis://localhost:6379", env="REDIS_URL")

    postgres_dsn: str = Field("postgresql://user:pass@localhost/db", env="POSTGRES_DSN")
    s3_bucket: str = Field("analytics-events", env="S3_BUCKET")
    elasticsearch_url: str = Field("http://localhost:9200", env="ELASTIC_URL")

    max_batch_size: int = Field(100, env="MAX_BATCH_SIZE")
    retention_days: int = Field(7, env="RETENTION_DAYS")

    class Config:
        env_file = ".env"
        case_sensitive = True


settings = EventProcessingSettings()
