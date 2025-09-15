import os
from pydantic import BaseSettings, Field


class Settings(BaseSettings):
    """Centralized configuration for ETL jobs"""

    # Database configurations
    POSTGRES_URI: str = Field(..., env="POSTGRES_URI")
    SNOWFLAKE_URI: str = Field(..., env="SNOWFLAKE_URI")

    # Cloud storage
    AWS_ACCESS_KEY_ID: str = Field(..., env="AWS_ACCESS_KEY_ID")
    AWS_SECRET_ACCESS_KEY: str = Field(..., env="AWS_SECRET_ACCESS_KEY")
    S3_BUCKET: str = Field(default="socialflow-analytics-bucket")

    # Kafka configs
    KAFKA_BROKERS: str = Field(default="localhost:9092")
    KAFKA_TOPIC: str = Field(default="socialflow-events")

    # Monitoring
    PROMETHEUS_PUSHGATEWAY: str = Field(default="http://localhost:9091")

    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()
