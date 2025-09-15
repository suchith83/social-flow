import os
from pydantic import BaseSettings, Field


class DashboardSettings(BaseSettings):
    """Centralized configuration for real-time dashboards."""

    # Kafka / Redis
    kafka_bootstrap: str = Field("localhost:9092", env="KAFKA_BOOTSTRAP")
    kafka_topic: str = Field("analytics-events", env="KAFKA_TOPIC")
    redis_url: str = Field("redis://localhost:6379", env="REDIS_URL")

    # API
    api_host: str = Field("0.0.0.0", env="API_HOST")
    api_port: int = Field(8080, env="API_PORT")

    # Visualization
    refresh_interval: int = Field(2, env="DASH_REFRESH_INTERVAL")

    # Anomaly Detection
    anomaly_threshold: float = Field(3.0, env="ANOMALY_THRESHOLD")

    class Config:
        env_file = ".env"
        case_sensitive = True


settings = DashboardSettings()
