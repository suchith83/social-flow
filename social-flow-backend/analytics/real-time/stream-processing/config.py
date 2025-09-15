from pydantic import BaseSettings, Field


class StreamProcessingSettings(BaseSettings):
    """Configuration for stream processing system."""

    kafka_bootstrap: str = Field("localhost:9092", env="KAFKA_BOOTSTRAP")
    kafka_input_topic: str = Field("processed-events", env="KAFKA_INPUT_TOPIC")
    kafka_output_topic: str = Field("analytics-stream", env="KAFKA_OUTPUT_TOPIC")

    redis_url: str = Field("redis://localhost:6379", env="REDIS_URL")

    window_size_sec: int = Field(60, env="WINDOW_SIZE_SEC")
    slide_interval_sec: int = Field(10, env="SLIDE_INTERVAL_SEC")

    checkpoint_interval: int = Field(30, env="CHECKPOINT_INTERVAL")

    class Config:
        env_file = ".env"
        case_sensitive = True


settings = StreamProcessingSettings()
