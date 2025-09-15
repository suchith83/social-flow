from pydantic import BaseSettings, Field
from typing import Optional


class PipelinesSettings(BaseSettings):
    # Core
    RANDOM_SEED: int = Field(default=42)
    N_JOBS: int = Field(default=4)

    # Paths
    BASE_DIR: str = Field(default="analytics/predictive")
    MODEL_DIR: str = Field(default="analytics/predictive/models")
    FEATURE_STORE: str = Field(default="feature_store")

    # Snowflake / DW
    SNOWFLAKE_URI: Optional[str] = Field(default=None)

    # MLflow (optional)
    MLFLOW_TRACKING_URI: Optional[str] = Field(default=None)
    MLFLOW_EXPERIMENT: str = Field(default="predictive-pipelines")

    # Orchestration
    SCHEDULE_INTERVAL: str = Field(default="@daily")

    # Monitoring
    PROMETHEUS_PUSHGATEWAY: Optional[str] = Field(default=None)

    class Config:
        env_file = ".env"
        case_sensitive = True


settings = PipelinesSettings()
