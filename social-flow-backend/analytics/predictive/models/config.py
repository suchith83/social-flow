from pydantic import BaseSettings, Field


class ModelsSettings(BaseSettings):
    # Data
    SNOWFLAKE_URI: str = Field(..., env="SNOWFLAKE_URI")
    FEATURE_STORE_PATH: str = Field(default="feature_store")  # local fs or s3 path prefix

    # Training
    RANDOM_SEED: int = Field(default=42)
    N_JOBS: int = Field(default=4)

    # Model storage
    MODEL_DIR: str = Field(default="models")
    DEFAULT_MODEL_NAME: str = Field(default="user_growth_xgb")

    # Optional MLFlow
    MLFLOW_URI: str | None = Field(default=None)

    class Config:
        env_file = ".env"
        case_sensitive = True


settings = ModelsSettings()
