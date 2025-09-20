import os
from dataclasses import dataclass

@dataclass
class Settings:
    APP_NAME: str = os.environ.get("SF_APP_NAME", "social-flow")
    SECRET_KEY: str = os.environ.get("SF_SECRET_KEY", "change-me-in-prod")
    ACCESS_TOKEN_EXPIRE_SECONDS: int = int(os.environ.get("SF_ACCESS_EXPIRE", "3600"))
    DB_PATH: str = os.environ.get("SF_DB_PATH", os.path.join(os.path.dirname(__file__), "..", "data", "sf.db"))

settings = Settings()
