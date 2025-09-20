import os
from dataclasses import dataclass
from typing import Optional

@dataclass
class Settings:
    ENV: str = os.environ.get("SF_MOBILE_ENV", "development")
    PUSH_PROVIDER: str = os.environ.get("SF_PUSH_PROVIDER", "fcm")  # 'fcm' or 'noop'
    PUSH_CREDENTIALS: Optional[str] = os.environ.get("SF_PUSH_CREDENTIALS")  # path or json
    API_PORT: int = int(os.environ.get("SF_MOBILE_PORT", "8101"))
    CACHE_TTL: int = int(os.environ.get("SF_MOBILE_CACHE_TTL", "60"))

settings = Settings()
