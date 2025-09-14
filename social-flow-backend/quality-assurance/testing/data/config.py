"""
Configuration for test data package. Uses environment variables for overrides.
"""

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class DataConfig:
    """Immutable configuration for data utilities."""
    fixtures_dir: str = os.getenv("TEST_FIXTURES_DIR", "tests/fixtures")
    default_seed: int = int(os.getenv("TEST_DATA_SEED", "42"))
    default_batch_size: int = int(os.getenv("TEST_DATA_BATCH", "50"))
    locale: str = os.getenv("TEST_DATA_LOCALE", "en_US")
    max_string_length: int = int(os.getenv("TEST_MAX_STRING_LEN", "256"))

    @staticmethod
    def from_env() -> "DataConfig":
        return DataConfig(
            fixtures_dir=os.getenv("TEST_FIXTURES_DIR", "tests/fixtures"),
            default_seed=int(os.getenv("TEST_DATA_SEED", "42")),
            default_batch_size=int(os.getenv("TEST_DATA_BATCH", "50")),
            locale=os.getenv("TEST_DATA_LOCALE", "en_US"),
            max_string_length=int(os.getenv("TEST_MAX_STRING_LEN", "256")),
        )


DATA_CONFIG = DataConfig.from_env()
