"""
Configuration loader for database tests using pydantic. Loads .env automatically.
"""

from pydantic import BaseSettings, Field


class DBSettings(BaseSettings):
    db_vendor: str = Field("postgres", env="DB_VENDOR")  # "postgres" or "cockroach"
    db_host: str = Field("localhost", env="DB_HOST")
    db_port: int = Field(5432, env="DB_PORT")
    db_user: str = Field("postgres", env="DB_USER")
    db_password: str = Field("postgres", env="DB_PASSWORD")
    db_name: str = Field("testdb", env="DB_NAME")
    test_schema_prefix: str = Field("test_", env="TEST_SCHEMA_PREFIX")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = DBSettings()
