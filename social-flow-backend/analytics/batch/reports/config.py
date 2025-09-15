from pydantic import BaseSettings, Field


class ReportSettings(BaseSettings):
    """Configuration for reporting system"""

    # Data warehouse connection
    SNOWFLAKE_URI: str = Field(..., env="SNOWFLAKE_URI")

    # Email configurations
    SMTP_SERVER: str = Field(default="smtp.gmail.com")
    SMTP_PORT: int = Field(default=587)
    SMTP_USER: str = Field(..., env="SMTP_USER")
    SMTP_PASSWORD: str = Field(..., env="SMTP_PASSWORD")

    # Report settings
    REPORT_OUTPUT_DIR: str = Field(default="reports/output")
    DEFAULT_FORMAT: str = Field(default="pdf")  # pdf / html

    class Config:
        env_file = ".env"
        case_sensitive = True


report_settings = ReportSettings()
