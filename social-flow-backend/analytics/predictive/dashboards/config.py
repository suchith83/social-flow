from pydantic import BaseSettings, Field


class DashboardSettings(BaseSettings):
    """Configuration for predictive dashboards"""

    # Data warehouse
    SNOWFLAKE_URI: str = Field(..., env="SNOWFLAKE_URI")

    # Dashboard
    DASHBOARD_HOST: str = Field(default="0.0.0.0")
    DASHBOARD_PORT: int = Field(default=8050)
    DASHBOARD_TITLE: str = Field(default="SocialFlow Predictive Analytics")

    # Auth
    ADMIN_USER: str = Field(default="admin")
    ADMIN_PASSWORD: str = Field(..., env="DASHBOARD_ADMIN_PASSWORD")

    # Monitoring
    PROMETHEUS_PUSHGATEWAY: str = Field(default="http://localhost:9091")

    class Config:
        env_file = ".env"
        case_sensitive = True


dashboard_settings = DashboardSettings()
