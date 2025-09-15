import snowflake.connector
import pandas as pd
from .config import dashboard_settings
from .utils import logger


class DataService:
    """Fetch predictive data from warehouse"""

    def __init__(self):
        self.conn = snowflake.connector.connect(dashboard_settings.SNOWFLAKE_URI)

    def fetch_user_growth_forecast(self):
        query = """
        SELECT forecast_date, predicted_users, lower_ci, upper_ci
        FROM user_growth_predictions
        ORDER BY forecast_date
        LIMIT 100
        """
        df = pd.read_sql(query, self.conn)
        logger.info(f"Fetched {len(df)} rows from user_growth_predictions")
        return df

    def fetch_video_engagement_forecast(self):
        query = """
        SELECT forecast_date, predicted_views, predicted_likes, predicted_comments
        FROM video_engagement_predictions
        ORDER BY forecast_date
        LIMIT 100
        """
        df = pd.read_sql(query, self.conn)
        logger.info(f"Fetched {len(df)} rows from video_engagement_predictions")
        return df
