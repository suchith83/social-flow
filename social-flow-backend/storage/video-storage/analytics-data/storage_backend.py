"""
Storage backend for analytics data.
Supports Postgres/ClickHouse for historical queries.
"""

import sqlalchemy as sa
from sqlalchemy.orm import sessionmaker, declarative_base
import requests

from .config import config
from .models import AggregatedMetrics
from .utils import logger

Base = declarative_base()


class MetricsTable(Base):
    __tablename__ = "video_metrics"
    video_id = sa.Column(sa.String, primary_key=True)
    total_views = sa.Column(sa.Integer)
    total_likes = sa.Column(sa.Integer)
    total_comments = sa.Column(sa.Integer)
    total_watch_time = sa.Column(sa.Float)
    last_updated = sa.Column(sa.DateTime)


class AnalyticsStorage:
    def __init__(self):
        self.engine = sa.create_engine(config.DB_URL)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)

    def save_metrics(self, metrics: AggregatedMetrics):
        with self.Session() as s:
            db_obj = MetricsTable(**metrics.dict())
            s.merge(db_obj)
            s.commit()
            logger.info(f"Saved metrics for {metrics.video_id}")

    def save_clickhouse(self, metrics: AggregatedMetrics):
        try:
            query = f"""
            INSERT INTO video_metrics (video_id, total_views, total_likes, total_comments, total_watch_time, last_updated)
            VALUES ('{metrics.video_id}', {metrics.total_views}, {metrics.total_likes}, {metrics.total_comments}, {metrics.total_watch_time}, '{metrics.last_updated}')
            """
            requests.post(config.CLICKHOUSE_URL, data=query)
        except Exception as e:
            logger.error(f"ClickHouse save failed: {e}")
