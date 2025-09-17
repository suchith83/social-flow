"""
Aggregator: combines events into metrics
"""

from collections import defaultdict
import datetime
from typing import Dict

from .models import AnalyticsEvent, AggregatedMetrics
from .utils import utc_now, logger


class AnalyticsAggregator:
    def __init__(self):
        self.metrics: Dict[str, AggregatedMetrics] = {}

    def update(self, event: AnalyticsEvent):
        video_id = event.video_id
        if video_id not in self.metrics:
            self.metrics[video_id] = AggregatedMetrics(
                video_id=video_id,
                total_views=0,
                total_likes=0,
                total_comments=0,
                total_watch_time=0.0,
                last_updated=utc_now(),
            )

        m = self.metrics[video_id]
        if event.event_type == "view":
            m.total_views += 1
        elif event.event_type == "like":
            m.total_likes += 1
        elif event.event_type == "comment":
            m.total_comments += 1
        elif event.event_type == "watch_time":
            m.total_watch_time += float(event.metadata.get("seconds", 0))

        m.last_updated = utc_now()
        logger.debug(f"Updated metrics for {video_id}: {m.dict()}")
