import pytest
from analytics_data.aggregator import AnalyticsAggregator
from analytics_data.models import AnalyticsEvent
from analytics_data.utils import utc_now


def test_aggregator_update():
    agg = AnalyticsAggregator()
    event = AnalyticsEvent(
        event_id="1",
        user_id="u1",
        video_id="v1",
        event_type="view",
        timestamp=utc_now(),
        metadata={},
    )
    agg.update(event)
    assert agg.metrics["v1"].total_views == 1
