import pytest
from analytics_data.processor import AnalyticsProcessor


def test_duplicate_event():
    p = AnalyticsProcessor()
    e = {"video_id": "v1", "event_type": "view"}
    ev1 = p.process(e)
    ev2 = p.process(e)
    assert ev1 is not None
    assert ev2 is None
