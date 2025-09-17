import pytest
from analytics_data.processor import AnalyticsProcessor


def test_valid_event():
    processor = AnalyticsProcessor()
    event = {
        "user_id": "u1",
        "video_id": "v1",
        "event_type": "view",
        "metadata": {"seconds": 30},
    }
    processed = processor.process(event)
    assert processed is not None
    assert processed.video_id == "v1"
