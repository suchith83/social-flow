"""
Processor for analytics events
- Validation
- Cleaning
- Enrichment
"""

from typing import Dict, Any
import datetime
import uuid

from .models import AnalyticsEvent
from .utils import json_hash, utc_now, logger


class AnalyticsProcessor:
    def __init__(self):
        self.seen_hashes = set()

    def process(self, event: Dict[str, Any]) -> AnalyticsEvent:
        """Validate and transform raw event"""
        try:
            event_obj = AnalyticsEvent(
                event_id=event.get("event_id", str(uuid.uuid4())),
                user_id=event.get("user_id"),
                video_id=event["video_id"],
                event_type=event["event_type"],
                timestamp=event.get("timestamp", utc_now()),
                metadata=event.get("metadata", {}),
            )
            h = json_hash(event_obj.dict())
            if h in self.seen_hashes:
                logger.warning("Duplicate event ignored")
                return None
            self.seen_hashes.add(h)
            return event_obj
        except Exception as e:
            logger.error(f"Invalid event: {e}")
            return None
