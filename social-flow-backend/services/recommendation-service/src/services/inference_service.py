# Objective: Provide real-time personalized recommendations with low latency.

# Input Cases:
# - GET /api/v1/recommendations/:user id - Get recommendations
# - POST /api/v1/recommendations/feedback - Record user feedback
# - GET /api/v1/trending - Get trending content

# Output Cases:
# - Ranked list of recommended videos
# - Explanation of recommendations
# - A/B test variant assignments
# - Real-time performance metrics

from typing import List, Dict, Any, Union
from dataclasses import dataclass, asdict
import json

# Guard imports from the monorepo common package; provide a DummyBroker when missing
try:
    from common.libraries.python.messaging.redis_broker import RedisBroker
except Exception:
    RedisBroker = None  # type: ignore

# Provide a simple no-op broker to allow local runs / tests without Redis
class DummyBroker:
    def publish(self, topic: str, payload: dict) -> None:
        # No-op publish; could be extended to write to a local file if needed.
        return

    def subscribe(self, topic: str, callback):
        # Not implemented for dummy
        return

@dataclass
class FeedbackMessage:
    user_id: str
    item_id: str
    action: str
    timestamp: int


class InferenceService:
    def __init__(self):
        # get_logger import might fail in some minimal test environments — try to import here.
        try:
            from common.libraries.python.monitoring.logger import get_logger as _get_logger
            self.logger = _get_logger("inference-service")
        except Exception:
            import logging
            l = logging.getLogger("inference-service")
            if not l.handlers:
                l.addHandler(logging.StreamHandler())
            l.setLevel(logging.INFO)
            self.logger = l

        # Use redis broker if available; otherwise use DummyBroker to avoid crashes.
        if RedisBroker is not None:
            try:
                self.broker = RedisBroker()
            except Exception:
                self.logger.warning("Failed to initialize RedisBroker, falling back to DummyBroker")
                self.broker = DummyBroker()
        else:
            self.logger.info("RedisBroker not available; using DummyBroker")
            self.broker = DummyBroker()

    def get_recommendations(self, user_id: str, limit: int = 20) -> List[Dict[str, Any]]:
        self.logger.info(f"get_recommendations user={user_id} limit={limit}")
        # Simple candidate generator: placeholder for model call
        return [{"item_id": f"video_{i}", "score": 1.0 / (i + 1)} for i in range(limit)]

    def record_feedback(self, feedback_data: Union[FeedbackMessage, dict, Any]) -> None:
        """
        Accepts FeedbackMessage dataclass instances, dicts, or pydantic models and
        publishes a normalized JSON payload to the feedback topic.
        """
        # Normalize to dict
        if hasattr(feedback_data, "dict"):  # pydantic model
            payload = feedback_data.dict()
        elif hasattr(feedback_data, "__dict__") and not isinstance(feedback_data, dict):
            # dataclass or object
            try:
                payload = asdict(feedback_data)  # will work for dataclasses
            except Exception:
                payload = dict(feedback_data.__dict__)
        else:
            payload = dict(feedback_data)

        # Basic validation / shape enforcement
        required_keys = {"user_id", "item_id", "action", "timestamp"}
        missing = required_keys - set(payload.keys())
        if missing:
            self.logger.error(f"Invalid feedback payload, missing keys: {missing}")
            raise ValueError(f"Missing keys in feedback payload: {missing}")

        self.logger.info(f"publishing feedback for user={payload.get('user_id')} item={payload.get('item_id')}")
        try:
            # Convert values to JSON-serializable types
            self.broker.publish("recommendation.feedback", payload)
        except Exception:
            self.logger.exception("Failed to publish feedback payload")

    def get_trending(self, limit: int = 20) -> List[Dict[str, Any]]:
        self.logger.info("get_trending")
        return [{"item_id": f"trending_{i}", "views": 1000 - i * 10} for i in range(limit)]
        self.logger.info("get_trending")
        return [{"item_id": f"trending_{i}", "views": 1000 - i * 10} for i in range(limit)]
