"""
Collector for raw analytics events.
Consumes from Kafka/Redis Streams and pushes to processor.
"""

import json
import time
from typing import Callable
from kafka import KafkaConsumer
import redis

from .config import config
from .utils import logger


class AnalyticsCollector:
    def __init__(self, on_event: Callable):
        self.on_event = on_event
        self.kafka_consumer = KafkaConsumer(
            "analytics-events",
            bootstrap_servers=config.KAFKA_BROKER,
            value_deserializer=lambda v: json.loads(v.decode("utf-8")),
            group_id="analytics-collector",
            auto_offset_reset="earliest",
            enable_auto_commit=True,
        )
        self.redis = redis.Redis.from_url(config.REDIS_URL)

    def run(self):
        logger.info("Starting AnalyticsCollector...")
        for msg in self.kafka_consumer:
            event = msg.value
            logger.debug(f"Collected event: {event}")
            try:
                self.on_event(event)
            except Exception as e:
                logger.error(f"Failed to process event: {e}")
                self.redis.lpush("analytics-dead-letter", json.dumps(event))


if __name__ == "__main__":
    def test_handler(event):
        print("Processed:", event)

    AnalyticsCollector(test_handler).run()
