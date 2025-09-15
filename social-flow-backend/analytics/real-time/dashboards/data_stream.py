import asyncio
import aiokafka
from typing import AsyncGenerator
from .config import settings
from .utils import get_logger

logger = get_logger(__name__)


async def consume_kafka() -> AsyncGenerator[dict, None]:
    """
    Asynchronous Kafka consumer yielding JSON events.
    """
    consumer = aiokafka.AIOKafkaConsumer(
        settings.kafka_topic,
        bootstrap_servers=settings.kafka_bootstrap,
        value_deserializer=lambda m: m.decode("utf-8"),
        auto_offset_reset="earliest",
        enable_auto_commit=True,
    )
    await consumer.start()
    try:
        logger.info("Kafka consumer started")
        async for msg in consumer:
            yield msg.value
    finally:
        await consumer.stop()
        logger.info("Kafka consumer stopped")
