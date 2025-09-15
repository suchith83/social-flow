import asyncio
import aiokafka
from typing import AsyncGenerator
from .config import settings
from .utils import get_logger

logger = get_logger(__name__)


async def read_from_kafka() -> AsyncGenerator[str, None]:
    """
    Consume raw events from Kafka topic asynchronously.
    Yields decoded event strings.
    """
    consumer = aiokafka.AIOKafkaConsumer(
        settings.kafka_topic,
        bootstrap_servers=settings.kafka_bootstrap,
        value_deserializer=lambda m: m.decode("utf-8"),
        auto_offset_reset="earliest",
        enable_auto_commit=True,
    )
    await consumer.start()
    logger.info(f"Subscribed to Kafka topic {settings.kafka_topic}")
    try:
        async for msg in consumer:
            yield msg.value
    finally:
        await consumer.stop()
        logger.info("Kafka consumer closed")
