import aiokafka
from typing import AsyncGenerator
from .config import settings
from .utils import get_logger

logger = get_logger(__name__)


async def consume_kafka() -> AsyncGenerator[str, None]:
    """Consume from Kafka input topic asynchronously."""
    consumer = aiokafka.AIOKafkaConsumer(
        settings.kafka_input_topic,
        bootstrap_servers=settings.kafka_bootstrap,
        value_deserializer=lambda m: m.decode("utf-8"),
        auto_offset_reset="earliest",
    )
    await consumer.start()
    logger.info(f"Subscribed to {settings.kafka_input_topic}")
    try:
        async for msg in consumer:
            yield msg.value
    finally:
        await consumer.stop()
        logger.info("Kafka consumer stopped")
