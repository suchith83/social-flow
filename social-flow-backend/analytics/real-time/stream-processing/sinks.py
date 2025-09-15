import aiokafka
from .config import settings
from .utils import get_logger

logger = get_logger(__name__)


async def publish_to_kafka(message: str) -> None:
    """Publish processed message to Kafka."""
    producer = aiokafka.AIOKafkaProducer(bootstrap_servers=settings.kafka_bootstrap)
    await producer.start()
    try:
        await producer.send_and_wait(settings.kafka_output_topic, message.encode("utf-8"))
        logger.debug(f"Published to {settings.kafka_output_topic}")
    finally:
        await producer.stop()
