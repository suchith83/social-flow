from .producer import KafkaProducer
from .consumer import KafkaConsumer
from .topics import TOPICS
from .schema_registry import validate_json  # optional helper

__all__ = ["KafkaProducer", "KafkaConsumer", "TOPICS", "validate_json"]