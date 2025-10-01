# common/libraries/python/messaging/__init__.py
"""
Messaging Library - Framework Agnostic

Features:
- Abstraction over Kafka, RabbitMQ, Redis Pub/Sub
- Unified producer/consumer API
- Serialization (JSON, Protobuf-ready)
- Middleware (logging, tracing, monitoring)
- Retry handling and Dead Letter Queue (DLQ)
"""

__all__ = [
    "config",
    "serializers",
    "base_broker",
    "kafka_broker",
    "rabbitmq_broker",
    "redis_broker",
    "middleware",
    "dlq",
    "messaging_service",
]
