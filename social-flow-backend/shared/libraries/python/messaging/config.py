# common/libraries/python/messaging/config.py
"""
Messaging configuration loader.
Supports environment variable overrides.
"""

import os

class MessagingConfig:
    BROKER_TYPE = os.getenv("BROKER_TYPE", "redis")  # kafka | rabbitmq | redis
    BROKER_URL = os.getenv("BROKER_URL", "redis://localhost:6379/0")
    DEFAULT_TOPIC = os.getenv("BROKER_DEFAULT_TOPIC", "default_topic")

    # Retry and DLQ
    MAX_RETRIES = int(os.getenv("BROKER_MAX_RETRIES", "3"))
    DLQ_TOPIC = os.getenv("BROKER_DLQ_TOPIC", "dead_letter_queue")

    # Serialization
    SERIALIZER = os.getenv("BROKER_SERIALIZER", "json")  # json | protobuf
