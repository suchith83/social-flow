
---

## `conftest.py`

```python
"""
Pytest fixtures for messaging integration tests.

Provides:
- kafka_producer / kafka_consumer (async & sync)
- rabbitmq connection & channel
- redis client & stream helpers

Fixtures are best-effort: if a broker is not reachable the fixture will skip tests gracefully.
"""

import os
import asyncio
import json
import time
import pytest
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).parent / ".env", override=False)

# config
KAFKA_BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
KAFKA_TOPIC = os.getenv("KAFKA_TEST_TOPIC", "integration-test-topic")
KAFKA_GROUP = os.getenv("KAFKA_CONSUMER_GROUP", "integration-test-group")

RABBIT_URL = os.getenv("RABBITMQ_URL", "amqp://guest:guest@localhost:5672/")
RABBIT_QUEUE = os.getenv("RABBITMQ_TEST_QUEUE", "integration_test_queue")
RABBIT_DLQ = os.getenv("RABBITMQ_DLQ", "integration_test_dlq")

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
REDIS_STREAM = os.getenv("REDIS_TEST_STREAM", "integration-test-stream")
REDIS_GROUP = os.getenv("REDIS_CONSUMER_GROUP", "integration-test-group")

MESSAGE_WAIT_SECONDS = int(os.getenv("MESSAGE_WAIT_SECONDS", "5"))

# clients
from clients.kafka_client import AsyncKafkaHelper, SyncKafkaHelper
from clients.rabbitmq_client import RabbitMQHelper
from clients.redis_streams_client import RedisStreamsHelper

@pytest.fixture(scope="session")
def event_loop():
    """Create module-level event loop for asyncio fixtures."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

# -------- Kafka fixtures (async) -----------
@pytest.fixture(scope="session")
async def kafka_async_helper():
    helper = AsyncKafkaHelper(bootstrap_servers=KAFKA_BOOTSTRAP, topic=KAFKA_TOPIC, group_id=KAFKA_GROUP)
    try:
        await helper.start()
    except Exception as e:
        pytest.skip(f"Kafka async helper cannot start: {e}")
    yield helper
    await helper.stop()

@pytest.fixture(scope="function")
def kafka_sync_helper():
    helper = SyncKafkaHelper(bootstrap_servers=KAFKA_BOOTSTRAP, topic=KAFKA_TOPIC)
    try:
        helper.start()
    except Exception as e:
        pytest.skip(f"Kafka sync helper cannot start: {e}")
    yield helper
    helper.stop()

# -------- RabbitMQ fixtures -----------
@pytest.fixture(scope="session")
def rabbitmq_helper():
    helper = RabbitMQHelper(amqp_url=RABBIT_URL)
    try:
        helper.connect()
    except Exception as e:
        pytest.skip(f"RabbitMQ not available: {e}")
    # ensure queues exist
    helper.declare_queue(RABBIT_QUEUE)
    helper.declare_queue(RABBIT_DLQ)
    yield helper
    helper.close()

# -------- Redis Streams fixtures -----------
@pytest.fixture(scope="session")
def redis_helper():
    helper = RedisStreamsHelper(redis_url=REDIS_URL)
    try:
        helper.connect()
    except Exception as e:
        pytest.skip(f"Redis not available: {e}")
    # ensure stream exists (XADD a dummy then delete)
    helper.ensure_stream(REDIS_STREAM)
    yield helper
    helper.close()
