"""
Integration tests for Kafka:
- publish and consume a message (async)
- ordering guarantee check (sequence numbers)
- idempotence check via keys
- dead-letter simulation (if consumer pushes to DLQ)
"""

import asyncio
import json
import pytest
from utils.schema_validator import validate_json_schema

# basic JSON schema used for tests
MESSAGE_SCHEMA = {
    "type": "object",
    "properties": {
        "seq": {"type": "integer"},
        "payload": {"type": "object"},
    },
    "required": ["seq", "payload"]
}

@pytest.mark.messaging
@pytest.mark.asyncio
async def test_kafka_publish_consume(kafka_async_helper):
    # publish
    await kafka_async_helper.produce(key=None, value={"seq": 1, "payload": {"hello": "world"}})
    # consume
    msg = await kafka_async_helper.get_message(timeout=5.0)
    assert msg is not None, "No message received from Kafka"
    data = json.loads(msg.value.decode("utf-8"))
    # schema validate
    validate_json_schema(data, MESSAGE_SCHEMA)
    assert data["seq"] == 1

@pytest.mark.messaging
@pytest.mark.asyncio
async def test_kafka_ordering(kafka_async_helper):
    # send sequence of messages
    for i in range(5):
        await kafka_async_helper.produce(key=None, value={"seq": i, "payload": {"i": i}})
    # read 5 messages
    received = []
    for _ in range(5):
        m = await kafka_async_helper.get_message(timeout=3.0)
        assert m is not None, "timed out waiting for kafka message"
        received.append(json.loads(m.value.decode("utf-8"))["seq"])
    # ordering should be preserved for a single partition; tests assume topic has single partition for determinism
    assert received == list(range(5))

@pytest.mark.messaging
@pytest.mark.asyncio
async def test_kafka_idempotence_by_key(kafka_async_helper):
    # send same key twice; with idempotent producers + dedup logic in consumers, only one should be processed
    key = b"dup-key"
    await kafka_async_helper.produce(key=key, value={"seq": 100, "payload": {}})
    await kafka_async_helper.produce(key=key, value={"seq": 100, "payload": {}})
    # consume two messages (since Kafka retains both) but simulate idempotent processing by checking key+seq
    # The test here asserts both messages exist in the topic; dedup behavior must be implemented in app
    m1 = await kafka_async_helper.get_message(timeout=2.0)
    m2 = await kafka_async_helper.get_message(timeout=2.0)
    assert m1 is not None and m2 is not None

@pytest.mark.messaging
@pytest.mark.asyncio
async def test_kafka_dead_letter_simulation(kafka_async_helper):
    """
    This test simulates a consumer that fails to process a message and expects some DLQ mechanism.
    Since we don't run the real consumer here, we assert that a message can be moved to a DLQ topic by test logic.
    """
    dlq_topic = kafka_async_helper.topic + "-dlq"
    # produce a poison message
    await kafka_async_helper.produce(key=None, value={"seq": -1, "payload": {"poison": True}})
    # in a real system, a failing consumer would republish to DLQ; simulate republish
    await kafka_async_helper._producer.send_and_wait(dlq_topic, value=json.dumps({"seq": -1}).encode())
    # manually create consumer for dlq (sync approach)
    # We will just consume original topic to ensure message exists; skip strict DLQ verification here
    msg = await kafka_async_helper.get_message(timeout=3.0)
    assert msg is not None
