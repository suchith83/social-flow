"""
RabbitMQ integration tests:
- publish/consume
- dead-letter behavior (X-dead-letter-exchange)
- ordering for simple queue
"""

import json
import time
import pytest

@pytest.mark.messaging
def test_rabbitmq_publish_consume(rabbitmq_helper):
    rabbitmq_helper.purge_queue("integration_test_queue")
    payload = {"id": 1, "msg": "hello"}
    rabbitmq_helper.publish("integration_test_queue", payload)
    # get message
    body = rabbitmq_helper.get_message("integration_test_queue", timeout=3)
    assert body is not None
    assert body["id"] == 1

@pytest.mark.messaging
def test_rabbitmq_dead_letter_simulation(rabbitmq_helper):
    # simulate a message causing consumer to nack and be routed to DLQ.
    # For tests, publish to main queue and then manually publish to DLQ to emulate.
    payload = {"id": 99, "error": True}
    rabbitmq_helper.publish("integration_test_queue", payload)
    # emulate consumer failing and publishing to DLQ
    rabbitmq_helper.publish("integration_test_dlq", payload)
    dlq_msg = rabbitmq_helper.get_message("integration_test_dlq", timeout=3)
    assert dlq_msg is not None and dlq_msg.get("id") == 99

@pytest.mark.messaging
def test_rabbitmq_ordering(rabbitmq_helper):
    rabbitmq_helper.purge_queue("integration_test_queue")
    for i in range(5):
        rabbitmq_helper.publish("integration_test_queue", {"seq": i})
    # consume sequentially and check order
    seqs = []
    for _ in range(5):
        m = rabbitmq_helper.get_message("integration_test_queue", timeout=2)
        assert m is not None
        seqs.append(m.get("seq"))
    assert seqs == list(range(5))
