# Event Streaming (Kafka-focused) — lightweight local stubs

This package provides a small Kafka-focused helper set for publishing and subscribing to events.

Key features:
- Topic constants in `event_streaming.kafka.topics`
- `KafkaProducer` wrapper that publishes JSON payloads
- `KafkaConsumer` wrapper that subscribes and delivers messages to callbacks
- Safe in-memory broker fallback for local development and unit tests when Kafka isn't available

Usage (producer):
```py
from event_streaming.kafka.producer import KafkaProducer
producer = KafkaProducer(bootstrap_servers="localhost:9092")
producer.publish("recommendation.feedback", {"user_id": "u1", "action": "view"})
```

Usage (consumer):
```py
from event_streaming.kafka.consumer import KafkaConsumer
def on_msg(msg): print("got", msg)
consumer = KafkaConsumer(bootstrap_servers="localhost:9092")
consumer.subscribe("recommendation.feedback", on_msg)
```

Notes:
- For production, install and configure Kafka and set correct bootstrap servers.
- The in-memory fallback is only for dev/tests and does not persist messages.
