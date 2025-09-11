# common/libraries/node/messaging

Production-ready Node.js messaging abstraction with adapters and high-level Producer/Consumer.

## Features

- Pluggable adapters: Kafka (`kafkajs`) and RabbitMQ (`amqplib`)
- High-level `Producer` with batching, retries, backoff, idempotency keys
- High-level `Consumer` with handler wrappers, retries, DLQ support
- Schema registry with AJV for validating messages
- Utilities for safe JSON handling, id generation, backoff
- Custom error types

## Installation

Install only what you need:

```bash
npm install kafkajs amqplib ajv ajv-formats dotenv
