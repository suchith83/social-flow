# Messaging Integration Tests

Location: `testing/integration/messaging-tests/`

## Supported brokers
- Kafka
- RabbitMQ
- Redis Streams

## Quick start (local)
1. Copy `.env.example` -> `.env` and adjust if needed.
2. Start test brokers:
   - Using Docker Compose in `ci/docker-compose.messaging.yml`:
     ```bash
     docker compose -f testing/integration/messaging-tests/ci/docker-compose.messaging.yml up -d
     ```
3. Install dependencies (use a venv):
   ```bash
   pip install -r testing/integration/messaging-tests/requirements.txt
