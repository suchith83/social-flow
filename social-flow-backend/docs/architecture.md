# Architecture — High Level

Overview
- Microservices architecture with language-agnostic services:
  - Python FastAPI services (recommendation, analytics, search, view-count)
  - Node services (video-service)
  - Workers (Celery / custom workers)
  - Common libraries shared under common/libraries

Key components
- API Gateway / App: central auth, routing, rate limits
- Auth service: JWT issuance, RBAC — used across services
- Storage: S3-compatible object store for videos, thumbnails
- DBs: PostgreSQL for relational data; Redis for cache & message broker
- Messaging: Redis pub/sub or Kafka for high-throughput events
- ML: ai-models/ stubs and model-serving pipelines for recommendations & moderation
- Workers: background processors for encoding, analytics, and ingestion

Data flows (example)
1. User uploads video → Video Service stores chunks in S3 → Enqueue encoding job.
2. Encoder worker processes video → writes HLS artifacts to S3 → updates Video metadata in DB.
3. View events emitted to analytics → Analytics service ingests and stores metrics.
4. Feedback events published to recommendation.feedback → Recommendation worker persists feedback for model retraining.

Design considerations
- Keep services stateless; persist state in DBs or object storage.
- Use idempotent operations for workers and retries.
- Expose health/readiness endpoints for orchestration.

Extensibility
- Replace ai-models stubs with production model servers (TorchServe, SageMaker).
- Swap DummyBroker with Redis/Kafka in production.
