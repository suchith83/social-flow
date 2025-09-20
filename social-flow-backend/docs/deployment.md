# Deployment Guide

This document provides quick instructions for local development and pointers for production.

Local development (recommended)
1. Copy config/.env.example -> .env and adjust values.
2. Start core infra:
   - Option A (docker-compose): docker compose -f deployment/docker-compose.dev.yml up --build
   - Option B (automation helper): python automation/setup_local.py --create-venv --apply
3. Start services (examples):
   - API: python -m uvicorn app.main:app --port 8000
   - Recommendation: python -m uvicorn services.recommendation-service.src.main:app --port 8003
   - Analytics: python -m uvicorn analytics.src.main:app --port 8010
4. Apply DB migrations / seeds:
   - sqlite: sqlite3 .data/dev.db < data/migrations/sqlite/001_initial_schema.sql
   - seed: sqlite3 .data/dev.db < data/seeds/development/seed_data.sql

Production guidance
- Use Terraform / IaC to provision cloud resources (see deployment/terraform).
- Package services as Docker images and deploy with Kubernetes/Helm (see deployment/helm).
- Use managed databases (RDS), object storage (S3), and a message system (Kafka or Redis).
- Manage secrets with a secret manager (AWS Secrets Manager, Vault).
- Enforce readiness/liveness probes, resource limits, and network policies.

CI/CD
- GitHub Actions examples are under cicd/github-actions/.
- Build images and push to a registry; use immutable tags and image signing where possible.

Rollbacks & migrations
- Run DB migrations in a backward-compatible manner; use feature flags for schema rollout.
- Keep backups and snapshots before destructive migrations.
