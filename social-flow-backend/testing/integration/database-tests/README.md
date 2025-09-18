# Database Integration Tests

Path: `testing/integration/database-tests/`

## Overview

This suite provides integration tests for the database layer including:
- Schema/migration checks (Alembic skeleton provided)
- Sync and async CRUD tests
- Transaction isolation & rollback tests
- Performance micro-benchmarks

## Quick start

1. Copy `.env.example` -> `.env` and set credentials.
2. Start a test database (see `ci/docker-compose.db.yml`) or point `.env` to an existing DB.
3. Create virtualenv and install:
