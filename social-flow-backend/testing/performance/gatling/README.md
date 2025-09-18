# Gatling Performance Suite

Path: `testing/performance/gatling/`

## Quick start

1. Install Java (11+) and sbt.
2. Copy `.env.example` to `.env` and configure `BASE_URL` and credentials.
3. Run a single scenario:
   ```bash
   cd testing/performance/gatling
   sbt "gatling:testOnly simulations.AuthSimulation"
