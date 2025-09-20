# Artillery Load Testing Suite

This module contains a **complete and advanced setup** for load-testing using [Artillery](https://artillery.io).

### Features
- Modular scenarios (smoke, user journey, soak).
- JWT authentication with service accounts.
- Reporting to InfluxDB and HTML summary.
- Dockerized execution.
- CI/CD integration with GitHub Actions.

### Usage

```bash
# Install dependencies
npm run install-deps

# Run smoke test
npm run smoke

# Run full user journey test
npm run user-journey

# Run soak test
npm run soak

# Aggregate reports
npm run report
