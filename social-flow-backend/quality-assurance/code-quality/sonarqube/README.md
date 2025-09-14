# SonarQube QA Package

This folder contains an enterprise-ready SonarQube integration package designed to help teams onboard Sonar analysis into local development, CI, and production environments.

What’s included:

- `docker-compose.yml` — run a local SonarQube + PostgreSQL stack for dev/testing.
- `sonar-project.properties` — advanced properties used to configure analysis for multi-module projects (Kotlin/Java/JS/TS/Python examples).
- `templates/sonar-scanner.Dockerfile` — reproducible scanner image for CI.
- `scripts/` — automation scripts (bootstrap SonarQube quality gates, import/export quality profiles).
- `config/` — example Quality Gate JSON and Quality Profile XML (starter).
- `ci/github-actions-sonar.yml` — example GitHub Actions workflow showing PR and push analysis (secure tokens via secrets).

Goals:
- Provide a repeatable environment to run Sonar locally.
- Offer automation to configure quality gates and import quality profiles.
- Provide a CI workflow ready to be adapted to your repository.

Important notes:
- This package configures Sonar via the REST API. You must have administrative access to the SonarQube instance to run the setup scripts that create quality gates and import profiles.
- For production SonarQube, prefer to use a managed installer or helm chart and secure the instance behind authentication + ingress rules.

---

## Quick start (local)

1. Start SonarQube locally (dev/test):
```bash
cd quality-assurance/code-quality/sonarqube
docker-compose up -d
# Wait ~1-2 minutes for the server to initialize (first startup is slower)
# Default web UI: http://localhost:9000
