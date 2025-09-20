# Synthetic Monitoring Suite

This module provides **synthetic monitoring checks** for APIs and web apps.  

### Features
- HTTP(S) uptime checks.
- Browser-based journeys (login, navigation).
- Threshold-based SLAs (latency, availability).
- Reporting to JSON & HTML.
- Docker + CI/CD integration.

---

### Usage

```bash
# Install dependencies
npm ci

# Run an API check
node scripts/run_check.js scenarios/api_check.js

# Run homepage check (Playwright)
node scripts/run_check.js scenarios/homepage_check.js

# Generate report
node scripts/report_generator.js reports/*.json
