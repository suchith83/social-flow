# Artillery Performance Test Suite

Location: `testing/performance/artillery/`

This folder contains Artillery scenarios, JS helpers, and CI integration to run performance tests for:
- Authentication flows
- API baseline endpoints
- Storage (upload & download)
- Multi-cloud failover simulation

## Quick start

1. Install dependencies (node >=16):
```bash
cd testing/performance/artillery
npm ci
