# Performance & Load Testing

This folder contains lightweight load-testing artifacts for Social Flow backend:
- k6 script (recommended): performance/k6/social_flow_test.js
- Artillery scenario: performance/artillery/social_flow.yml
- Simple runners: performance/benchmarks/run_k6.sh and performance/benchmarks/runner.py
- Prometheus query examples for analyzing results

Quickstart (k6)
1. Install k6: https://k6.io/docs/getting-started/installation/
2. Set target service base URL (default localhost:8003):
   export BASE_URL=http://localhost:8003
3. Run:
   bash performance/benchmarks/run_k6.sh --vus 50 --duration 60s

Quickstart (Artillery)
1. Install Artillery: `npm i -g artillery`
2. Run:
   artillery run performance/artillery/social_flow.yml

Notes
- Tests exercise `/health`, `/recommendations/{user_id}`, `/trending`, and `POST /feedback`.
- Adjust VUs, duration and endpoints via environment variables.
- Use the Prometheus queries in performance/metrics/prometheus_query_examples.md to evaluate impact.
