# Log Analysis Suite

This module provides **advanced log monitoring & analytics tools**.  

### Features
- JSON & plain-text log parsing.
- Pluggable backends: Local FS, Elasticsearch, Grafana Loki.
- Metrics aggregation (counts, error rates, latency).
- Anomaly detection with rules + optional ML (e.g., Isolation Forest).
- CI/CD integration to fail builds if error thresholds exceeded.

---

### Usage

```bash
# Install dependencies
npm ci

# Run analysis on local logs
node scripts/analyze_logs.js ./logs/app.log

# Tail logs in real-time
node scripts/tail_logs.js ./logs/app.log

# Generate HTML/JSON reports
node scripts/report_generator.js reports/*.json
