# JMeter Performance Test Suite

Path: `testing/performance/jmeter/`

## Overview

This JMeter suite provides:
- Modular test plans (.jmx) for auth, baseline API, storage upload/download, and multi-cloud failover simulation.
- CSV feeders for test data (users, upload keys, item ids).
- Groovy JSR223 helpers for token caching, multipart uploads (presigned), result processing.
- Dockerized JMeter for headless execution and distributed mode.
- CI-friendly runner (GitHub Actions example).
- InfluxDB/Graphite backend listener wiring (optional) and HTML report post-processing.

## Quick start (local)

1. Copy `user.properties` -> `.jmeter.properties` and edit `base_url`, credentials etc (or set env vars).
2. Build Docker image (if desired):
   ```bash
   docker build -t perf-jmeter:latest .
