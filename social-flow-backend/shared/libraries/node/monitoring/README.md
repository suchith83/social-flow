# common/libraries/node/monitoring

Production-ready monitoring tooling for Node.js services.

## Features
- Prometheus metrics via `prom-client`
- System and process collectors (event loop, GC, CPU, memory)
- Expose `/metrics` via small Express server
- Optional Prometheus Pushgateway support
- OpenTelemetry tracing bootstrap (Jaeger / OTLP)
- Express instrumentation middleware (request counts & latency)
- Health check registry & aggregator
- Alerts (Slack / PagerDuty) helper
- Structured logging wrapper (pino)

## Quick start

Install dependencies:
```bash
npm install prom-client express pino @opentelemetry/api @opentelemetry/sdk-node @opentelemetry/auto-instrumentations-node @opentelemetry/exporter-jaeger @opentelemetry/exporter-collector node-fetch axios dotenv
