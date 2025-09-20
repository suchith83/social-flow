# Monitoring — Quickstart & Components

This folder provides local-ready monitoring artifacts and configuration for:
- Prometheus (scrape config + alert rules)
- Alertmanager (simple receiver)
- OpenTelemetry Collector (OTLP <> Prometheus/Jaeger)
- Grafana dashboard stubs
- A tiny local metrics exporter for development tests

Quick local dev
1. Start Prometheus and Alertmanager:
   docker run -d --name alertmanager -p 9093:9093 -v $(pwd)/alertmanager:/etc/alertmanager prom/alertmanager
   docker run -d --name prometheus -p 9090:9090 -v $(pwd)/prometheus/prometheus.yml:/etc/prometheus/prometheus.yml prom/prometheus

2. Start the OTEL collector (optional):
   docker run -d --name otel-collector -p 4317:4317 -p 55681:55681 -v $(pwd)/opentelemetry/collector-config.yaml:/etc/otel/config.yaml otel/opentelemetry-collector

3. Run a local service that exposes Prometheus metrics (example):
   python monitoring/local_metrics/app_metrics.py

Notes
- Replace Alertmanager receivers with Slack/Email configs in production.
- Update scrape targets (prometheus.yml) to point to your services and ports.
- Grafana dashboards are provided as JSON stubs in grafana/dashboards; import them into Grafana for visualization.
