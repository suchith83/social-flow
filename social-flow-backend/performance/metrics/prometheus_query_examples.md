# Prometheus query examples for performance runs

- Total request rate (per job):
  sum(rate(http_requests_total{job=~".*"}[1m])) by (job)

- 5xx error rate (percentage):
  sum(rate(http_requests_total{status=~"5.."}[5m])) by (job)
  /
  sum(rate(http_requests_total[5m])) by (job)

- 95th percentile latency (per job):
  histogram_quantile(0.95, sum(rate(http_request_duration_seconds_bucket[5m])) by (le, job))

- Uptime:
  up{job="recommendation-service"}

Use these queries in Grafana or Prometheus UI after running load tests to evaluate service behavior.
