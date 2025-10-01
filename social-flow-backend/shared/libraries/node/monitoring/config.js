/**
 * Centralized configuration for monitoring module.
 * Loads env via dotenv and provides typed defaults.
 */

require('dotenv').config();

const bool = (v, d = false) => (v === undefined ? d : String(v).toLowerCase() === 'true');
const int = (v, d) => (v === undefined ? d : parseInt(v, 10));

module.exports = {
  // General
  SERVICE_NAME: process.env.SERVICE_NAME || 'unknown-service',
  ENV: process.env.NODE_ENV || 'development',

  // Prometheus / metrics
  METRICS_ENABLED: bool(process.env.METRICS_ENABLED, true),
  METRICS_PORT: int(process.env.METRICS_PORT, 9464), // default Prometheus exporter port for this service
  METRICS_ROUTE: process.env.METRICS_ROUTE || '/metrics',
  PUSHGATEWAY_ENABLED: bool(process.env.PUSHGATEWAY_ENABLED, false),
  PUSHGATEWAY_URL: process.env.PUSHGATEWAY_URL || 'http://localhost:9091',
  PUSH_INTERVAL_SEC: int(process.env.PUSH_INTERVAL_SEC, 30),

  // Node process collectors config
  COLLECT_PROCESS_METRICS: bool(process.env.COLLECT_PROCESS_METRICS, true),
  COLLECT_GC_METRICS: bool(process.env.COLLECT_GC_METRICS, true),

  // Tracing
  TRACING_ENABLED: bool(process.env.TRACING_ENABLED, false),
  OTEL_EXPORTER: process.env.OTEL_EXPORTER || 'jaeger', // 'jaeger' or 'collector'
  OTEL_SERVICE_NAME: process.env.OTEL_SERVICE_NAME || (process.env.SERVICE_NAME || 'unknown-service'),
  OTEL_JAEGER_HOST: process.env.OTEL_JAEGER_HOST || 'http://localhost:14268/api/traces',
  OTEL_COLLECTOR_URL: process.env.OTEL_COLLECTOR_URL || 'http://localhost:4318/v1/traces',

  // Alerts (Slack & PagerDuty)
  ALERTS_ENABLED: bool(process.env.ALERTS_ENABLED, false),
  SLACK_WEBHOOK: process.env.SLACK_WEBHOOK || '',
  PAGERDUTY_INTEGRATION_KEY: process.env.PAGERDUTY_INTEGRATION_KEY || '',
  ALERTS_DEFAULT_CHANNEL: process.env.ALERTS_DEFAULT_CHANNEL || '#alerts',

  // Logging defaults (used by logger.js)
  LOG_LEVEL: process.env.LOG_LEVEL || (process.env.NODE_ENV === 'production' ? 'info' : 'debug'),
};
