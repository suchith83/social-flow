/**
 * Prometheus metrics wrapper using prom-client
 * - Central registry
 * - Common metrics: http_request_duration, http_request_total, process_*, gc metrics, custom counters/gauges/histograms/summaries
 * - Helpers for instrumenting functions and middleware
 *
 * Implementation notes:
 * - Use a single Registry for the process; allow creation of separate registries if needed.
 * - Avoid metric re-registration on hot-reload by checking existence.
 */

const client = require('prom-client');
const config = require('./config');
const logger = require('./logger').base;

const defaultRegistry = new client.Registry();

// use default global metrics collection if desired
if (config.COLLECT_PROCESS_METRICS) {
  // collect default metrics (cpu, memory, event loop delay). Node.js version may limit some metrics.
  client.collectDefaultMetrics({
    register: defaultRegistry,
    labels: { service: config.SERVICE_NAME, env: config.ENV },
    // default: collect every 10s
    timeout: 10000,
  });
}

// A registry map to avoid duplicate metric creation
const metricCache = new Map();

/**
 * Safe metric creation / get-or-create helper.
 * Returns either existing metric or creates a new one.
 */
function getOrCreateMetric(key, factoryFn) {
  if (metricCache.has(key)) return metricCache.get(key);
  const metric = factoryFn();
  metricCache.set(key, metric);
  defaultRegistry.registerMetric(metric);
  return metric;
}

/**
 * Common metrics
 */
const httpRequestTotal = getOrCreateMetric('http_request_total', () =>
  new client.Counter({
    name: 'http_request_total',
    help: 'Total number of HTTP requests',
    labelNames: ['method', 'route', 'code', 'service', 'env'],
  })
);

const httpRequestDuration = getOrCreateMetric('http_request_duration_seconds', () =>
  new client.Histogram({
    name: 'http_request_duration_seconds',
    help: 'HTTP request duration in seconds',
    labelNames: ['method', 'route', 'code', 'service', 'env'],
    // buckets tuned for typical web services: from 5ms to 10s
    buckets: [0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10],
  })
);

const customCounters = {};
const customGauges = {};
const customHistograms = {};
const customSummaries = {};

/**
 * Create or get a custom counter
 */
function counter(name, help, labelNames = []) {
  const key = `counter:${name}`;
  return getOrCreateMetric(key, () =>
    new client.Counter({ name, help, labelNames: [...labelNames, 'service', 'env'] })
  );
}

/**
 * Create or get a custom gauge
 */
function gauge(name, help, labelNames = []) {
  const key = `gauge:${name}`;
  return getOrCreateMetric(key, () =>
    new client.Gauge({ name, help, labelNames: [...labelNames, 'service', 'env'] })
  );
}

/**
 * Create or get histogram
 */
function histogram(name, help, labelNames = [], buckets) {
  const key = `histogram:${name}`;
  return getOrCreateMetric(key, () =>
    new client.Histogram({ name, help, labelNames: [...labelNames, 'service', 'env'], buckets })
  );
}

/**
 * Create or get summary
 */
function summary(name, help, labelNames = []) {
  const key = `summary:${name}`;
  return getOrCreateMetric(key, () =>
    new client.Summary({ name, help, labelNames: [...labelNames, 'service', 'env'] })
  );
}

/**
 * Returns metrics exposition string (for Prometheus scrape)
 */
async function metricsAsPrometheus() {
  try {
    // Include default labels for all metrics via registry.metrics() doesn't automatically add labels,
    // so we set default labels globally:
    defaultRegistry.setDefaultLabels({ service: config.SERVICE_NAME, env: config.ENV });
    return await defaultRegistry.metrics();
  } catch (err) {
    logger.error({ err }, 'Failed to gather metrics');
    throw err;
  }
}

/**
 * Instrument promise-returning function and measure its duration.
 * options:
 *    metric: optional Histogram instance or name
 *    labels: additional labels
 */
async function instrument(nameOrMetric, labels, fn) {
  const metric = typeof nameOrMetric === 'string' ? histogram(nameOrMetric, nameOrMetric) : nameOrMetric;
  const end = metric.startTimer({ ...labels, service: config.SERVICE_NAME, env: config.ENV });
  try {
    const res = await fn();
    end({ code: '0' });
    return res;
  } catch (err) {
    end({ code: '1' });
    throw err;
  }
}

/**
 * Expose primitives and helpers
 */
module.exports = {
  client,
  Registry: client.Registry,
  defaultRegistry,
  httpRequestTotal,
  httpRequestDuration,
  counter,
  gauge,
  histogram,
  summary,
  metricsAsPrometheus,
  instrument,
};
