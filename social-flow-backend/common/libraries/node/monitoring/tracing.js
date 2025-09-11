/**
 * OpenTelemetry tracing bootstrapper
 *
 * - Boots a NodeSDK when TRACING_ENABLED is true
 * - Configurable exporter: Jaeger or OTLP/Collector
 * - Auto-instrumentations for HTTP/Express, gRPC, pg, mysql, etc.
 *
 * Notes:
 * - This file attempts safe initialization and idempotence (avoid double-init).
 * - For performance-critical apps you may want to customize sampling, processor, and exporter options.
 */

const { trace, context } = require('@opentelemetry/api');
const config = require('./config');
const logger = require('./logger').base;

let sdkInstance = null;

async function initTracing(opts = {}) {
  if (!config.TRACING_ENABLED) {
    logger.debug('Tracing disabled by config');
    return null;
  }
  if (sdkInstance) {
    logger.warn('Tracing already initialized');
    return sdkInstance;
  }

  // Dynamically import to avoid requiring heavy deps when disabled
  try {
    const { NodeSDK } = require('@opentelemetry/sdk-node');
    const { getNodeAutoInstrumentations } = require('@opentelemetry/auto-instrumentations-node');

    let exporter;
    if (config.OTEL_EXPORTER === 'jaeger') {
      const { JaegerExporter } = require('@opentelemetry/exporter-jaeger');
      exporter = new JaegerExporter({ endpoint: config.OTEL_JAEGER_HOST });
    } else {
      const { OTLPTraceExporter } = require('@opentelemetry/exporter-collector');
      exporter = new OTLPTraceExporter({ url: config.OTEL_COLLECTOR_URL });
    }

    sdkInstance = new NodeSDK({
      traceExporter: exporter,
      instrumentations: [getNodeAutoInstrumentations()],
      serviceName: config.OTEL_SERVICE_NAME,
    });

    await sdkInstance.start();
    logger.info('OpenTelemetry tracing started', { exporter: config.OTEL_EXPORTER });
    return sdkInstance;
  } catch (err) {
    logger.error({ err }, 'Failed to initialize tracing');
    throw err;
  }
}

async function shutdownTracing() {
  if (!sdkInstance) return;
  try {
    await sdkInstance.shutdown();
    logger.info('Tracing shutdown complete');
    sdkInstance = null;
  } catch (err) {
    logger.warn({ err }, 'Failed to shut down tracing gracefully');
  }
}

module.exports = {
  initTracing,
  shutdownTracing,
  getTracer: (name = config.SERVICE_NAME) => trace.getTracer(name),
};
