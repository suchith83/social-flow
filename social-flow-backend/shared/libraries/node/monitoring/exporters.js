/**
 * Exporter helpers:
 * - Prometheus HTTP endpoint (exposes metrics route)
 * - Pushgateway pusher (periodic push)
 * - Optional convenience functions to start/stop exporters
 */

const express = require('express');
const fetch = require('node-fetch'); // node-fetch v2 style; alternatively use axios
const config = require('./config');
const metrics = require('./metrics');
const collectors = require('./collectors');
const logger = require('./logger').base;

let pushIntervalHandle = null;
let appServer = null;

/**
 * Start an express server that exposes /metrics and simple /health endpoints
 * Returns the http server instance.
 */
function startMetricsServer({ port = config.METRICS_PORT, route = config.METRICS_ROUTE, healthRoute = '/health' } = {}) {
  if (!config.METRICS_ENABLED) {
    logger.debug('Metrics server disabled by config');
    return null;
  }

  const app = express();

  app.get(healthRoute, async (req, res) => {
    res.json({ service: config.SERVICE_NAME, env: config.ENV, status: 'ok', ts: new Date().toISOString() });
  });

  app.get(route, async (req, res) => {
    try {
      const body = await metrics.metricsAsPrometheus();
      res.set('Content-Type', metrics.client.register.contentType);
      res.send(body);
    } catch (err) {
      logger.error({ err }, 'Failed to expose metrics');
      res.status(500).send('metrics error');
    }
  });

  appServer = app.listen(port, () => {
    logger.info({ port, route }, 'Metrics endpoint listening');
  });

  return appServer;
}

/**
 * Push metrics to Prometheus Pushgateway periodically.
 * Uses Prometheus text format and HTTP PUT.
 */
async function startPushgatewayPush({ pushgatewayUrl = config.PUSHGATEWAY_URL, intervalSec = config.PUSH_INTERVAL_SEC } = {}) {
  if (!config.PUSHGATEWAY_ENABLED) {
    logger.debug('Pushgateway disabled by config');
    return;
  }
  if (pushIntervalHandle) {
    logger.warn('Pushgateway already running');
    return;
  }

  async function pushOnce() {
    try {
      const body = await metrics.metricsAsPrometheus();
      const url = `${pushgatewayUrl}/metrics/job/${encodeURIComponent(config.SERVICE_NAME)}`;
      const resp = await fetch(url, {
        method: 'PUT',
        body,
        headers: { 'Content-Type': metrics.client.register.contentType },
        timeout: 5000,
      });
      if (!resp.ok) {
        logger.warn({ status: resp.status, statusText: resp.statusText }, 'Pushgateway push non-ok');
      } else {
        logger.debug('Pushed metrics to pushgateway');
      }
    } catch (err) {
      logger.warn({ err }, 'Failed to push metrics to pushgateway');
    }
  }

  // initial push and schedule
  await pushOnce();
  pushIntervalHandle = setInterval(pushOnce, intervalSec * 1000);
  pushIntervalHandle.unref();
}

/**
 * stop pushgateway pushes
 */
function stopPushgatewayPush() {
  if (pushIntervalHandle) {
    clearInterval(pushIntervalHandle);
    pushIntervalHandle = null;
  }
}

/**
 * stop metrics server
 */
function stopMetricsServer() {
  if (appServer) {
    appServer.close(() => logger.info('Metrics server stopped'));
    appServer = null;
  }
}

module.exports = {
  startMetricsServer,
  stopMetricsServer,
  startPushgatewayPush,
  stopPushgatewayPush,
};
