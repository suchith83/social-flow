/**
 * Express middleware to instrument HTTP requests:
 * - increments request counter
 * - measures request duration
 * - captures status codes and route label
 *
 * Use:
 *   const { httpMiddleware } = require('common/libraries/node/monitoring/middleware');
 *   app.use(httpMiddleware());
 *
 * NOTE: For route label we try to use req.route.path if available (after router),
 * but this middleware should be added at top-level; if you want route based labels,
 * consider mounting instrumentation after route definitions or use a wrapper per-route.
 */

const metrics = require('./metrics');
const logger = require('./logger').base;

function httpMiddleware(opts = {}) {
  const reqCounter = metrics.httpRequestTotal;
  const reqDuration = metrics.httpRequestDuration;

  return (req, res, next) => {
    const startHr = process.hrtime.bigint();
    // compute route label fallback
    const route = (req.route && req.route.path) || req.path || 'unknown';
    const method = req.method;

    // When response finishes, record metrics
    res.on('finish', () => {
      const statusCode = res.statusCode;
      const durNs = Number(process.hrtime.bigint() - startHr);
      const durSec = durNs / 1e9;
      try {
        reqCounter.inc({ method, route, code: String(statusCode), service: require('./config').SERVICE_NAME, env: require('./config').ENV }, 1);
        reqDuration.observe({ method, route, code: String(statusCode), service: require('./config').SERVICE_NAME, env: require('./config').ENV }, durSec);
      } catch (err) {
        logger.debug({ err }, 'Failed to record http metrics');
      }
    });

    // pass-through
    next();
  };
}

module.exports = {
  httpMiddleware,
};
