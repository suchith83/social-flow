/**
 * Health checks module
 *
 * - Exposes a health check interface that can be used by the HTTP server health endpoint.
 * - Supports registering custom async checks (databases, queues, external services)
 * - Aggregates statuses and returns structured result.
 */

const logger = require('./logger').base;
const metrics = require('./metrics');

const checks = new Map();

/**
 * Register a health check
 * @param {string} name
 * @param {function(): Promise<{ healthy: boolean, info?: any }>} fn
 */
function registerCheck(name, fn) {
  if (typeof fn !== 'function') {
    throw new Error('Health check must be a function');
  }
  checks.set(name, fn);
}

/**
 * Remove a health check
 */
function unregisterCheck(name) {
  checks.delete(name);
}

/**
 * Run all checks with timeout and return aggregated result
 * @param {number} timeoutMs
 */
async function runChecks(timeoutMs = 2000) {
  const results = {};
  const promises = [];

  for (const [name, fn] of checks.entries()) {
    const p = Promise.race([
      (async () => {
        try {
          const r = await fn();
          results[name] = { healthy: !!(r && r.healthy), info: r && r.info };
        } catch (err) {
          results[name] = { healthy: false, error: String(err) };
        }
      })(),
      new Promise((res) =>
        setTimeout(() => {
          results[name] = { healthy: false, error: 'timeout' };
          res();
        }, timeoutMs)
      ),
    ]);
    promises.push(p);
  }

  await Promise.all(promises);

  const healthy = Object.values(results).every((r) => r && r.healthy);
  // Optional: emit metrics for health check result
  const gauge = metrics.gauge('service_health_status', 'Service aggregated health (1 healthy, 0 unhealthy)', ['check']);
  for (const [name, res] of Object.entries(results)) {
    gauge.set({ check: name, service: require('./config').SERVICE_NAME, env: require('./config').ENV }, res.healthy ? 1 : 0);
  }

  return { healthy, checks: results, ts: new Date().toISOString() };
}

module.exports = {
  registerCheck,
  unregisterCheck,
  runChecks,
};
