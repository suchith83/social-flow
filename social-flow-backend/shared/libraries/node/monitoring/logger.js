/**
 * Lightweight wrapper around pino logger for consistent logs across services.
 * - Provides structured JSON logs
 * - Exposes child logger creation helpers for correlation IDs and request contexts
 */

const pino = require('pino');
const config = require('./config');

const baseLogger = pino({
  level: config.LOG_LEVEL,
  name: config.SERVICE_NAME,
  timestamp: pino.stdTimeFunctions.isoTime,
  safe: true, // ensure unsafe circular objects don't crash
});

function createLogger(bindings = {}) {
  return baseLogger.child(bindings);
}

module.exports = {
  base: baseLogger,
  createLogger,
};
