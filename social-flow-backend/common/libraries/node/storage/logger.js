/**
 * storage/logger.js
 * Lightweight pino-based logger wrapper; falls back to console
 */

let pino;
try {
  pino = require('pino');
} catch (e) {
  pino = null;
}

const config = require('./config');
if (pino) {
  module.exports = pino({ level: config.LOG_LEVEL, name: 'storage' });
} else {
  const fallback = {
    debug: (...a) => console.debug('[storage][debug]', ...a),
    info: (...a) => console.info('[storage][info]', ...a),
    warn: (...a) => console.warn('[storage][warn]', ...a),
    error: (...a) => console.error('[storage][error]', ...a),
    child: () => fallback,
  };
  module.exports = fallback;
}
