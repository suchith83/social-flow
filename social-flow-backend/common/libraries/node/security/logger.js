/**
 * Small logger wrapper for the security module.
 * Uses pino if available, otherwise falls back to console.
 */

let pino;
try {
  pino = require('pino');
} catch (e) {
  pino = null;
}

const config = require('./config');

if (pino) {
  const logger = pino({
    name: 'security',
    level: config.ENV === 'production' ? 'info' : 'debug',
    timestamp: pino.stdTimeFunctions.isoTime,
  });
  module.exports = logger;
} else {
  // Tiny console fallback
  const fallback = {
    info: (...a) => console.log('[security][info]', ...a),
    warn: (...a) => console.warn('[security][warn]', ...a),
    error: (...a) => console.error('[security][error]', ...a),
    debug: (...a) => console.debug('[security][debug]', ...a),
    child: () => fallback,
  };
  module.exports = fallback;
}
