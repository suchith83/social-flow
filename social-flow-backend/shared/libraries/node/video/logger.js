/**
 * lightweight logger wrapper (pino if available, otherwise console)
 */

const config = require('./config');

let pino;
try {
  pino = require('pino');
} catch (e) {
  pino = null;
}

if (pino) {
  module.exports = pino({ level: config.LOG_LEVEL, name: 'video-lib', timestamp: pino.stdTimeFunctions.isoTime });
} else {
  const fallback = {
    debug: (...a) => console.debug('[video]', ...a),
    info: (...a) => console.info('[video]', ...a),
    warn: (...a) => console.warn('[video]', ...a),
    error: (...a) => console.error('[video]', ...a),
    child: () => fallback,
  };
  module.exports = fallback;
}
