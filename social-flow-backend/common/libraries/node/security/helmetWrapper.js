/**
 * helmetWrapper.js
 *
 * Provides a wrapper around helmet and adds configuration for secure headers.
 * Also integrates CSP via the csp module and exposes an Express middleware factory.
 *
 * Usage:
 *   const helmetMiddleware = helmetWrapper({ frameAncestors: "'self'" });
 *   app.use(helmetMiddleware);
 */

const helmet = require('helmet');
const config = require('./config');
const csp = require('./csp');
const logger = require('./logger');

function defaultOptions() {
  return {
    contentSecurityPolicy: false, // we will provide CSP separately for more control
    frameguard: { action: 'deny' },
    hsts: { maxAge: config.HEADERS.hstsMaxAge, includeSubDomains: true, preload: true },
    referrerPolicy: { policy: config.HEADERS.referrerPolicy },
    xssFilter: true,
    noSniff: true,
    dnsPrefetchControl: { allow: false },
    expectCt: false,
  };
}

/**
 * createHelmetMiddleware
 * @param {Object} opts - option overrides
 * @param {Object|false} cspOptions - if object, will install CSP middleware via csp.build()
 */
function createHelmetMiddleware(opts = {}, cspOptions = {}) {
  const merged = { ...defaultOptions(), ...opts };
  const base = helmet(merged);

  const cspMiddleware = cspOptions === false ? (req, res, next) => next() : csp.build(cspOptions);

  return (req, res, next) => {
    try {
      base(req, res, () => {
        cspMiddleware(req, res, next);
      });
    } catch (err) {
      logger.warn({ err }, 'helmet middleware error');
      next();
    }
  };
}

module.exports = {
  createHelmetMiddleware,
};
