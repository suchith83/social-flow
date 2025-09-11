/**
 * csrf.js
 *
 * Lightweight double-submit-cookie CSRF protection:
 * - generates a secure token and sets cookie
 * - validates token from header or request body
 *
 * Notes:
 * - This is intentionally simple and works well for single-origin browser apps.
 * - For cross-site scenarios or high security use stateful tokens stored server-side.
 */

const crypto = require('crypto');
const config = require('./config');
const { timingSafeEqual } = require('./crypto');
const logger = require('./logger');

function generateCsrfToken() {
  return crypto.randomBytes(config.CSRF.tokenLengthBytes).toString('base64');
}

/**
 * Middleware to set CSRF cookie on GET/HEAD requests if not present
 */
function csrfCookieSetter(opts = {}) {
  const cookieName = opts.cookieName || config.CSRF.cookieName;
  const cookieOpts = { httpOnly: false, sameSite: 'lax', secure: config.ENV === 'production', path: '/', maxAge: 60 * 60 * 24 }; // 1 day
  return (req, res, next) => {
    try {
      if ((req.method === 'GET' || req.method === 'HEAD') && !req.cookies?.[cookieName]) {
        const token = generateCsrfToken();
        res.cookie(cookieName, token, cookieOpts);
      }
    } catch (err) {
      logger.warn({ err }, 'csrf cookie setter error');
    }
    next();
  };
}

/**
 * Middleware to validate CSRF for non-safe methods
 * - validates header or body token equals cookie token using constant-time compare
 */
function csrfValidator(opts = {}) {
  const cookieName = opts.cookieName || config.CSRF.cookieName;
  const headerName = opts.headerName || config.CSRF.headerName;
  return (req, res, next) => {
    const safeMethods = ['GET', 'HEAD', 'OPTIONS'];
    if (safeMethods.includes(req.method)) return next();
    try {
      const cookieToken = req.cookies?.[cookieName];
      const headerToken = req.get(headerName) || req.body?.csrfToken || req.body?.csrf;
      if (!cookieToken || !headerToken) {
        res.status(403).send('CSRF token missing');
        return;
      }
      if (!timingSafeEqual(Buffer.from(cookieToken, 'utf8'), Buffer.from(headerToken, 'utf8'))) {
        res.status(403).send('CSRF validation failed');
        return;
      }
    } catch (err) {
      logger.warn({ err }, 'csrf validation error');
      res.status(403).send('CSRF validation failed');
      return;
    }
    return next();
  };
}

module.exports = {
  csrfCookieSetter,
  csrfValidator,
  generateCsrfToken,
};
