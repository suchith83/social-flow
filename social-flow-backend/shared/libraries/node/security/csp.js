/**
 * csp.js
 *
 * Content Security Policy builder helper.
 * Produces a middleware that sets a strict CSP header with safe defaults.
 *
 * Accepts:
 *  { directives: { defaultSrc: ["'self'"], scriptSrc: [...], styleSrc: [...] }, reportUri: '' }
 *
 * The builder also supports generating nonces for inline scripts/styles (returns req.cspNonce)
 */

const crypto = require('crypto');

const config = require('./config');

function generateNonce() {
  return crypto.randomBytes(16).toString('base64');
}

/**
 * canonicalizeDirective: turns arrays into space-joined string
 */
function canonicalizeDirective(arr) {
  if (!arr) return '';
  return arr.join(' ');
}

function buildHeader(directives = {}, extras = {}) {
  const defaultDirectives = {
    "default-src": ["'self'"],
    "script-src": ["'self'"],
    "style-src": ["'self'"],
    "img-src": ["'self'", 'data:'],
    "connect-src": ["'self'"],
    "font-src": ["'self'", 'data:'],
    "object-src": ["'none'"],
    "base-uri": ["'self'"],
    "frame-ancestors": [config.HEADERS.frameAncestors || "'none'"],
  };

  const merged = { ...defaultDirectives, ...directives };
  const parts = [];
  for (const [k, v] of Object.entries(merged)) {
    parts.push(`${k} ${canonicalizeDirective(v)}`);
  }
  if (extras.reportUri) parts.push(`report-uri ${extras.reportUri}`);
  return parts.join('; ');
}

/**
 * build middleware
 * options:
 *  - directives: object
 *  - reportUri: string
 *  - withNonce: boolean
 */
function build(options = {}) {
  const { directives = {}, reportUri, withNonce = true } = options;
  const headerBase = buildHeader(directives, { reportUri });
  return (req, res, next) => {
    try {
      if (withNonce) {
        const nonce = generateNonce();
        req.cspNonce = nonce;
        // add nonce to script-src and style-src
        const scriptPart = `script-src 'self' 'nonce-${nonce}'`;
        const stylePart = `style-src 'self' 'nonce-${nonce}'`;
        // prefer the headerBase but replace script/style parts using simple regex
        let header = headerBase;
        header = header.replace(/script-src [^;]+/, scriptPart);
        header = header.replace(/style-src [^;]+/, stylePart);
        res.setHeader('Content-Security-Policy', header);
      } else {
        res.setHeader('Content-Security-Policy', headerBase);
      }
    } catch (err) {
      // fail-open: log and continue
      console.warn('CSP build error', err);
    }
    next();
  };
}

module.exports = {
  build,
  generateNonce,
  buildHeader,
};
