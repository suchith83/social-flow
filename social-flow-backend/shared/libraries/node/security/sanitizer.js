/**
 * sanitizer.js
 *
 * Utilities for sanitizing user input to prevent XSS/HTML injection and other attacks.
 * Uses DOMPurify via jsdom when available (server-side sanitization).
 *
 * Exports:
 *  - sanitizeHtml(html, opts)
 *  - sanitizeObject(obj, options) -> recursively sanitize string fields
 *
 * If jsdom/DOMPurify isn't installed, falls back to a simple whitelist stripper (less safe).
 */

const config = require('./config');
const logger = require('./logger');

let DOMPurify, JSDOM;
try {
  JSDOM = require('jsdom').JSDOM;
  DOMPurify = require('dompurify')(new JSDOM('').window);
} catch (e) {
  DOMPurify = null;
}

const { allowedTags, allowedAttributes } = config.SANITIZER;

/**
 * sanitizeHtml
 */
function sanitizeHtml(input, opts = {}) {
  if (!input) return input;
  if (DOMPurify) {
    const clean = DOMPurify.sanitize(input, {
      ALLOWED_TAGS: opts.allowedTags || allowedTags,
      ALLOWED_ATTR: opts.allowedAttributes || allowedAttributes,
    });
    return clean;
  }
  // crude fallback: strip tags except a small whitelist
  try {
    return String(input)
      .replace(/<\/?[^>]+(>|$)/g, '') // remove tags
      .replace(/[<>]/g, '');
  } catch (e) {
    logger.warn({ e }, 'sanitizer fallback failed');
    return '';
  }
}

/**
 * Recursively sanitize object string values
 */
function sanitizeObject(obj, opts = {}) {
  if (obj === null || obj === undefined) return obj;
  if (typeof obj === 'string') return sanitizeHtml(obj, opts);
  if (Array.isArray(obj)) return obj.map((v) => sanitizeObject(v, opts));
  if (typeof obj === 'object') {
    const res = {};
    for (const k of Object.keys(obj)) {
      res[k] = sanitizeObject(obj[k], opts);
    }
    return res;
  }
  return obj;
}

module.exports = {
  sanitizeHtml,
  sanitizeObject,
};
