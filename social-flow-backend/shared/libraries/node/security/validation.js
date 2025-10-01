/**
 * validation.js
 *
 * Small validation helpers wrapping AJV for JSON-schema validation and additional utilities.
 *
 * - createValidator(schema) -> validate(data) throws on invalid
 * - validateAndSanitize(schema, data, sanitizer) -> validates then sanitizes string fields
 */

let Ajv;
try {
  Ajv = require('ajv').default;
} catch (e) {
  Ajv = null;
}
const addFormats = (() => { try { return require('ajv-formats').default; } catch (e) { return null; } })();
const logger = require('./logger');

function createValidator(schema, opts = {}) {
  if (!Ajv) {
    throw new Error('AJV library is required for schema validation (npm i ajv ajv-formats)');
  }
  const ajv = new Ajv({ allErrors: true, ...opts });
  if (addFormats) addFormats(ajv);
  const validate = ajv.compile(schema);
  return (data) => {
    const ok = validate(data);
    if (!ok) {
      const err = new Error('Validation failed');
      err.details = validate.errors;
      throw err;
    }
    return true;
  };
}

/**
 * validateAndSanitize:
 * - validate via AJV
 * - sanitize string fields via sanitizer.sanitizeObject
 */
function validateAndSanitize(schema, data, sanitizer) {
  const validate = createValidator(schema);
  validate(data); // throws if invalid
  if (sanitizer && typeof sanitizer.sanitizeObject === 'function') {
    return sanitizer.sanitizeObject(data);
  }
  return data;
}

module.exports = {
  createValidator,
  validateAndSanitize,
};
