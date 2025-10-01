/**
 * common/libraries/node/messaging/schema.js
 *
 * Wrapper around AJV for message schema validation and registry.
 * Allows registering schemas by name and validating messages prior to produce/after consume.
 */

const Ajv = require('ajv').default;
const addFormats = require('ajv-formats').default;
const Config = require('./config');
const { SchemaValidationError } = require('./errors');

class SchemaRegistry {
  constructor(opts = {}) {
    this.ajv = new Ajv({ allErrors: true, strict: Config.SCHEMA.strict });
    addFormats(this.ajv);
    this.schemas = new Map();
  }

  register(name, schema) {
    const validate = this.ajv.compile(schema);
    this.schemas.set(name, { schema, validate });
    return validate;
  }

  validate(name, payload) {
    if (!this.schemas.has(name)) {
      throw new Error(`Schema ${name} not found`);
    }
    const { validate } = this.schemas.get(name);
    const ok = validate(payload);
    if (!ok) {
      throw new SchemaValidationError(`Validation failed for ${name}`, validate.errors || []);
    }
    return true;
  }

  getSchema(name) {
    return this.schemas.get(name)?.schema ?? null;
  }
}

module.exports = new SchemaRegistry();
