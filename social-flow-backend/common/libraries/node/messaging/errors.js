/**
 * common/libraries/node/messaging/errors.js
 * Custom error types for messaging operations.
 */

class MessagingError extends Error {
  constructor(message, meta = {}) {
    super(message);
    this.name = 'MessagingError';
    this.meta = meta;
  }
}

class BrokerError extends MessagingError {
  constructor(message, meta = {}) {
    super(message, meta);
    this.name = 'BrokerError';
  }
}

class ProduceError extends MessagingError {
  constructor(message, meta = {}) {
    super(message, meta);
    this.name = 'ProduceError';
  }
}

class ConsumeError extends MessagingError {
  constructor(message, meta = {}) {
    super(message, meta);
    this.name = 'ConsumeError';
  }
}

class SchemaValidationError extends MessagingError {
  constructor(message, errors = [], meta = {}) {
    super(message, meta);
    this.name = 'SchemaValidationError';
    this.validationErrors = errors;
  }
}

module.exports = {
  MessagingError,
  BrokerError,
  ProduceError,
  ConsumeError,
  SchemaValidationError,
};
