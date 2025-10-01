/**
 * common/libraries/node/database/errors.js
 * Custom error types to wrap DB errors and enable clearer retry/handling logic.
 */

class DBError extends Error {
  constructor(message, meta = {}) {
    super(message);
    this.name = 'DBError';
    this.meta = meta;
  }
}

class ConnectionError extends DBError {
  constructor(message, meta = {}) {
    super(message, meta);
    this.name = 'ConnectionError';
  }
}

class QueryError extends DBError {
  constructor(message, meta = {}) {
    super(message, meta);
    this.name = 'QueryError';
  }
}

class TransactionError extends DBError {
  constructor(message, meta = {}) {
    super(message, meta);
    this.name = 'TransactionError';
  }
}

module.exports = {
  DBError,
  ConnectionError,
  QueryError,
  TransactionError,
};
