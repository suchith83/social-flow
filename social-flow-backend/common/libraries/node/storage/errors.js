/**
 * storage/errors.js
 * Custom error types to clarify failures
 */

class StorageError extends Error {
  constructor(message, meta = {}) {
    super(message);
    this.name = 'StorageError';
    this.meta = meta;
  }
}

class NotFoundError extends StorageError {
  constructor(message = 'Not found', meta = {}) {
    super(message, meta);
    this.name = 'NotFoundError';
  }
}

class UploadError extends StorageError {
  constructor(message = 'Upload failed', meta = {}) {
    super(message, meta);
    this.name = 'UploadError';
  }
}

class PresignError extends StorageError {
  constructor(message = 'Presign failed', meta = {}) {
    super(message, meta);
    this.name = 'PresignError';
  }
}

module.exports = {
  StorageError,
  NotFoundError,
  UploadError,
  PresignError,
};
