/**
 * Custom error classes used by video library
 */

class VideoError extends Error {
  constructor(message, meta = {}) {
    super(message);
    this.name = 'VideoError';
    this.meta = meta;
  }
}

class TranscodeError extends VideoError {
  constructor(message, meta = {}) {
    super(message, meta);
    this.name = 'TranscodeError';
  }
}

class ThumbnailError extends VideoError {
  constructor(message, meta = {}) {
    super(message, meta);
    this.name = 'ThumbnailError';
  }
}

class MetadataError extends VideoError {
  constructor(message, meta = {}) {
    super(message, meta);
    this.name = 'MetadataError';
  }
}

module.exports = {
  VideoError,
  TranscodeError,
  ThumbnailError,
  MetadataError,
};
