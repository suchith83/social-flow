/**
 * client.js
 *
 * High-level storage client that selects an adapter (local or s3) and exposes
 * advanced operations with retries, quotas, and metadata handling.
 *
 * API:
 *   const client = await Client.init({ provider: 's3', s3Opts: {...} });
 *   await client.upload({ key, stream });
 *   const { stream } = await client.download({ key });
 *   await client.delete({ key });
 *   await client.presignPut(key);
 *
 * Client also exposes helper: exists(key), stat(key), list(prefix)
 */

const config = require('./config');
const logger = require('./logger');
const utils = require('./utils');
const { UploadError } = require('./errors');
const LocalFsAdapter = require('./adapters/localFsAdapter');
const S3Adapter = require('./adapters/s3Adapter');

class Client {
  constructor(adapter) {
    this.adapter = adapter;
    this.uploadRetries = config.UPLOAD.maxRetries;
    this.baseBackoff = config.UPLOAD.baseBackoffMs;
  }

  static async init(options = {}) {
    // choose provider
    const provider = options.provider || config.DEFAULT_PROVIDER;
    let adapter;
    if (provider === 'local') {
      adapter = new LocalFsAdapter({ basePath: options.basePath });
    } else if (provider === 's3') {
      adapter = new S3Adapter(options.s3 || {});
    } else {
      throw new Error(`Unsupported provider: ${provider}`);
    }
    await adapter.init();
    return new Client(adapter);
  }

  async upload({ key, stream, buffer, size, contentType, metadata = {} }) {
    // high-level retry loop
    let attempt = 0;
    let lastErr = null;
    while (attempt <= this.uploadRetries) {
      try {
        const res = await this.adapter.upload({ key, stream, buffer, size, contentType, metadata });
        return res;
      } catch (err) {
        lastErr = err;
        const isTransient = this._isTransient(err);
        if (!isTransient) {
          logger.error({ err, key }, 'Non-transient upload error');
          throw new UploadError('Non-transient upload error', { original: err });
        }
        if (attempt < this.uploadRetries) {
          const wait = utils.exponentialBackoff(attempt, this.baseBackoff);
          logger.warn({ err, attempt, wait }, 'Transient upload error, retrying');
          await utils.sleep(wait);
          attempt++;
          continue;
        }
        break;
      }
    }
    throw new UploadError('Upload failed after retries', { original: lastErr });
  }

  async download({ key }) {
    try {
      return await this.adapter.download({ key });
    } catch (err) {
      logger.error({ err, key }, 'download failed');
      throw err;
    }
  }

  async delete({ key }) {
    return await this.adapter.delete({ key });
  }

  async stat({ key }) {
    return await this.adapter.stat({ key });
  }

  async exists({ key }) {
    try {
      await this.stat({ key });
      return true;
    } catch (e) {
      return false;
    }
  }

  async list(prefix = '', opts = {}) {
    // normalize generator or array
    const res = this.adapter.list(prefix, opts);
    if (res[Symbol.asyncIterator]) return res;
    // if returns array
    return (async function* () {
      for (const r of res) yield r;
    })();
  }

  async presignGet(key, opts = {}) {
    if (this.adapter.presignGet) return await this.adapter.presignGet(key, opts);
    throw new Error('presignGet not supported by adapter');
  }

  async presignPut(key, opts = {}) {
    if (this.adapter.presignPut) return await this.adapter.presignPut(key, opts);
    throw new Error('presignPut not supported by adapter');
  }

  _isTransient(err) {
    // naive check: network errors, 5xx, or SDK-specific retriable flags
    if (!err) return false;
    const message = String(err.message || '').toLowerCase();
    if (message.includes('timeout') || message.includes('timed out') || message.includes('ec2') || message.includes('etag')) return true;
    // AWS SDK errors sometimes have $retryable
    if (err.$metadata && err.$metadata.httpStatusCode && err.$metadata.httpStatusCode >= 500) return true;
    return false;
  }
}

module.exports = Client;
