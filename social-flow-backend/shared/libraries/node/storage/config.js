/**
 * storage/config.js
 * Environment-driven configuration with safe defaults
 */

require('dotenv').config();

const int = (v, d) => (v === undefined ? d : parseInt(v, 10));
const bool = (v, d = false) => (v === undefined ? d : String(v).toLowerCase() === 'true');

module.exports = {
  DEFAULT_PROVIDER: process.env.STORAGE_PROVIDER || 'local', // 'local' or 's3'
  LOCAL: {
    basePath: process.env.STORAGE_LOCAL_PATH || './data/storage',
  },
  S3: {
    bucket: process.env.STORAGE_S3_BUCKET || '',
    region: process.env.STORAGE_S3_REGION || 'us-east-1',
    endpoint: process.env.STORAGE_S3_ENDPOINT || '', // for S3-compatible
    accessKeyId: process.env.AWS_ACCESS_KEY_ID || '',
    secretAccessKey: process.env.AWS_SECRET_ACCESS_KEY || '',
    signatureVersion: process.env.AWS_SIGNATURE_VERSION || 'v4',
  },
  UPLOAD: {
    maxRetries: int(process.env.STORAGE_UPLOAD_MAX_RETRIES || '3', 3),
    baseBackoffMs: int(process.env.STORAGE_UPLOAD_BASE_BACKOFF_MS || '200', 200),
    multipartThresholdBytes: int(process.env.STORAGE_MULTIPART_THRESHOLD_BYTES || (5 * 1024 * 1024), 5 * 1024 * 1024), // 5MB
    multipartPartSize: int(process.env.STORAGE_MULTIPART_PART_SIZE || (5 * 1024 * 1024), 5 * 1024 * 1024), // 5MB
  },
  PRESIGN: {
    urlExpiresSec: int(process.env.STORAGE_PRESIGN_EXPIRES || '900', 900), // 15 minutes
  },
  QUOTA: {
    defaultBucketQuotaBytes: int(process.env.STORAGE_DEFAULT_BUCKET_QUOTA_BYTES || (10 * 1024 * 1024 * 1024), 10 * 1024 * 1024 * 1024), // 10GB
  },
  LOG_LEVEL: process.env.STORAGE_LOG_LEVEL || (process.env.NODE_ENV === 'production' ? 'info' : 'debug'),
};
