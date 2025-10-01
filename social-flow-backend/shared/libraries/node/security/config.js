/**
 * Centralized configuration for security library.
 * Pulls from environment variables and provides sensible defaults.
 */

require('dotenv').config();

const bool = (v, d = false) => (v === undefined ? d : String(v).toLowerCase() === 'true');
const int = (v, d) => (v === undefined ? d : parseInt(v, 10));

module.exports = {
  ENV: process.env.NODE_ENV || 'development',

  // KMS / key rotation
  KMS: {
    adapter: process.env.KMS_ADAPTER || 'file', // 'file' or 'aws-kms' (adapter pluggable)
    fileKeyPath: process.env.KMS_FILE_KEY_PATH || './secrets/keys.json',
    rotationIntervalDays: int(process.env.KMS_ROTATION_INTERVAL_DAYS || '90', 90),
  },

  // crypto
  CRYPTO: {
    masterKeyEnv: process.env.SECURITY_MASTER_KEY || '', // optional env master key
    aesGcmKeyLength: int(process.env.AES_GCM_KEY_LENGTH || '32', 32), // bytes
    hkdfSalt: process.env.HKDF_SALT || 'security-default-salt',
  },

  // CSRF
  CSRF: {
    cookieName: process.env.CSRF_COOKIE_NAME || 'csrf_token',
    headerName: process.env.CSRF_HEADER_NAME || 'x-csrf-token',
    tokenLengthBytes: int(process.env.CSRF_TOKEN_LENGTH_BYTES || '32', 32),
  },

  // Rate limiter
  RATE_LIMITER: {
    inMemory: bool(process.env.RATE_LIMITER_INMEMORY || true),
    redisUrl: process.env.RATE_LIMITER_REDIS_URL || '',
    defaultWindowMs: int(process.env.RATE_LIMITER_WINDOW_MS || '60000', 60000),
    defaultMax: int(process.env.RATE_LIMITER_MAX || '100', 100),
    defaultBurst: int(process.env.RATE_LIMITER_BURST || '20', 20),
  },

  // Helmet / secure headers defaults
  HEADERS: {
    hstsMaxAge: int(process.env.HEADERS_HSTS_MAX_AGE || '15552000', 15552000), // 180 days
    referrerPolicy: process.env.HEADERS_REFERRER_POLICY || 'no-referrer',
    frameAncestors: process.env.HEADERS_FRAME_ANCESTORS || "'none'",
  },

  // Auditing
  AUDIT: {
    logFile: process.env.AUDIT_LOG_FILE || './logs/security-audit.log',
    enabled: bool(process.env.AUDIT_ENABLED || true),
  },

  // Sanitization
  SANITIZER: {
    allowedTags: (process.env.SANITIZER_ALLOWED_TAGS || 'b,i,em,strong,a,span,ul,ol,li').split(','),
    allowedAttributes: ['href', 'title', 'rel', 'target', 'class'],
  },
};
