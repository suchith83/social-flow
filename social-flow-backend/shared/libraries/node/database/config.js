/**
 * common/libraries/node/database/config.js
 * Environment-driven configuration with safe defaults.
 */

require('dotenv').config();

const DEFAULTS = {
  DB_CLIENT: process.env.DB_CLIENT || 'pg', // 'pg' or 'mysql'
  DB_HOST: process.env.DB_HOST || '127.0.0.1',
  DB_PORT: process.env.DB_PORT || (process.env.DB_CLIENT === 'mysql' ? 3306 : 5432),
  DB_USER: process.env.DB_USER || 'postgres',
  DB_PASSWORD: process.env.DB_PASSWORD || '',
  DB_DATABASE: process.env.DB_DATABASE || 'app_db',
  DB_POOL_MIN: parseInt(process.env.DB_POOL_MIN || '2', 10),
  DB_POOL_MAX: parseInt(process.env.DB_POOL_MAX || '10', 10),
  DB_IDLE_TIMEOUT: parseInt(process.env.DB_IDLE_TIMEOUT || '30000', 10),
  DB_CONNECTION_TIMEOUT: parseInt(process.env.DB_CONNECTION_TIMEOUT || '2000', 10),
  DB_QUERY_RETRY: parseInt(process.env.DB_QUERY_RETRY || '3', 10),
  DB_QUERY_RETRY_BACKOFF_MS: parseInt(process.env.DB_QUERY_RETRY_BACKOFF_MS || '100', 10),
  MIGRATIONS_DIR: process.env.MIGRATIONS_DIR || './migrations',
  LOG_QUERIES: process.env.LOG_QUERIES === 'true' || false,
};

module.exports = {
  ...DEFAULTS,
};
