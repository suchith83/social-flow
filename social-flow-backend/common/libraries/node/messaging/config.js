/**
 * common/libraries/node/messaging/config.js
 * Environment-driven configuration with safe defaults.
 */

require('dotenv').config();

module.exports = {
  BROKER: process.env.MSG_BROKER || 'kafka', // kafka | rabbit
  KAFKA: {
    brokers: (process.env.KAFKA_BROKERS || 'localhost:9092').split(','),
    clientId: process.env.KAFKA_CLIENT_ID || 'app-client',
    ssl: process.env.KAFKA_SSL === 'true' || false,
    sasl: process.env.KAFKA_SASL ? JSON.parse(process.env.KAFKA_SASL) : null,
    connectionTimeout: parseInt(process.env.KAFKA_CONNECTION_TIMEOUT || '3000', 10),
  },
  RABBIT: {
    url: process.env.RABBIT_URL || 'amqp://localhost:5672',
    prefetch: parseInt(process.env.RABBIT_PREFETCH || '20', 10),
  },
  PRODUCER: {
    defaultRetries: parseInt(process.env.MSG_PRODUCER_RETRIES || '5', 10),
    baseBackoffMs: parseInt(process.env.MSG_PRODUCER_BASE_BACKOFF_MS || '100', 10),
    batchMaxSize: parseInt(process.env.MSG_PRODUCER_BATCH_MAX_SIZE || '100', 10),
    batchMaxTimeMs: parseInt(process.env.MSG_PRODUCER_BATCH_MAX_TIME_MS || '200', 10),
    idempotent: process.env.MSG_PRODUCER_IDEMPOTENT === 'true' || false,
  },
  CONSUMER: {
    autoAck: process.env.MSG_CONSUMER_AUTO_ACK === 'true' || false,
    concurrency: parseInt(process.env.MSG_CONSUMER_CONCURRENCY || '5', 10),
    maxRetries: parseInt(process.env.MSG_CONSUMER_MAX_RETRIES || '5', 10),
    deadLetterSuffix: process.env.MSG_DEAD_LETTER_SUFFIX || '.dlq',
  },
  SCHEMA: {
    strict: process.env.MSG_SCHEMA_STRICT === 'true' || false,
  },
};
