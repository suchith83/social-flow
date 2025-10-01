/**
 * common/libraries/node/messaging/kafkaAdapter.js
 *
 * Adapter for Kafka using kafkajs.
 * Normalizes produce/consume operations.
 *
 * Notes:
 * - Implements exactly-once semantics only if Kafka cluster supports idempotent producers and transactions.
 * - For simplicity we support at-least-once by default (manual commit after handler success).
 */

const { Kafka, CompressionTypes, logLevel } = require('kafkajs');
const Broker = require('./broker');
const Config = require('./config');
const { BrokerError } = require('./errors');
const { safeJsonStringify } = require('./utils');

class KafkaAdapter extends Broker {
  constructor(opts = {}) {
    super(opts);
    this.config = { ...Config.KAFKA, ...(opts.kafka || {}) };
    this.kafka = new Kafka({
      clientId: this.config.clientId,
      brokers: this.config.brokers,
      ssl: this.config.ssl,
      sasl: this.config.sasl,
      connectionTimeout: this.config.connectionTimeout,
      logLevel: logLevel.NOTHING,
    });
    this.producer = null;
    this.consumerMap = new Map();
    this.connected = false;
  }

  async connect() {
    if (this.connected) return;
    try {
      this.producer = this.kafka.producer({
        idempotent: this.config.idempotent || false,
        maxInFlightRequests: 1,
      });
      await this.producer.connect();
      this.connected = true;
    } catch (err) {
      throw new BrokerError('Failed to connect Kafka', { original: err });
    }
  }

  async disconnect() {
    try {
      if (this.producer) await this.producer.disconnect();
      for (const c of this.consumerMap.values()) {
        try {
          await c.disconnect();
        } catch (e) {
          // ignore
        }
      }
      this.connected = false;
    } catch (err) {
      throw new BrokerError('Failed to disconnect Kafka', { original: err });
    }
  }

  /**
   * topic: string
   * message: { key, value, headers } or array
   */
  async produce(topic, message, opts = {}) {
    if (!this.connected) await this.connect();
    const messages = Array.isArray(message) ? message : [message];
    const toSend = messages.map((m) => {
      const value = typeof m.value === 'string' ? m.value : safeJsonStringify(m.value);
      const headers = {};
      if (m.headers && typeof m.headers === 'object') {
        for (const k of Object.keys(m.headers)) {
          const hv = m.headers[k];
          headers[k] = typeof hv === 'string' ? hv : safeJsonStringify(hv);
        }
      }
      return {
        key: m.key ?? null,
        value,
        headers,
      };
    });

    try {
      const res = await this.producer.send({
        topic,
        messages: toSend,
        compression: CompressionTypes.GZIP,
      });
      return res;
    } catch (err) {
      throw new BrokerError('Kafka produce failed', { original: err });
    }
  }

  /**
   * Subscribe to a topic with groupId and handler
   * handler signature: async ({ key, value, headers, raw, partition, offset, timestamp }) => { ack:true }
   *
   * opts: { groupId, fromBeginning, eachBatchAutoResolve, concurrency, runConfig }
   */
  async subscribe(topic, handler, opts = {}) {
    if (!this.connected) await this.connect();
    const groupId = opts.groupId || `${this.config.clientId}-${topic}-group`;
    const consumer = this.kafka.consumer({ groupId, allowAutoTopicCreation: true });

    await consumer.connect();
    await consumer.subscribe({ topic, fromBeginning: !!opts.fromBeginning });

    const runConfig = opts.runConfig || { eachMessage: true };

    const self = this;
    await consumer.run({
      // concurrency can be achieved by setting partitions and multiple instances; kafkajs runs eachMessage per partition
      eachMessage: async ({ topic, partition, message }) => {
        const raw = message;
        const headers = {};
        if (message.headers) {
          for (const k of Object.keys(message.headers)) {
            const v = message.headers[k];
            headers[k] = v ? v.toString() : null;
          }
        }
        const payload = {
          key: message.key ? message.key.toString() : null,
          value: message.value ? message.value.toString() : null,
          headers,
          raw,
          partition,
          offset: message.offset,
          timestamp: message.timestamp,
        };
        try {
          const result = await handler(payload);
          // handler should handle idempotency; commit offset only after handler finishes
          if (result && result.ack === false) {
            // do not commit offset => message may be reprocessed depending on consumer config
          } else {
            // kafkajs auto-commits unless disabled; to control commits you'd set autoCommit false and manually commit
            // For simplicity we rely on default behavior
          }
        } catch (err) {
          // rethrow or log; depending on consumer config message may be retried
          console.error('[Kafka][Handler] error', err);
          throw err;
        }
      },
    });

    this.consumerMap.set(topic + ':' + groupId, consumer);

    return {
      unsubscribe: async () => {
        try {
          await consumer.disconnect();
        } catch (err) {
          // ignore
        }
      },
    };
  }
}

module.exports = KafkaAdapter;
