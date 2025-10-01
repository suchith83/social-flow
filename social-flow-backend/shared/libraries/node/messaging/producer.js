/**
 * common/libraries/node/messaging/producer.js
 *
 * High-level Producer abstraction. Provides:
 *  - send(topic, message, opts)
 *  - sendBatch(topic, messages, opts)
 *  - automatic retries with backoff
 *  - optional batching for throughput
 *  - schema validation hook
 *
 * Example:
 *  const p = new Producer({ adapter: new KafkaAdapter(), schema: Schema });
 *  await p.send('user.events', { key: '123', value: {...}, headers: {} });
 */

const Config = require('./config');
const { ProduceError } = require('./errors');
const { backoffMs, genId, safeJsonStringify, attachIdempotency } = require('./utils');

class Producer {
  /**
   * @param {Object} opts
   *   - adapter: instance of Broker adapter (KafkaAdapter/RabbitAdapter)
   *   - schema: optional schema registry instance
   *   - defaultRetries, baseBackoffMs, batch settings...
   */
  constructor(opts = {}) {
    if (!opts.adapter) throw new Error('adapter is required');
    this.adapter = opts.adapter;
    this.schema = opts.schema || null;
    this.retries = opts.defaultRetries ?? Config.PRODUCER.defaultRetries;
    this.baseBackoff = opts.baseBackoffMs ?? Config.PRODUCER.baseBackoffMs;
    this.batchMaxSize = opts.batchMaxSize ?? Config.PRODUCER.batchMaxSize;
    this.batchMaxTimeMs = opts.batchMaxTimeMs ?? Config.PRODUCER.batchMaxTimeMs;
    this.idempotent = opts.idempotent ?? Config.PRODUCER.idempotent;
    this.batchQueues = new Map();
    this.batchTimers = new Map();
    this.closed = false;
  }

  async connect() {
    if (this.adapter.connect) await this.adapter.connect();
  }

  async disconnect() {
    // flush batches
    for (const [topic] of this.batchQueues) {
      await this._flushBatch(topic);
    }
    if (this.adapter.disconnect) await this.adapter.disconnect();
    this.closed = true;
  }

  async send(topic, message, opts = {}) {
    // Validate schema if provided
    if (opts.schemaName && this.schema) {
      this.schema.validate(opts.schemaName, message.value);
    }

    // Attach idempotency key if requested
    const idKey = opts.idempotencyKey || (this.idempotent ? genId('idp-') : undefined);
    if (idKey) message.headers = attachIdempotency(message.headers || {}, idKey);

    const attempts = opts.retries ?? this.retries;
    let attempt = 0;
    let lastErr;
    while (attempt <= attempts) {
      try {
        await this.adapter.produce(topic, message, opts);
        return true;
      } catch (err) {
        lastErr = err;
        if (attempt < attempts) {
          const wait = backoffMs(attempt, this.baseBackoff);
          await new Promise((r) => setTimeout(r, wait));
          attempt++;
          continue;
        }
        throw new ProduceError('Failed to send message', { original: err, topic, message });
      }
    }
    throw new ProduceError('Retries exhausted', { original: lastErr, topic });
  }

  /**
   * Batch API: adds a message to in-memory queue and flushes either when size/time threshold hit.
   */
  async sendBatch(topic, message, opts = {}) {
    if (!this.batchQueues.has(topic)) this.batchQueues.set(topic, []);
    const queue = this.batchQueues.get(topic);
    queue.push({ message, opts });

    // start timer if not exists
    if (!this.batchTimers.has(topic)) {
      const t = setTimeout(async () => {
        try {
          await this._flushBatch(topic);
        } catch (e) {
          console.error('[Producer] batch flush failed', e);
        }
      }, this.batchMaxTimeMs);
      this.batchTimers.set(topic, t);
    }

    // flush if size hit
    if (queue.length >= this.batchMaxSize) {
      await this._flushBatch(topic);
    }
  }

  async _flushBatch(topic) {
    const queue = this.batchQueues.get(topic) || [];
    if (!queue.length) return;

    // clear timer
    const t = this.batchTimers.get(topic);
    if (t) {
      clearTimeout(t);
      this.batchTimers.delete(topic);
    }

    // group messages
    const msgs = queue.map((q) => {
      const m = q.message;
      return m;
    });

    this.batchQueues.set(topic, []); // reset queue

    // call adapter.produce with array
    try {
      await this.adapter.produce(topic, msgs, { batch: true });
    } catch (err) {
      // on failure, try to re-queue or surface error
      console.error('[Producer] batch produce failed', err);
      // naive fallback: try individual sends with retries
      for (const m of msgs) {
        try {
          await this.send(topic, m);
        } catch (e) {
          console.error('[Producer] fallback send failed', e);
        }
      }
    }
  }
}

module.exports = Producer;
