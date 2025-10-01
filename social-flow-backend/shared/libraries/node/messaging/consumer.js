/**
 * common/libraries/node/messaging/consumer.js
 *
 * High-level Consumer abstraction. Features:
 *  - subscribe to topic/queue with concurrency control and automatic retries
 *  - schema validation, dead-letter handling, ack/nack semantics
 *  - handler wrapper that provides helpers: ack(), nack(), requeue()
 */

const Config = require('./config');
const { ConsumeError } = require('./errors');
const { backoffMs, sleep, safeJsonParse } = require('./utils');

class Consumer {
  /**
   * @param {Object} opts
   *  - adapter: Broker adapter instance
   *  - schema: optional Schema registry
   *  - concurrency, maxRetries, deadLetterSuffix ...
   */
  constructor(opts = {}) {
    if (!opts.adapter) throw new Error('adapter is required');
    this.adapter = opts.adapter;
    this.schema = opts.schema || null;
    this.concurrency = opts.concurrency ?? Config.CONSUMER.concurrency;
    this.maxRetries = opts.maxRetries ?? Config.CONSUMER.maxRetries;
    this.deadLetterSuffix = opts.deadLetterSuffix ?? Config.CONSUMER.deadLetterSuffix;
    this.running = false;
    this._subscriptions = [];
  }

  async connect() {
    if (this.adapter.connect) await this.adapter.connect();
  }

  async disconnect() {
    for (const s of this._subscriptions) {
      try {
        if (s.unsubscribe) await s.unsubscribe();
      } catch (e) {}
    }
    if (this.adapter.disconnect) await this.adapter.disconnect();
    this.running = false;
  }

  /**
   * subscribe to a topic/queue
   * handler: async function(payload, helpers) -> { ack: true|false } or throw
   * opts: forwarded to adapter
   */
  async subscribe(topic, handler, opts = {}) {
    const wrapper = this._makeHandlerWrapper(handler, opts);
    const subscription = await this.adapter.subscribe(topic, wrapper, opts);
    this._subscriptions.push(subscription);
    return subscription;
  }

  _makeHandlerWrapper(handler, opts = {}) {
    const self = this;
    return async (payload) => {
      // normalize payload.value to object if JSON
      let parsedValue = payload.value;
      if (typeof payload.value === 'string') {
        const maybe = safeJsonParse(payload.value);
        if (maybe !== null) parsedValue = maybe;
      }

      const context = {
        ack: false,
        nack: false,
        requeue: false,
        attempts: 0,
        metadata: payload.headers || {},
        helpers: {
          ack: () => { context.ack = true; },
          nack: (requeue = false) => { context.nack = true; context.requeue = requeue; },
        },
      };

      // simple retry loop
      let attempt = 0;
      while (attempt <= this.maxRetries) {
        try {
          // schema validation if present
          if (opts.schemaName && this.schema) {
            this.schema.validate(opts.schemaName, parsedValue);
          }

          const result = await handler({ ...payload, value: parsedValue }, context.helpers);
          // if handler returns object instructing ack/nack, honor it
          if (result && result.ack === false) {
            // allow nack
            return { ack: false, requeue: !!result.requeue };
          }
          // default ack unless handler asked not to
          return { ack: true };
        } catch (err) {
          attempt++;
          context.attempts = attempt;
          // if retries remain, backoff and retry
          if (attempt <= this.maxRetries) {
            const wait = backoffMs(attempt);
            await sleep(wait);
            continue;
          }
          // otherwise send to dead-letter if configured
          if (opts.deadLetterEnabled !== false) {
            try {
              await self._sendToDLQ(payload, err, opts);
            } catch (dlqErr) {
              console.error('[Consumer] failed to send to DLQ', dlqErr);
            }
          }
          // bubble error
          throw new ConsumeError('Handler failed after retries', { original: err, topic: opts.queue || opts.topic || 'unknown' });
        }
      }
    };
  }

  async _sendToDLQ(payload, err, opts = {}) {
    const dlqTopic = (opts.deadLetterTopic || (opts.queue || opts.topic) + this.deadLetterSuffix);
    const msg = {
      key: payload.key,
      value: payload.value,
      headers: {
        ...payload.headers,
        'x-dead-letter-reason': String(err && (err.message || err)),
        'x-original-topic': opts.topic || opts.queue,
      },
    };
    // best-effort produce to DLQ
    try {
      await this.adapter.produce(dlqTopic, msg, {});
    } catch (e) {
      console.error('[Consumer][DLQ] produce failed', e);
    }
  }
}

module.exports = Consumer;
