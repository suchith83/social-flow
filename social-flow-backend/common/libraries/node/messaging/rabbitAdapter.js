/**
 * common/libraries/node/messaging/rabbitAdapter.js
 *
 * Adapter for RabbitMQ using amqplib.
 * Normalizes produce/consume operations.
 */

const amqplib = require('amqplib');
const Broker = require('./broker');
const Config = require('./config');
const { BrokerError } = require('./errors');
const { safeJsonStringify } = require('./utils');

class RabbitAdapter extends Broker {
  constructor(opts = {}) {
    super(opts);
    this.url = opts.url || Config.RABBIT.url;
    this.prefetch = opts.prefetch || Config.RABBIT.prefetch || 20;
    this.conn = null;
    this.channels = new Map();
  }

  async connect() {
    if (this.conn) return;
    try {
      this.conn = await amqplib.connect(this.url);
      // handle close
      this.conn.on('close', () => {
        console.warn('[Rabbit] connection closed');
        this.conn = null;
      });
      this.conn.on('error', (err) => {
        console.error('[Rabbit] connection error', err);
      });
    } catch (err) {
      throw new BrokerError('Failed to connect RabbitMQ', { original: err });
    }
  }

  async disconnect() {
    try {
      for (const ch of this.channels.values()) {
        try {
          await ch.close();
        } catch (e) {}
      }
      this.channels.clear();
      if (this.conn) await this.conn.close();
      this.conn = null;
    } catch (err) {
      throw new BrokerError('Failed to disconnect RabbitMQ', { original: err });
    }
  }

  async _getChannel(name = 'default') {
    if (!this.conn) await this.connect();
    if (this.channels.has(name)) return this.channels.get(name);
    const ch = await this.conn.createChannel();
    ch.prefetch(this.prefetch);
    this.channels.set(name, ch);
    return ch;
  }

  /**
   * Produce to exchange or queue
   * message: { key, value, headers } where key maps to routingKey
   */
  async produce(destination, message, opts = {}) {
    const ch = await this._getChannel((opts.channelName || 'producer'));
    // if destination contains exchange: 'exchangeName' or queue: '', simplest: use default exchange to queue if options.queue=true
    const messages = Array.isArray(message) ? message : [message];
    const results = [];
    try {
      // Ensure queue exists if producing directly to queue
      if (opts.toQueue) {
        await ch.assertQueue(destination, { durable: true });
      } else if (opts.exchange) {
        await ch.assertExchange(destination, opts.exchangeType || 'topic', { durable: true });
      }

      for (const m of messages) {
        const body = typeof m.value === 'string' ? Buffer.from(m.value) : Buffer.from(safeJsonStringify(m.value));
        const headers = m.headers || {};
        const routingKey = m.key || (opts.routingKey || '');
        let ok = false;
        if (opts.toQueue) {
          ok = ch.sendToQueue(destination, body, { persistent: true, headers, messageId: headers['x-msg-idempotency-key'] || undefined });
        } else {
          ok = ch.publish(destination, routingKey, body, { persistent: true, headers, messageId: headers['x-msg-idempotency-key'] || undefined });
        }
        results.push({ ok, routingKey });
      }
      return results;
    } catch (err) {
      throw new BrokerError('Rabbit produce failed', { original: err });
    }
  }

  /**
   * Subscribe to queue with handler
   * handler receives { key, value, headers, raw, msg } and should return { ack: true|false, nack: true|false, requeue: true|false }
   *
   * opts: { queue, exchange, routingKey, durable, prefetch }
   */
  async subscribe(destination, handler, opts = {}) {
    const ch = await this._getChannel(opts.channelName || 'consumer');
    const queue = opts.queue || destination;
    try {
      await ch.assertQueue(queue, { durable: true });
      if (opts.exchange) {
        await ch.assertExchange(opts.exchange, opts.exchangeType || 'topic', { durable: true });
        await ch.bindQueue(queue, opts.exchange, opts.routingKey || '#');
      }
      const consumerTag = (await ch.consume(queue, async (msg) => {
        if (!msg) return;
        const headers = msg.properties.headers || {};
        const payload = {
          key: msg.fields.routingKey || null,
          value: msg.content ? msg.content.toString() : null,
          headers,
          raw: msg,
          msg,
        };
        try {
          const result = await handler(payload);
          if (result && result.ack === false) {
            // by default do not ack
            if (result.nack) ch.nack(msg, false, !!result.requeue);
          } else {
            ch.ack(msg);
          }
        } catch (err) {
          console.error('[Rabbit][Handler] error', err);
          // On error nack and optionally requeue false so message lands in DLQ via DLX
          try {
            ch.nack(msg, false, !!opts.requeue);
          } catch (e) {
            console.error('[Rabbit] nack failed', e);
          }
        }
      }, { noAck: false })).consumerTag;

      return {
        unsubscribe: async () => {
          try {
            await ch.cancel(consumerTag);
          } catch (e) {}
        },
      };
    } catch (err) {
      throw new BrokerError('Rabbit subscribe failed', { original: err });
    }
  }
}

module.exports = RabbitAdapter;
