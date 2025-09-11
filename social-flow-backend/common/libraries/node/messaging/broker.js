/**
 * common/libraries/node/messaging/broker.js
 *
 * Abstract Broker adapter interface. Concrete adapters (Kafka/Rabbit) should implement:
 *  - connect()
 *  - disconnect()
 *  - produce(topic, message, opts)
 *  - subscribe(topic, handler, opts)
 *  - ack/ nack (if applicable)
 *
 * The adapters should normalize message shape to:
 *  {
 *    key: (string|null),
 *    value: Buffer | string,
 *    headers: Object
 *  }
 *
 * This class is a guideline and not enforced, but Producer/Consumer will expect the contract.
 */

class Broker {
  constructor(opts = {}) {
    this.opts = opts;
  }

  async connect() {
    throw new Error('connect() not implemented');
  }

  async disconnect() {
    throw new Error('disconnect() not implemented');
  }

  /**
   * Produce a single message or batch.
   * @param {string} topic
   * @param {object|object[]} message
   * @param {object} opts
   */
  async produce(topic, message, opts = {}) {
    throw new Error('produce() not implemented');
  }

  /**
   * Subscribe to a topic/queue with a handler function.
   * handler receives { key, value, headers, raw } and should return { ack: true|false, requeue: true|false } or throw
   */
  async subscribe(topic, handler, opts = {}) {
    throw new Error('subscribe() not implemented');
  }
}

module.exports = Broker;
