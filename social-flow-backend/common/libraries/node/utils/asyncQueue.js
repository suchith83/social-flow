/**
 * asyncQueue.js
 *
 * Implements a bounded async queue with backpressure and worker semantics.
 * - enqueue(item): returns a Promise resolved when item is queued (or rejected if closed)
 * - startWorker(fn, concurrency): runs worker function that pulls items and processes them
 * - close() stops accepting new items and resolves when drained
 *
 * Features:
 * - optional prefetch/backpressure via maxSize
 * - graceful shutdown
 * - emits simple events via callbacks
 *
 * This is useful when streaming events into a bounded processor.
 */

const EventEmitter = require('events');

class AsyncQueue extends EventEmitter {
  constructor({ maxSize = 1000 } = {}) {
    super();
    this.maxSize = maxSize;
    this._queue = [];
    this._waitingResolvers = [];
    this._closed = false;
    this._drainPromise = null;
    this._drainResolve = null;
  }

  get size() {
    return this._queue.length;
  }

  enqueue(item, { signal } = {}) {
    if (this._closed) return Promise.reject(new Error('Queue closed'));
    if (signal && signal.aborted) return Promise.reject(new Error('Aborted'));
    // backpressure: if queue is full, wait until space
    if (this._queue.length >= this.maxSize) {
      return new Promise((resolve, reject) => {
        const onAbort = () => { reject(new Error('Aborted')); cleanup(); };
        const cleanup = () => {
          if (signal) signal.removeEventListener('abort', onAbort);
        };
        if (signal) signal.addEventListener('abort', onAbort);
        const check = () => {
          if (this._closed) {
            cleanup();
            return reject(new Error('Queue closed'));
          }
          if (this._queue.length < this.maxSize) {
            this._queue.push(item);
            cleanup();
            this._notifyOne();
            resolve();
          } else {
            // retry later
            setImmediate(check);
          }
        };
        check();
      });
    } else {
      this._queue.push(item);
      this._notifyOne();
      return Promise.resolve();
    }
  }

  _notifyOne() {
    const resolver = this._waitingResolvers.shift();
    if (resolver) resolver();
  }

  async _dequeue() {
    if (this._queue.length > 0) return this._queue.shift();
    if (this._closed) return null;
    // wait for an item or close
    await new Promise((resolve) => this._waitingResolvers.push(resolve));
    if (this._queue.length === 0) return null;
    return this._queue.shift();
  }

  startWorker(workerFn, { concurrency = 1, onError = null } = {}) {
    if (typeof workerFn !== 'function') throw new Error('workerFn required');
    const runners = [];
    for (let i = 0; i < concurrency; i++) {
      runners.push((async () => {
        while (true) {
          const item = await this._dequeue();
          if (item === null) break;
          try {
            await workerFn(item);
            this.emit('processed', item);
          } catch (err) {
            this.emit('error', err, item);
            if (onError) onError(err, item);
          }
        }
      })());
    }
    return Promise.all(runners);
  }

  async drain() {
    if (!this._drainPromise) {
      this._drainPromise = new Promise((resolve) => {
        this._drainResolve = resolve;
        const check = () => {
          if (this._queue.length === 0) return resolve();
          setTimeout(check, 50);
        };
        check();
      });
    }
    return this._drainPromise;
  }

  close() {
    this._closed = true;
    // notify waiters to unblock
    while (this._waitingResolvers.length) {
      const r = this._waitingResolvers.shift();
      r();
    }
  }
}

module.exports = AsyncQueue;
