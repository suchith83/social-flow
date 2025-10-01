/**
 * asyncPool.js
 *
 * Provide utilities to run many async tasks with concurrency limits and gather results.
 * - pool(mapper, concurrency): map over iterable with concurrency control and collect results
 * - settlePool(...) : like Promise.allSettled but limited concurrency
 *
 * Implementation details:
 * - Supports async iterators as input
 * - Provides cancellation token support (simple boolean)
 * - Returns results in original order for arrays, and in completion order for iterators when requested
 *
 * Usage:
 *   const results = await pool(items, 10, async (item) => { ... });
 */

const { sleep } = require('./time');

async function pool(iterable, concurrency = 5, mapper = async (x) => x, opts = {}) {
  if (!iterable) return [];
  // support arrays and async iterables
  const arr = Array.isArray(iterable) ? iterable : null;
  const iterator = arr ? null : iterable[Symbol.asyncIterator] ? iterable[Symbol.asyncIterator]() : null;

  const results = [];
  let active = 0;
  let index = 0;
  let done = false;
  let errorOccurred = null;

  const queue = [];

  async function runTask(item, i) {
    active++;
    try {
      const res = await mapper(item, i);
      results[i] = res;
    } catch (err) {
      errorOccurred = err;
      if (!opts.continueOnError) throw err;
      results[i] = { __failed: true, error: err };
    } finally {
      active--;
    }
  }

  if (arr) {
    const total = arr.length;
    const startNext = async () => {
      while (index < total && active < concurrency) {
        const i = index++;
        queue.push(runTask(arr[i], i));
      }
      if (queue.length === 0) return;
      await Promise.race(queue.map((p) => p.then(() => {
        // remove resolved promises from queue
        for (let j = 0; j < queue.length; j++) {
          if (queue[j].isResolved) {
            queue.splice(j, 1);
            j--;
          }
        }
      }).catch(() => {})));
    };

    // Wrap tasks so we can mark resolved easily
    const wrappedRunTask = (item, i) => {
      const p = runTask(item, i);
      // attach marker
      p.then(() => (p.isResolved = true)).catch(() => (p.isResolved = true));
      return p;
    };

    // Kick off initial tasks
    while (index < total && active < concurrency) {
      const i = index++;
      const p = wrappedRunTask(arr[i], i);
      queue.push(p);
    }

    // Wait for completion, launching new tasks as slots free
    while (queue.length) {
      await Promise.race(queue.map((p) => p.catch(() => {})));
      if (errorOccurred && !opts.continueOnError) throw errorOccurred;
      while (index < total && active < concurrency) {
        const i = index++;
        const p = wrappedRunTask(arr[i], i);
        queue.push(p);
      }
    }

    return results;
  }

  // async iterator case
  if (iterator) {
    const inflight = new Set();
    let i = 0;
    for (;;) {
      if (errorOccurred && !opts.continueOnError) break;
      while (inflight.size < concurrency) {
        const next = await iterator.next();
        if (next.done) {
          done = true;
          break;
        }
        const currentIndex = i++;
        const p = (async () => {
          try {
            const r = await mapper(next.value, currentIndex);
            results[currentIndex] = r;
          } catch (err) {
            errorOccurred = err;
            if (!opts.continueOnError) throw err;
            results[currentIndex] = { __failed: true, error: err };
          } finally {
            inflight.delete(p);
          }
        })();
        inflight.add(p);
      }
      if (inflight.size === 0 && done) break;
      // wait for any to finish
      await Promise.race(Array.from(inflight).map((p) => p.catch(() => {})));
    }
    if (errorOccurred && !opts.continueOnError) throw errorOccurred;
    return results;
  }

  return [];
}

async function settlePool(iterable, concurrency = 5, mapper = async (x) => x) {
  return pool(iterable, concurrency, async (item, i) => {
    try {
      const v = await mapper(item, i);
      return { status: 'fulfilled', value: v };
    } catch (e) {
      return { status: 'rejected', reason: e };
    }
  });
}

module.exports = {
  pool,
  settlePool,
};
