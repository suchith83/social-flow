/**
 * common/libraries/node/database/client.js
 *
 * High-level DB client used by services:
 *  - query(text, params, options)
 *  - transaction(fn) => runs provided function with transactional client
 *  - withRetry wrapper for transient errors
 *
 * The client normalizes results across pg and mysql.
 */

const initPool = require('./pool');
const Config = require('./config');
const { isTransientError, backoff, sleep, paramPlaceholder } = require('./utils');
const { QueryError, TransactionError } = require('./errors');

class DBClient {
  /**
   * create a DBClient that wraps a pool instance.
   * pool should be initialized with initPool()
   */
  constructor(poolInstance = null, opts = {}) {
    this.pool = poolInstance || null;
    this.opts = opts;
    this.logQueries = Config.LOG_QUERIES;
    this.retry = Config.DB_QUERY_RETRY;
    this.baseBackoff = Config.DB_QUERY_RETRY_BACKOFF_MS;
  }

  static async init(customConfig = {}) {
    const pool = await initPool(customConfig);
    return new DBClient(pool);
  }

  _normalizeResult(raw, client) {
    // Normalize pg and mysql results to { rows, rowCount }
    if (client === 'pg') {
      return { rows: raw.rows, rowCount: raw.rowCount };
    }
    // mysql2 returns [rows, fields]
    if (client === 'mysql') {
      const [rows] = raw;
      return { rows, rowCount: Array.isArray(rows) ? rows.length : 0 };
    }
    return raw;
  }

  async query(text, params = [], opts = {}) {
    const attempts = opts.retries ?? this.retry;
    let attempt = 0;
    let lastErr;

    const pool = this.pool;
    if (!pool) throw new Error('Pool not initialized');

    while (attempt <= attempts) {
      const client = await pool.acquire();
      try {
        if (this.logQueries) {
          console.debug(`[DB][QUERY](${pool.client})`, text, params);
        }

        const rawRes = await client.query(text, params);
        const res = this._normalizeResult(rawRes, pool.client);
        client.release();
        return res;
      } catch (err) {
        client.release();
        lastErr = err;

        if (isTransientError(err) && attempt < attempts) {
          const back = backoff(attempt, this.baseBackoff);
          await sleep(back);
          attempt += 1;
          continue;
        }

        // Not transient or retries exhausted
        throw new QueryError('Query failed', { original: err, sql: text, params });
      }
    }

    throw new QueryError('Query retry exhausted', { original: lastErr, sql: text });
  }

  /**
   * transaction(fn)
   * Runs user function with a transactional client. Supports nested transactions using savepoints.
   *
   * fn receives: txClient which has .query(), .commit(), .rollback()
   *
   * returns the value returned by fn
   */
  async transaction(fn, opts = {}) {
    const pool = this.pool;
    if (!pool) throw new Error('Pool not initialized');

    const rawClient = await pool.acquire();
    // Use the adapter client directly for transaction control
    const tx = rawClient;
    const clientType = pool.client;

    const savepointStack = [];

    // Helper to run queries on parent connection
    const runQuery = async (text, params = []) => {
      try {
        const rawRes = await tx.query(text, params);
        return this._normalizeResult(rawRes, clientType);
      } catch (err) {
        throw err;
      }
    };

    // Transaction API provided to user function
    const txApi = {
      query: runQuery,
      // nested transaction using savepoint
      savepoint: async () => {
        const sp = `sp_${Math.random().toString(36).slice(2, 8)}`;
        if (clientType === 'pg') {
          await tx.query(`SAVEPOINT ${sp}`);
        } else {
          await tx.query(`SAVEPOINT ${sp}`);
        }
        savepointStack.push(sp);
        return sp;
      },
      releaseSavepoint: async (sp) => {
        if (!sp) sp = savepointStack.pop();
        if (!sp) return;
        await tx.query(`RELEASE SAVEPOINT ${sp}`);
      },
      rollbackToSavepoint: async (sp) => {
        if (!sp) sp = savepointStack.pop();
        if (!sp) return;
        await tx.query(`ROLLBACK TO SAVEPOINT ${sp}`);
      },
    };

    try {
      // Begin transaction
      await tx.begin();

      const result = await fn(txApi);

      await tx.commit();
      rawClient.release();
      return result;
    } catch (err) {
      try {
        await tx.rollback();
      } catch (rollbackErr) {
        // swallow rollback error but annotate
        console.error('[DB][TRANSACTION] rollback error', rollbackErr);
      } finally {
        rawClient.release();
      }
      throw new TransactionError('Transaction failed', { original: err });
    }
  }

  /**
   * health check: try to acquire and run a simple query
   */
  async health() {
    const pool = this.pool;
    if (!pool) throw new Error('Pool not initialized');

    const client = await pool.acquire();
    try {
      if (pool.client === 'pg') {
        await client.query('SELECT 1');
      } else {
        await client.query('SELECT 1');
      }
      client.release();
      return { healthy: true, client: pool.client };
    } catch (err) {
      client.release();
      return { healthy: false, client: pool.client, error: err.message || String(err) };
    }
  }
}

module.exports = DBClient;
