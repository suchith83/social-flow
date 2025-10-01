/**
 * common/libraries/node/database/pool.js
 *
 * Connection pool abstraction that supports:
 *  - Postgres (pg)
 *  - MySQL (mysql2)
 *
 * Exposes: initPool(config), acquire(), release(client), end()
 *
 * The pool always yields a thin adapter that offers .query(text, params) and
 * .execute for prepared statements. For transactions, client.begin/commit/rollback are provided.
 *
 * This file intentionally wraps native pools and normalizes error shapes.
 */

const Config = require('./config');
const { ConnectionError } = require('./errors');
const { genRandomId } = require('./utils');

let PoolInstance = null;

async function initPool(customConfig = {}) {
  const cfg = { ...Config, ...customConfig };
  if (PoolInstance) return PoolInstance;

  if (cfg.DB_CLIENT === 'pg') {
    // Postgres
    const { Pool } = require('pg');
    const pgPool = new Pool({
      host: cfg.DB_HOST,
      port: cfg.DB_PORT,
      user: cfg.DB_USER,
      password: cfg.DB_PASSWORD,
      database: cfg.DB_DATABASE,
      min: cfg.DB_POOL_MIN,
      max: cfg.DB_POOL_MAX,
      idleTimeoutMillis: cfg.DB_IDLE_TIMEOUT,
      connectionTimeoutMillis: cfg.DB_CONNECTION_TIMEOUT,
    });

    // Basic health check
    pgPool.on('error', (err) => {
      console.error('[DB][pg pool error]', err);
    });

    PoolInstance = {
      client: 'pg',
      raw: pgPool,
      acquire: async () => {
        try {
          const pgClient = await pgPool.connect();
          const id = genRandomId(6);
          const adapter = {
            __id: id,
            client: 'pg',
            raw: pgClient,
            query: async (text, params = []) => {
              // pg uses parameterized queries with $1 etc
              return pgClient.query(text, params);
            },
            release: () => {
              pgClient.release();
            },
            begin: async () => {
              await pgClient.query('BEGIN');
            },
            commit: async () => {
              await pgClient.query('COMMIT');
            },
            rollback: async () => {
              await pgClient.query('ROLLBACK');
            },
            // allow access to underlying connection if necessary
            getRaw: () => pgClient,
          };
          return adapter;
        } catch (err) {
          throw new ConnectionError('Failed to acquire pg client', { original: err });
        }
      },
      end: async () => {
        await pgPool.end();
      },
    };

    return PoolInstance;
  }

  // Default to mysql2
  if (cfg.DB_CLIENT === 'mysql') {
    const mysql = require('mysql2/promise');
    const pool = mysql.createPool({
      host: cfg.DB_HOST,
      port: cfg.DB_PORT,
      user: cfg.DB_USER,
      password: cfg.DB_PASSWORD,
      database: cfg.DB_DATABASE,
      waitForConnections: true,
      connectionLimit: cfg.DB_POOL_MAX,
      queueLimit: 0,
    });

    PoolInstance = {
      client: 'mysql',
      raw: pool,
      acquire: async () => {
        try {
          const conn = await pool.getConnection();
          const id = genRandomId(6);
          const adapter = {
            __id: id,
            client: 'mysql',
            raw: conn,
            query: async (text, params = []) => {
              // mysql2 returns [rows, fields]
              return conn.query(text, params);
            },
            release: () => {
              conn.release();
            },
            begin: async () => {
              await conn.beginTransaction();
            },
            commit: async () => {
              await conn.commit();
            },
            rollback: async () => {
              await conn.rollback();
            },
            getRaw: () => conn,
          };
          return adapter;
        } catch (err) {
          throw new ConnectionError('Failed to acquire mysql connection', { original: err });
        }
      },
      end: async () => {
        await pool.end();
      },
    };

    return PoolInstance;
  }

  throw new Error(`Unsupported DB_CLIENT: ${cfg.DB_CLIENT}`);
}

function getPool() {
  if (!PoolInstance) throw new Error('Pool not initialized. Call initPool() first.');
  return PoolInstance;
}

module.exports = initPool;
module.exports.getPool = getPool;
