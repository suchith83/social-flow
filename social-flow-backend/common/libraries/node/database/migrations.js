/**
 * common/libraries/node/database/migrations.js
 *
 * Tiny migration runner that stores applied migrations in a migrations table.
 *
 * Migrations are JS files in the migrations directory exporting `up` and `down` async functions.
 *
 * Usage:
 *  const migrator = new Migrations(dbClient, { dir: './migrations' });
 *  await migrator.init();
 *  await migrator.up(); // applies pending migrations
 *
 * This runner is intentionally minimal (no parallel runners). For heavy apps prefer a battle-tested tool.
 */

const fs = require('fs');
const path = require('path');
const Config = require('./config');

class Migrations {
  constructor(dbClient, opts = {}) {
    if (!dbClient) throw new Error('dbClient required');
    this.db = dbClient;
    this.dir = opts.dir || Config.MIGRATIONS_DIR;
    this.table = opts.table || 'schema_migrations';
  }

  async init() {
    // Ensure migrations table exists
    const createSqlPg = `CREATE TABLE IF NOT EXISTS ${this.table} (id SERIAL PRIMARY KEY, name TEXT NOT NULL UNIQUE, run_on TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW())`;
    const createSqlMysql = `CREATE TABLE IF NOT EXISTS ${this.table} (id INT AUTO_INCREMENT PRIMARY KEY, name VARCHAR(255) NOT NULL UNIQUE, run_on TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP)`;

    // Try create for both; rely on client to handle differences
    try {
      await this.db.query(createSqlPg);
    } catch (err) {
      // try mysql variant
      try {
        await this.db.query(createSqlMysql);
      } catch (e) {
        // ignore — we just ensure migrations table exists for whichever DB
      }
    }
  }

  async listLocalMigrations() {
    if (!fs.existsSync(this.dir)) return [];
    const files = fs.readdirSync(this.dir).filter((f) => f.endsWith('.js'));
    files.sort();
    return files;
  }

  async listApplied() {
    const res = await this.db.query(`SELECT name FROM ${this.table} ORDER BY name`);
    return (res.rows || []).map((r) => r.name);
  }

  async up() {
    await this.init();
    const locals = await this.listLocalMigrations();
    const applied = await this.listApplied();
    const pending = locals.filter((f) => !applied.includes(f));
    for (const file of pending) {
      const full = path.join(this.dir, file);
      const mod = require(full);
      if (!mod.up || typeof mod.up !== 'function') {
        console.warn(`[MIGRATIONS] skipping ${file}: no up() function`);
        continue;
      }
      console.info(`[MIGRATIONS] applying ${file}`);
      await this.db.transaction(async (tx) => {
        await mod.up(tx); // pass transactional client
        // record migration
        await tx.query(`INSERT INTO ${this.table} (name) VALUES ($1)`, [file]);
      });
    }
    return pending;
  }

  async down(steps = 1) {
    await this.init();
    const applied = await this.listApplied();
    const toRollback = applied.slice(-steps).reverse();
    for (const file of toRollback) {
      const full = path.join(this.dir, file);
      const mod = require(full);
      if (!mod.down || typeof mod.down !== 'function') {
        console.warn(`[MIGRATIONS] skipping ${file}: no down() function`);
        continue;
      }
      console.info(`[MIGRATIONS] reverting ${file}`);
      await this.db.transaction(async (tx) => {
        await mod.down(tx);
        await tx.query(`DELETE FROM ${this.table} WHERE name = $1`, [file]);
      });
    }
    return toRollback;
  }
}

module.exports = Migrations;
