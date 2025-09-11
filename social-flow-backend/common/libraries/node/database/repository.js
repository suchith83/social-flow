/**
 * common/libraries/node/database/repository.js
 *
 * Base repository class providing common CRUD operations, to be extended per-model/aggregate.
 *
 * Example:
 *  class UserRepo extends Repository {
 *    constructor(db) { super(db, 'users'); }
 *  }
 *
 * It uses QueryBuilder to generate SQL and DBClient to execute queries.
 */

const QueryBuilder = require('./queryBuilder');
const Config = require('./config');

class Repository {
  /**
   * @param {DBClient} dbClient - instance of DBClient
   * @param {string} table - table name
   * @param {Object} options - { client: 'pg'|'mysql' }
   */
  constructor(dbClient, table, options = {}) {
    if (!dbClient) throw new Error('dbClient is required');
    if (!table) throw new Error('table is required');
    this.db = dbClient;
    this.table = table;
    this.client = options.client || Config.DB_CLIENT;
  }

  async create(data) {
    const { text, params } = QueryBuilder.insert(this.table, data, this.client);
    const res = await this.db.query(text, params);
    return res.rows[0] ?? null;
  }

  async findById(id) {
    const { text, params } = QueryBuilder.select(this.table, ['*'], { id }, {}, this.client);
    const res = await this.db.query(text, params);
    return res.rows[0] ?? null;
  }

  async findOne(where = {}) {
    const { text, params } = QueryBuilder.select(this.table, ['*'], where, { limit: 1 }, this.client);
    const res = await this.db.query(text, params);
    return res.rows[0] ?? null;
  }

  async find(where = {}, options = {}) {
    const { text, params } = QueryBuilder.select(this.table, ['*'], where, options, this.client);
    const res = await this.db.query(text, params);
    return res.rows;
  }

  async update(where = {}, data = {}) {
    const { text, params } = QueryBuilder.update(this.table, data, where, this.client);
    const res = await this.db.query(text, params);
    return res.rows;
  }

  async delete(where = {}) {
    const { text, params } = QueryBuilder.delete(this.table, where, this.client);
    const res = await this.db.query(text, params);
    return res;
  }

  async transaction(fn) {
    return this.db.transaction(fn);
  }
}

module.exports = Repository;
