/**
 * common/libraries/node/database/queryBuilder.js
 *
 * Lightweight SQL builder:
 *  - insert(table, data)
 *  - update(table, data, where)
 *  - delete(table, where)
 *  - select(table, fields, where, options)
 *
 * The builder returns { text, params } for use with .query(text, params)
 * It respects parameter placeholder differences between drivers via client param.
 *
 * NOTE: This is intentionally simple — for complex queries use raw SQL or a full query library.
 */

const { safeIdent, paramPlaceholder } = require('./utils');

function buildWhere(where = {}, client = 'pg', startIndex = 1) {
  const clauses = [];
  const params = [];
  let idx = startIndex;
  for (const key of Object.keys(where)) {
    const field = safeIdent(key);
    const ph = paramPlaceholder(client, idx++);
    clauses.push(`${field} = ${ph}`);
    params.push(where[key]);
  }
  return {
    text: clauses.length ? `WHERE ${clauses.join(' AND ')}` : '',
    params,
    nextIndex: idx,
  };
}

function insert(table, data = {}, client = 'pg') {
  if (!table || Object.keys(data).length === 0) {
    throw new Error('insert requires table and non-empty data');
  }
  const cols = Object.keys(data).map(safeIdent);
  const params = Object.values(data);
  if (client === 'pg') {
    const placeholders = cols.map((_, i) => `$${i + 1}`);
    const text = `INSERT INTO ${safeIdent(table)} (${cols.join(',')}) VALUES (${placeholders.join(',')}) RETURNING *`;
    return { text, params };
  }
  // mysql uses ?
  const placeholders = cols.map(() => '?');
  const text = `INSERT INTO ${safeIdent(table)} (${cols.join(',')}) VALUES (${placeholders.join(',')})`;
  return { text, params };
}

function update(table, data = {}, where = {}, client = 'pg') {
  if (!table || Object.keys(data).length === 0) {
    throw new Error('update requires table and non-empty data');
  }
  const setClauses = [];
  const params = [];
  let idx = 1;
  for (const [k, v] of Object.entries(data)) {
    const field = safeIdent(k);
    const ph = client === 'pg' ? `$${idx}` : '?';
    setClauses.push(`${field} = ${ph}`);
    params.push(v);
    idx++;
  }
  const whereRes = buildWhere(where, client, idx);
  const text = `UPDATE ${safeIdent(table)} SET ${setClauses.join(', ')} ${whereRes.text} RETURNING *`;
  return { text, params: params.concat(whereRes.params) };
}

function del(table, where = {}, client = 'pg') {
  const whereRes = buildWhere(where, client);
  const text = `DELETE FROM ${safeIdent(table)} ${whereRes.text}`;
  return { text, params: whereRes.params };
}

function select(table, fields = ['*'], where = {}, options = {}, client = 'pg') {
  const cols = fields.map((f) => (f === '*' ? '*' : safeIdent(f))).join(',');
  const whereRes = buildWhere(where, client);
  let text = `SELECT ${cols} FROM ${safeIdent(table)} ${whereRes.text}`;
  if (options.orderBy) {
    text += ` ORDER BY ${safeIdent(options.orderBy)} ${options.order || 'ASC'}`;
  }
  if (options.limit) {
    text += ` LIMIT ${options.limit}`;
  }
  if (options.offset) {
    text += ` OFFSET ${options.offset}`;
  }
  return { text, params: whereRes.params };
}

module.exports = {
  insert,
  update,
  delete: del,
  select,
};
