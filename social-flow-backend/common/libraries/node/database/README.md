# common/libraries/node/database

Production-ready Node.js database helper library.

## Features
- Supports PostgreSQL (`pg`) and MySQL (`mysql2/promise`)
- Connection pooling abstraction
- Safe parameterized queries
- Lightweight query builder
- Repository base class for CRUD
- Transactions with nested savepoint support
- Simple migration runner
- Health checks and retry/backoff for transient errors
- Custom error types for clearer handling

## Quick Example

```js
const { Pool, Client, Repository, Migrations } = require('common/libraries/node/database');

// initialize pool and client
(async () => {
  const pool = await require('./pool')(); // or DBClient.init()
  const db = await Client.init();
  const repo = new Repository(db, 'users');

  // create
  const u = await repo.create({ email: 'a@example.com', name: 'Alice' });

  // find
  const got = await repo.findById(u.id);

  // transaction
  await db.transaction(async (tx) => {
    await tx.query('UPDATE users SET balance = balance - $1 WHERE id = $2', [10, u.id]);
    await tx.query('UPDATE users SET balance = balance + $1 WHERE id = $2', [10, 2]);
  });

  // run migrations
  const migrator = new Migrations(db, { dir: './migrations' });
  await migrator.up();
})();
