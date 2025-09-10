// ========================================
// File: transaction.go
// ========================================
package database

import (
	"context"
	"database/sql"
	"errors"
)

// Transaction wraps sql.Tx with context
type Transaction struct {
	tx *sql.Tx
}

// Begin starts a transaction
func (d *Database) Begin(ctx context.Context) (*Transaction, error) {
	tx, err := d.DB.BeginTx(ctx, nil)
	if err != nil {
		return nil, err
	}
	return &Transaction{tx: tx}, nil
}

// Exec executes a statement inside transaction
func (t *Transaction) Exec(ctx context.Context, query string, args ...any) (sql.Result, error) {
	if t.tx == nil {
		return nil, errors.New("transaction not initialized")
	}
	return t.tx.ExecContext(ctx, query, args...)
}

// Commit commits transaction
func (t *Transaction) Commit() error {
	return t.tx.Commit()
}

// Rollback aborts transaction
func (t *Transaction) Rollback() error {
	return t.tx.Rollback()
}
