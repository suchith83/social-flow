package cockroachdb

import (
	"context"
	"database/sql"
	"time"
)

// WithTransaction retries a transaction in case of serialization errors
func WithTransaction(ctx context.Context, db *sql.DB, fn func(*sql.Tx) error) error {
	var err error
	for i := 0; i < 5; i++ {
		tx, beginErr := db.BeginTx(ctx, nil)
		if beginErr != nil {
			return beginErr
		}

		err = fn(tx)
		if err != nil {
			_ = tx.Rollback()
			continue
		}

		commitErr := tx.Commit()
		if commitErr != nil {
			// Retry if commit fails due to serialization
			if commitErr == sql.ErrTxDone {
				time.Sleep(time.Duration(i+1) * 100 * time.Millisecond)
				continue
			}
			return commitErr
		}
		return nil
	}
	return err
}
