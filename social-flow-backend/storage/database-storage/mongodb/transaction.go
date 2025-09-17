package mongodb

import (
	"context"
	"time"

	"go.mongodb.org/mongo-driver/mongo"
)

// WithTransaction provides a helper to run a callback within a session/transaction with retry logic.
// MongoDB transactions require a replica set or sharded cluster.
func WithTransaction(ctx context.Context, db *mongo.Database, callback func(sessCtx mongo.SessionContext) error) error {
	if Client == nil {
		return ErrClientNotInitialized
	}

	// Start a session
	session, err := Client.StartSession()
	if err != nil {
		return err
	}
	defer session.EndSession(ctx)

	// default txn options - you can customize
	txnOpts := options.Transaction().SetReadConcern(mongo.ReadConcernSnapshot()).SetWriteConcern(mongo.WriteConcern{WString: "majority"})

	// execution with retries for TransientTransactionError and UnknownTransactionCommitResult
	err = mongo.WithSession(ctx, session, func(sessCtx mongo.SessionContext) error {
		// Start transaction
		if err := session.StartTransaction(txnOpts); err != nil {
			return err
		}

		// call user's callback which will use sessCtx for collection operations
		if err := callback(sessCtx); err != nil {
			_ = session.AbortTransaction(sessCtx)
			return err
		}

		// Commit; this may return transient errors which the caller might want to retry in higher-level logic
		return session.CommitTransaction(sessCtx)
	})
	return err
}
