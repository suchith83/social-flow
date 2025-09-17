package cockroachdb

import (
	"context"
	"database/sql"
	"log"
	"time"

	_ "github.com/lib/pq" // CockroachDB uses Postgres driver
)

var DB *sql.DB

// InitConnection initializes a global CockroachDB connection
func InitConnection(cfg *DBConfig) error {
	db, err := sql.Open("postgres", cfg.DSN())
	if err != nil {
		return err
	}

	// Configure connection pool
	db.SetMaxOpenConns(cfg.MaxConns)
	db.SetMaxIdleConns(cfg.MaxIdleConns)
	db.SetConnMaxLifetime(time.Hour)

	// Test connection
	ctx, cancel := context.WithTimeout(context.Background(), cfg.ConnTimeout)
	defer cancel()

	if err := db.PingContext(ctx); err != nil {
		return err
	}

	DB = db
	log.Println("✅ CockroachDB connection established successfully.")
	return nil
}
