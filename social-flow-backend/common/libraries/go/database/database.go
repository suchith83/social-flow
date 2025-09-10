// ========================================
// File: database.go
// ========================================
package database

import (
	"context"
	"database/sql"
	"errors"
	"fmt"
	"time"

	_ "github.com/go-sql-driver/mysql"    // MySQL driver
	_ "github.com/lib/pq"                 // Postgres driver
	_ "github.com/mattn/go-sqlite3"       // SQLite driver

	"github.com/sony/gobreaker"
)

type DBType string

const (
	Postgres DBType = "postgres"
	MySQL    DBType = "mysql"
	SQLite   DBType = "sqlite"
)

// Config holds database configuration
type Config struct {
	Type       DBType
	DSN        string
	MaxOpen    int
	MaxIdle    int
	MaxLife    time.Duration
	EnableCB   bool // Circuit breaker
}

// Database wraps sql.DB with utilities
type Database struct {
	DB        *sql.DB
	breaker   *gobreaker.CircuitBreaker
	dbType    DBType
}

// NewDatabase initializes a new DB connection with pooling
func NewDatabase(cfg Config) (*Database, error) {
	db, err := sql.Open(string(cfg.Type), cfg.DSN)
	if err != nil {
		return nil, fmt.Errorf("failed to open DB: %w", err)
	}

	db.SetMaxOpenConns(cfg.MaxOpen)
	db.SetMaxIdleConns(cfg.MaxIdle)
	db.SetConnMaxLifetime(cfg.MaxLife)

	// Circuit breaker (optional)
	var breaker *gobreaker.CircuitBreaker
	if cfg.EnableCB {
		breaker = gobreaker.NewCircuitBreaker(gobreaker.Settings{
			Name:        string(cfg.Type) + "-db",
			MaxRequests: 5,
			Timeout:     5 * time.Second,
			Interval:    10 * time.Second,
		})
	}

	return &Database{
		DB:      db,
		breaker: breaker,
		dbType:  cfg.Type,
	}, nil
}

// QueryContext with optional circuit breaker
func (d *Database) QueryContext(ctx context.Context, query string, args ...any) (*sql.Rows, error) {
	if d.breaker != nil {
		res, err := d.breaker.Execute(func() (interface{}, error) {
			return d.DB.QueryContext(ctx, query, args...)
		})
		if err != nil {
			return nil, err
		}
		return res.(*sql.Rows), nil
	}
	return d.DB.QueryContext(ctx, query, args...)
}

// ExecContext with optional circuit breaker
func (d *Database) ExecContext(ctx context.Context, query string, args ...any) (sql.Result, error) {
	if d.breaker != nil {
		res, err := d.breaker.Execute(func() (interface{}, error) {
			return d.DB.ExecContext(ctx, query, args...)
		})
		if err != nil {
			return nil, err
		}
		return res.(sql.Result), nil
	}
	return d.DB.ExecContext(ctx, query, args...)
}

// Ping tests connectivity
func (d *Database) Ping(ctx context.Context) error {
	return d.DB.PingContext(ctx)
}
