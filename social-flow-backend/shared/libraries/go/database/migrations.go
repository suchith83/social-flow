// ========================================
// File: migrations.go
// ========================================
package database

import (
	"database/sql"
	"fmt"

	"github.com/pressly/goose/v3"
)

// RunMigrations runs goose migrations on database
func RunMigrations(db *sql.DB, migrationDir string) error {
	if err := goose.SetDialect("postgres"); err != nil {
		return err
	}
	if err := goose.Up(db, migrationDir); err != nil {
		return fmt.Errorf("failed migrations: %w", err)
	}
	return nil
}
