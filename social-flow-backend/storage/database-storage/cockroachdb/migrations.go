package cockroachdb

import (
	"database/sql"
	"log"
)

// RunMigrations applies schema migrations (idempotent)
func RunMigrations(db *sql.DB) error {
	statements := []string{
		`CREATE TABLE IF NOT EXISTS users (
			id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
			username STRING UNIQUE NOT NULL,
			email STRING UNIQUE NOT NULL,
			created_at TIMESTAMPTZ DEFAULT now()
		)`,
		`CREATE TABLE IF NOT EXISTS videos (
			id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
			user_id UUID REFERENCES users(id) ON DELETE CASCADE,
			title STRING NOT NULL,
			description STRING,
			view_count INT DEFAULT 0,
			uploaded_at TIMESTAMPTZ DEFAULT now()
		)`,
	}

	for _, stmt := range statements {
		_, err := db.Exec(stmt)
		if err != nil {
			return err
		}
	}
	log.Println("✅ CockroachDB migrations applied successfully.")
	return nil
}
