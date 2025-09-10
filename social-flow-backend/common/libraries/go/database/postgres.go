// ========================================
// File: postgres.go
// ========================================
package database

import (
	"fmt"
)

// BuildPostgresDSN builds DSN string from config
func BuildPostgresDSN(host string, port int, user, password, dbname string, sslmode string) string {
	return fmt.Sprintf(
		"host=%s port=%d user=%s password=%s dbname=%s sslmode=%s",
		host, port, user, password, dbname, sslmode,
	)
}
