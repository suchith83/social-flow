// ========================================
// File: sqlite.go
// ========================================
package database

import "fmt"

// BuildSQLiteDSN builds SQLite DSN string (file path)
func BuildSQLiteDSN(filepath string, mode string) string {
	return fmt.Sprintf("file:%s?cache=shared&mode=%s", filepath, mode)
}
