// ========================================
// File: mysql.go
// ========================================
package database

import (
	"fmt"
)

// BuildMySQLDSN builds MySQL DSN string
func BuildMySQLDSN(user, password, host string, port int, dbname string, params string) string {
	return fmt.Sprintf("%s:%s@tcp(%s:%d)/%s?%s", user, password, host, port, dbname, params)
}
