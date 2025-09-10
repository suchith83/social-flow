// ========================================
// File: health.go
// ========================================
package database

import "context"

// HealthCheck verifies DB and cache connections
func HealthCheck(ctx context.Context, db *Database, cache *Cache) error {
	if db != nil {
		if err := db.Ping(ctx); err != nil {
			return err
		}
	}
	if cache != nil {
		if err := cache.client.Ping(ctx).Err(); err != nil {
			return err
		}
	}
	return nil
}
