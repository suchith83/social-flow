// ========================================
// File: cache.go
// ========================================
package database

import (
	"context"
	"time"

	"github.com/redis/go-redis/v9"
)

// Cache wraps Redis for caching
type Cache struct {
	client *redis.Client
}

// NewCache initializes Redis cache
func NewCache(addr, password string, db int) *Cache {
	client := redis.NewClient(&redis.Options{
		Addr:     addr,
		Password: password,
		DB:       db,
	})
	return &Cache{client: client}
}

func (c *Cache) Set(ctx context.Context, key string, value interface{}, ttl time.Duration) error {
	return c.client.Set(ctx, key, value, ttl).Err()
}

func (c *Cache) Get(ctx context.Context, key string) (string, error) {
	return c.client.Get(ctx, key).Result()
}

func (c *Cache) Delete(ctx context.Context, key string) error {
	return c.client.Del(ctx, key).Err()
}
