package redis

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"time"

	"github.com/google/uuid"
	redis "github.com/redis/go-redis/v9"
)

// ErrCacheMiss exported
var ErrCacheMiss = redis.Nil

// prefixedKey returns key with optional configured prefix to avoid collisions
func prefixedKey(prefix, k string) string {
	if prefix == "" {
		return k
	}
	return prefix + ":" + k
}

// SetJSON stores a marshalled JSON value with TTL.
// Value is marshalled using encoding/json.
func SetJSON(ctx context.Context, key string, value interface{}, ttl time.Duration, prefix string) error {
	if Client == nil {
		return fmt.Errorf("redis client not initialized")
	}
	k := prefixedKey(prefix, key)
	b, err := json.Marshal(value)
	if err != nil {
		return err
	}
	return Client.Set(ctx, k, b, ttl).Err()
}

// GetJSON retrieves and unmarshals a JSON value.
func GetJSON(ctx context.Context, key string, dest interface{}, prefix string) error {
	if Client == nil {
		return fmt.Errorf("redis client not initialized")
	}
	k := prefixedKey(prefix, key)
	b, err := Client.Get(ctx, k).Bytes()
	if err != nil {
		return err
	}
	return json.Unmarshal(b, dest)
}

// Memoize executes fn to compute a value if not present in cache.
// It uses a simple "single flight" approach via SETNX lock to avoid stampede.
// ttl is cache TTL; lockTTL is the time to hold the lock while computing the value.
func Memoize[T any](ctx context.Context, key string, ttl, lockTTL time.Duration, prefix string, fn func() (T, error)) (T, error) {
	var zero T
	if Client == nil {
		return zero, fmt.Errorf("redis client not initialized")
	}
	k := prefixedKey(prefix, key)
	// Try to get existing value
	var data []byte
	val, err := Client.Get(ctx, k).Bytes()
	if err == nil {
		if err := json.Unmarshal(val, &data); err == nil {
			// Unmarshal into T
			var out T
			if err := json.Unmarshal(val, &out); err == nil {
				return out, nil
			}
		}
	}
	if err != nil && err != redis.Nil {
		// Non-miss error
		return zero, err
	}

	// Acquire computation lock using a unique token
	lockKey := k + ":lock"
	token := uuid.New().String()
	acquired, err := Client.SetNX(ctx, lockKey, token, lockTTL).Result()
	if err != nil {
		return zero, err
	}

	if !acquired {
		// Another worker is computing. Simple strategy: wait with backoff for value.
		waitCtx, cancel := context.WithTimeout(ctx, lockTTL+time.Second)
		defer cancel()
		ticker := time.NewTicker(50 * time.Millisecond)
		defer ticker.Stop()
		for {
			select {
			case <-waitCtx.Done():
				return zero, fmt.Errorf("timed out waiting for cache for key %s", key)
			case <-ticker.C:
				val, err := Client.Get(ctx, k).Bytes()
				if err == nil {
					var out T
					if err := json.Unmarshal(val, &out); err == nil {
						return out, nil
					}
					return zero, err
				}
				if err != redis.Nil {
					// real error
					return zero, err
				}
			}
		}
	}

	// We hold the lock and must compute & set the value, then release lock
	defer Client.Del(ctx, lockKey)

	result, fnErr := fn()
	if fnErr != nil {
		var zero T
		return zero, fnErr
	}
	// Marshal and set
	bs, err := json.Marshal(result)
	if err != nil {
		return result, err
	}
	if err := Client.Set(ctx, k, bs, ttl).Err(); err != nil {
		// best-effort — still return computed value
		return result, err
	}
	return result, nil
}

// Delete a key
func DeleteKey(ctx context.Context, key, prefix string) error {
	if Client == nil {
		return fmt.Errorf("redis client not initialized")
	}
	return Client.Del(ctx, prefixedKey(prefix, key)).Err()
}
