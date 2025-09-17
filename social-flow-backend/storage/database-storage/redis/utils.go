package redis

import (
	"context"
	"fmt"
	"time"

	redis "github.com/redis/go-redis/v9"
)

// AcquireLock acquires a distributed lock with expiration using SET NX and token.
// Returns token which must be used to release the lock.
func AcquireLock(ctx context.Context, key string, ttl time.Duration, prefix string) (string, error) {
	if Client == nil {
		return "", fmt.Errorf("redis client not initialized")
	}
	token := uuid.New().String()
	ok, err := Client.SetNX(ctx, prefixedKey(prefix, key), token, ttl).Result()
	if err != nil {
		return "", err
	}
	if !ok {
		return "", fmt.Errorf("lock not acquired")
	}
	return token, nil
}

// ReleaseLock releases the lock only if token matches (safe unlock) via Lua script.
var releaseLockScript = redis.NewScript(`
if redis.call("GET", KEYS[1]) == ARGV[1] then
  return redis.call("DEL", KEYS[1])
else
  return 0
end
`)

func ReleaseLock(ctx context.Context, key, token, prefix string) (bool, error) {
	if Client == nil {
		return false, fmt.Errorf("redis client not initialized")
	}
	res, err := releaseLockScript.Run(ctx, Client, []string{prefixedKey(prefix, key)}, token).Result()
	if err != nil {
		return false, err
	}
	n, ok := res.(int64)
	return ok && n > 0, nil
}

// HealthCheck pings redis quickly and returns error if unhealthy
func HealthCheck(ctx context.Context) error {
	if Client == nil {
		return fmt.Errorf("redis client not initialized")
	}
	return Client.Ping(ctx).Err()
}
