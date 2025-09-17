package redis

import (
	"context"
	"crypto/tls"
	"fmt"
	"time"

	redis "github.com/redis/go-redis/v9"
)

// Client is the exported Redis client interface (wrapper around go-redis Client/ClusterClient)
// Use the concrete type when necessary; otherwise rely on universal.Client interface methods.
var Client redis.Cmdable

// initConnection sets global Client according to Config.
// It also performs a ping with context timeout to validate connection.
func InitConnection(ctx context.Context, cfg *Config) error {
	if cfg == nil {
		return fmt.Errorf("nil redis config")
	}

	optsCommon := func(c redis.UniversalOptions) redis.UniversalOptions {
		c.Password = cfg.Password
		c.DB = cfg.DB
		c.PoolSize = cfg.PoolSize
		c.MinIdleConns = cfg.MinIdleConns
		c.DialTimeout = cfg.DialTimeout
		c.ReadTimeout = cfg.ReadTimeout
		c.WriteTimeout = cfg.WriteTimeout
		c.MaxRetries = cfg.MaxRetries
		c.MinRetryBackoff = cfg.RetryBackoff
		c.MaxRetryBackoff = cfg.RetryBackoff * 5
		// TLS config passed in if provided
		if cfg.TLSConfig != nil {
			c.TLSConfig = cfg.TLSConfig
		}
		// Each key will be prefixed by KeyPrefix (we'll apply that in helper functions)
		return c
	}

	var universalOpts redis.UniversalOptions
	switch cfg.Mode {
	case RedisModeCluster:
		universalOpts = redis.UniversalOptions{
			Addrs: cfg.Addrs,
			DB:    cfg.DB, // ignored by cluster; kept for compatibility
		}
	case RedisModeSentinel:
		universalOpts = redis.UniversalOptions{
			Addrs:      cfg.Addrs,
			MasterName: cfg.SentinelMaster,
			DB:         cfg.DB,
		}
	default: // single node
		universalOpts = redis.UniversalOptions{
			Addrs: []string{cfg.Addrs[0]},
			DB:    cfg.DB,
		}
	}

	universalOpts = optsCommon(universalOpts)

	// Create a UniversalClient which supports single/sentinel/cluster transparently.
	client := redis.NewUniversalClient(&universalOpts)

	// Ping to validate; use a short timeout
	pingCtx, cancel := context.WithTimeout(ctx, cfg.DialTimeout+time.Second)
	defer cancel()
	if err := client.Ping(pingCtx).Err(); err != nil {
		_ = client.Close()
		return fmt.Errorf("redis ping failed: %w", err)
	}

	// Assign to global Client (Cmdable interface)
	Client = client

	// Optionally: run a lightweight health check goroutine and metrics exporter here.
	return nil
}

// CloseConnection cleans up the underlying client.
func CloseConnection(ctx context.Context) error {
	if Client == nil {
		return nil
	}
	// Client may be *redis.Client or *redis.ClusterClient or *redis.Ring
	if c, ok := Client.(interface{ Close() error }); ok {
		return c.Close()
	}
	return nil
}
