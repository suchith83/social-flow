package redis

import (
	"crypto/tls"
	"os"
	"strconv"
	"strings"
	"time"
)

// RedisMode - supported modes
type RedisMode string

const (
	RedisModeSingle   RedisMode = "single"
	RedisModeSentinel RedisMode = "sentinel"
	RedisModeCluster  RedisMode = "cluster"
)

// Config contains configuration to connect to Redis in various modes.
type Config struct {
	Mode           RedisMode
	Addrs          []string      // single: ["host:port"], cluster: list, sentinel: sentinel addrs
	Password       string
	DB             int           // database number for single/sentinel (not used in cluster)
	SentinelMaster string        // sentinel master name (for sentinel mode)
	PoolSize       int           // max connections
	MinIdleConns   int
	DialTimeout    time.Duration
	ReadTimeout    time.Duration
	WriteTimeout   time.Duration
	TLSConfig      *tls.Config   // optional TLS
	MaxRetries     int
	RetryBackoff   time.Duration
	KeyPrefix      string        // optional prefix for keys (multitenancy)
}

// LoadConfigFromEnv is a convenience loader from environment variables.
// You can replace this with Vault/SSM/Config provider in prod.
func LoadConfigFromEnv() *Config {
	mode := RedisMode(getEnv("REDIS_MODE", "single"))
	addrs := splitCSV(getEnv("REDIS_ADDRS", "localhost:6379"))
	return &Config{
		Mode:           mode,
		Addrs:          addrs,
		Password:       getEnv("REDIS_PASSWORD", ""),
		DB:             getEnvInt("REDIS_DB", 0),
		SentinelMaster: getEnv("REDIS_SENTINEL_MASTER", ""),
		PoolSize:       getEnvInt("REDIS_POOL_SIZE", 50),
		MinIdleConns:   getEnvInt("REDIS_MIN_IDLE", 10),
		DialTimeout:    time.Duration(getEnvInt("REDIS_DIAL_TIMEOUT", 5)) * time.Second,
		ReadTimeout:    time.Duration(getEnvInt("REDIS_READ_TIMEOUT", 3)) * time.Second,
		WriteTimeout:   time.Duration(getEnvInt("REDIS_WRITE_TIMEOUT", 3)) * time.Second,
		MaxRetries:     getEnvInt("REDIS_MAX_RETRIES", 3),
		RetryBackoff:   time.Duration(getEnvInt("REDIS_RETRY_BACKOFF_MS", 100)) * time.Millisecond,
		KeyPrefix:      getEnv("REDIS_KEY_PREFIX", ""),
		// TLSConfig left nil; load or construct externally if needed
	}
}

// Helpers
func getEnv(key, fallback string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return fallback
}

func getEnvInt(key string, fallback int) int {
	if v := os.Getenv(key); v != "" {
		if i, err := strconv.Atoi(v); err == nil {
			return i
		}
	}
	return fallback
}

func splitCSV(s string) []string {
	var out []string
	for _, p := range strings.Split(s, ",") {
		p = strings.TrimSpace(p)
		if p != "" {
			out = append(out, p)
		}
	}
	return out
}
