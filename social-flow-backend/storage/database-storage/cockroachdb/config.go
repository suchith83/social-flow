package cockroachdb

import (
	"fmt"
	"os"
	"time"
)

// DBConfig holds configuration for CockroachDB connection
type DBConfig struct {
	Host         string
	Port         string
	User         string
	Password     string
	Database     string
	SSLMode      string
	MaxConns     int
	MaxIdleConns int
	ConnTimeout  time.Duration
}

// LoadConfig loads configuration from environment variables
func LoadConfig() *DBConfig {
	return &DBConfig{
		Host:         getEnv("CRDB_HOST", "localhost"),
		Port:         getEnv("CRDB_PORT", "26257"),
		User:         getEnv("CRDB_USER", "root"),
		Password:     getEnv("CRDB_PASSWORD", ""),
		Database:     getEnv("CRDB_DATABASE", "socialflow"),
		SSLMode:      getEnv("CRDB_SSL_MODE", "disable"), // disable for local dev
		MaxConns:     getEnvInt("CRDB_MAX_CONNS", 50),
		MaxIdleConns: getEnvInt("CRDB_MAX_IDLE_CONNS", 10),
		ConnTimeout:  time.Second * time.Duration(getEnvInt("CRDB_CONN_TIMEOUT", 5)),
	}
}

func (c *DBConfig) DSN() string {
	return fmt.Sprintf(
		"postgresql://%s:%s@%s:%s/%s?sslmode=%s&connect_timeout=%d",
		c.User, c.Password, c.Host, c.Port, c.Database, c.SSLMode, int(c.ConnTimeout.Seconds()),
	)
}
