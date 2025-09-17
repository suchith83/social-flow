package elasticsearch

import (
	"os"
	"strconv"
	"time"
)

// ESConfig holds configuration for Elasticsearch client
type ESConfig struct {
	Addresses []string
	Username  string
	Password  string
	Timeout   time.Duration
}

// LoadConfig loads Elasticsearch config from environment variables
func LoadConfig() *ESConfig {
	return &ESConfig{
		Addresses: []string{getEnv("ES_HOST", "http://localhost:9200")},
		Username:  getEnv("ES_USER", ""),
		Password:  getEnv("ES_PASSWORD", ""),
		Timeout:   time.Second * time.Duration(getEnvInt("ES_TIMEOUT", 10)),
	}
}

func getEnv(key, fallback string) string {
	if value := os.Getenv(key); value != "" {
		return value
	}
	return fallback
}

func getEnvInt(key string, fallback int) int {
	if value := os.Getenv(key); value != "" {
		if i, err := strconv.Atoi(value); err == nil {
			return i
		}
	}
	return fallback
}
