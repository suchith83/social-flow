package cockroachdb

import (
	"log"
	"os"
	"strconv"
)

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
		log.Printf("⚠️ Invalid int for %s, using fallback %d\n", key, fallback)
	}
	return fallback
}
