package elasticsearch

import (
	"log"
)

// LogError logs Elasticsearch errors
func LogError(err error) {
	if err != nil {
		log.Printf("❌ Elasticsearch error: %v", err)
	}
}
