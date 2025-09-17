package elasticsearch

import (
	"context"
	"log"
	"time"

	es "github.com/elastic/go-elasticsearch/v8"
)

var Client *es.Client

// InitConnection initializes Elasticsearch client
func InitConnection(cfg *ESConfig) error {
	esCfg := es.Config{
		Addresses: cfg.Addresses,
		Username:  cfg.Username,
		Password:  cfg.Password,
	}

	client, err := es.NewClient(esCfg)
	if err != nil {
		return err
	}

	// Test connection with a ping
	ctx, cancel := context.WithTimeout(context.Background(), cfg.Timeout)
	defer cancel()

	res, err := client.Ping(client.Ping.WithContext(ctx))
	if err != nil {
		return err
	}
	defer res.Body.Close()

	if res.IsError() {
		log.Printf("⚠️ Elasticsearch ping failed: %s", res.String())
	} else {
		log.Println("✅ Elasticsearch connection established successfully.")
	}

	Client = client
	return nil
}
