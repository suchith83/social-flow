package mongodb

import (
	"context"
	"errors"
	"fmt"
	"log"
	"strings"
	"time"

	"go.mongodb.org/mongo-driver/mongo"
	"go.mongodb.org/mongo-driver/mongo/options"
)

var Client *mongo.Client

// InitClient connects to MongoDB with the given configuration and sets the global Client.
// It also performs a ping to validate the connection, sets connection pool sizes, and returns the default database instance.
func InitClient(cfg *MongoConfig) (*mongo.Database, error) {
	if cfg == nil {
		return nil, errors.New("nil MongoConfig passed")
	}

	uri := cfg.ConnectionURI()
	opts := options.Client().ApplyURI(uri)

	// Connection pool / timeouts
	opts.SetConnectTimeout(cfg.ConnectTimeout)
	if cfg.MaxPoolSize > 0 {
		opts.SetMaxPoolSize(cfg.MaxPoolSize)
	}
	if cfg.MinPoolSize > 0 {
		opts.SetMinPoolSize(cfg.MinPoolSize)
	}
	if cfg.SocketTimeout > 0 {
		opts.SetSocketTimeout(cfg.SocketTimeout)
	}

	// Context for initial connection
	ctx, cancel := context.WithTimeout(context.Background(), cfg.ConnectTimeout+time.Second*5)
	defer cancel()

	client, err := mongo.Connect(ctx, opts)
	if err != nil {
		return nil, fmt.Errorf("mongo connect error: %w", err)
	}

	// Ping primary
	if err := client.Ping(ctx, nil); err != nil {
		// disconnect on ping failure
		_ = client.Disconnect(ctx)
		return nil, fmt.Errorf("mongo ping error: %w", err)
	}

	Client = client
	log.Println("✅ MongoDB client initialized and ping successful")

	db := client.Database(cfg.Database)
	return db, nil
}

// CloseClient disconnects the global client.
func CloseClient(ctx context.Context) error {
	if Client == nil {
		return nil
	}
	err := Client.Disconnect(ctx)
	if err != nil {
		return fmt.Errorf("error disconnecting mongo client: %w", err)
	}
	Client = nil
	return nil
}
