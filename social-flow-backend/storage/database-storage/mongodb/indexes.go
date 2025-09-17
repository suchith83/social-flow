package mongodb

import (
	"context"
	"log"
	"time"

	"go.mongodb.org/mongo-driver/bson"
	"go.mongodb.org/mongo-driver/mongo"
	"go.mongodb.org/mongo-driver/mongo/options"
)

// EnsureIndexes creates recommended indexes for users and videos collections.
// Run this at application start; it's idempotent.
func EnsureIndexes(ctx context.Context, db *mongo.Database) error {
	ctx, cancel := context.WithTimeout(ctx, 15*time.Second)
	defer cancel()

	// USERS indexes: unique on email & username for quick lookups and uniqueness enforcement
	users := db.Collection("users")
	userIndexModels := []mongo.IndexModel{
		{
			Keys: bson.D{{Key: "email", Value: 1}},
			Options: options.Index().SetUnique(true).SetName("idx_users_email"),
		},
		{
			Keys: bson.D{{Key: "username", Value: 1}},
			Options: options.Index().SetUnique(true).SetName("idx_users_username"),
		},
		{
			Keys:    bson.D{{Key: "created_at", Value: -1}},
			Options: options.Index().SetName("idx_users_created_at"),
		},
	}
	if _, err := users.Indexes().CreateMany(ctx, userIndexModels); err != nil {
		log.Printf("error creating user indexes: %v", err)
		return err
	}
	log.Println("✅ user indexes ensured")

	// VIDEOS indexes: text index for search, compound indexes for relevant queries
	videos := db.Collection("videos")
	videoIndexModels := []mongo.IndexModel{
		{
			// text index on title+description+tags for full-text search
			Keys: bson.D{{Key: "title", Value: "text"}, {Key: "description", Value: "text"}, {Key: "tags", Value: "text"}},
			Options: options.Index().SetDefaultLanguage("english").SetName("idx_videos_text"),
		},
		{
			Keys:    bson.D{{Key: "user_id", Value: 1}, {Key: "uploaded_at", Value: -1}},
			Options: options.Index().SetName("idx_videos_user_uploaded"),
		},
		{
			Keys:    bson.D{{Key: "view_count", Value: -1}},
			Options: options.Index().SetName("idx_videos_popular"),
		},
	}
	if _, err := videos.Indexes().CreateMany(ctx, videoIndexModels); err != nil {
		log.Printf("error creating video indexes: %v", err)
		return err
	}
	log.Println("✅ video indexes ensured")

	return nil
}
