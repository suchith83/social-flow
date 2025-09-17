package mongodb

import (
	"context"
	"errors"
	"time"

	"go.mongodb.org/mongo-driver/bson"
	"go.mongodb.org/mongo-driver/mongo"
	"go.mongodb.org/mongo-driver/mongo/options"
)

// VideoRepository encapsulates video-related DB operations.
type VideoRepository struct {
	col *mongo.Collection
	db  *mongo.Database
}

// NewVideoRepository returns a new repository bound to the provided database.
func NewVideoRepository(db *mongo.Database) *VideoRepository {
	return &VideoRepository{
		col: db.Collection("videos"),
		db:  db,
	}
}

// Create inserts a new video document.
func (r *VideoRepository) Create(ctx context.Context, v *VideoModel) error {
	if v == nil {
		return errors.New("video is nil")
	}
	now := time.Now().UTC()
	v.UploadedAt = now
	v.UpdatedAt = now
	if v.ViewCount == 0 {
		v.ViewCount = 0
	}
	_, err := r.col.InsertOne(ctx, v)
	return err
}

// FindByID retrieves a video by id.
func (r *VideoRepository) FindByID(ctx context.Context, id string) (*VideoModel, error) {
	var v VideoModel
	err := r.col.FindOne(ctx, bson.M{"_id": id}).Decode(&v)
	if err == mongo.ErrNoDocuments {
		return nil, ErrNotFound
	}
	return &v, err
}

// IncrementView increments the view count in an atomic update.
func (r *VideoRepository) IncrementView(ctx context.Context, id string) error {
	update := bson.M{"$inc": bson.M{"view_count": int64(1)}, "$set": bson.M{"updated_at": time.Now().UTC()}}
	res, err := r.col.UpdateOne(ctx, bson.M{"_id": id}, update)
	if err != nil {
		return err
	}
	if res.MatchedCount == 0 {
		return ErrNotFound
	}
	return nil
}

// FindByUser returns paginated videos uploaded by a user.
func (r *VideoRepository) FindByUser(ctx context.Context, userID string, limit int64, after time.Time) ([]*VideoModel, error) {
	filter := bson.M{"user_id": userID}
	if !after.IsZero() {
		filter["uploaded_at"] = bson.M{"$lt": after}
	}
	opts := options.Find().SetSort(bson.D{{Key: "uploaded_at", Value: -1}}).SetLimit(limit)
	cursor, err := r.col.Find(ctx, filter, opts)
	if err != nil {
		return nil, err
	}
	defer cursor.Close(ctx)

	out := []*VideoModel{}
	for cursor.Next(ctx) {
		var v VideoModel
		if err := cursor.Decode(&v); err != nil {
			return nil, err
		}
		out = append(out, &v)
	}
	return out, cursor.Err()
}

// SearchText performs a text search on title/description/tags using MongoDB text index.
func (r *VideoRepository) SearchText(ctx context.Context, q string, limit int64) ([]*VideoModel, error) {
	filter := bson.M{"$text": bson.M{"$search": q}}
	opts := options.Find().SetSort(bson.D{{Key: "score", Value: bson.M{"$meta": "textScore"}}}).SetProjection(bson.M{"score": bson.M{"$meta": "textScore"}}).SetLimit(limit)
	cursor, err := r.col.Find(ctx, filter, opts)
	if err != nil {
		return nil, err
	}
	defer cursor.Close(ctx)

	out := []*VideoModel{}
	for cursor.Next(ctx) {
		var v VideoModel
		if err := cursor.Decode(&v); err != nil {
			return nil, err
		}
		out = append(out, &v)
	}
	return out, cursor.Err()
}
