package mongodb

import (
	"context"
	"errors"
	"time"

	"go.mongodb.org/mongo-driver/bson"
	"go.mongodb.org/mongo-driver/mongo"
	"go.mongodb.org/mongo-driver/mongo/options"
)

// UserRepository encapsulates user-related DB operations.
type UserRepository struct {
	col *mongo.Collection
}

// NewUserRepository returns a new repository bound to the provided database.
func NewUserRepository(db *mongo.Database) *UserRepository {
	return &UserRepository{
		col: db.Collection("users"),
	}
}

// Create inserts a new user. It enforces uniqueness constraints (email/username) via indexes.
func (r *UserRepository) Create(ctx context.Context, user *UserModel) error {
	if user == nil {
		return errors.New("user is nil")
	}
	now := time.Now().UTC()
	user.CreatedAt = now
	user.UpdatedAt = now

	_, err := r.col.InsertOne(ctx, user)
	if mongo.IsDuplicateKeyError(err) {
		// return a domain-friendly error (your app might wrap this)
		return ErrConflict
	}
	return err
}

// FindByEmail finds user by email
func (r *UserRepository) FindByEmail(ctx context.Context, email string) (*UserModel, error) {
	filter := bson.M{"email": email}
	var u UserModel
	err := r.col.FindOne(ctx, filter).Decode(&u)
	if err == mongo.ErrNoDocuments {
		return nil, ErrNotFound
	}
	return &u, err
}

// FindByID finds user by _id
func (r *UserRepository) FindByID(ctx context.Context, id string) (*UserModel, error) {
	filter := bson.M{"_id": id}
	var u UserModel
	err := r.col.FindOne(ctx, filter).Decode(&u)
	if err == mongo.ErrNoDocuments {
		return nil, ErrNotFound
	}
	return &u, err
}

// Update updates a user's mutable fields and returns the updated document
func (r *UserRepository) Update(ctx context.Context, id string, update bson.M) (*UserModel, error) {
	update["updated_at"] = time.Now().UTC()
	opts := options.FindOneAndUpdate().SetReturnDocument(options.After)
	var updated UserModel
	err := r.col.FindOneAndUpdate(ctx, bson.M{"_id": id}, bson.M{"$set": update}, opts).Decode(&updated)
	if err == mongo.ErrNoDocuments {
		return nil, ErrNotFound
	}
	return &updated, err
}

// Delete removes a user document. Consider soft-delete for production.
func (r *UserRepository) Delete(ctx context.Context, id string) error {
	res, err := r.col.DeleteOne(ctx, bson.M{"_id": id})
	if err != nil {
		return err
	}
	if res.DeletedCount == 0 {
		return ErrNotFound
	}
	return nil
}
