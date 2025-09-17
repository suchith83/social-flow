package mongodb

import "time"

// UserModel represents a user document in MongoDB.
type UserModel struct {
	ID        string    `bson:"_id,omitempty" json:"id"`
	Username  string    `bson:"username" json:"username"`
	Email     string    `bson:"email" json:"email"`
	Bio       string    `bson:"bio,omitempty" json:"bio,omitempty"`
	AvatarURL string    `bson:"avatar_url,omitempty" json:"avatar_url,omitempty"`
	CreatedAt time.Time `bson:"created_at" json:"created_at"`
	UpdatedAt time.Time `bson:"updated_at" json:"updated_at"`
	// Add other common fields like flags, roles, etc.
}

// VideoModel represents a video document in MongoDB.
type VideoModel struct {
	ID          string    `bson:"_id,omitempty" json:"id"`
	UserID      string    `bson:"user_id" json:"user_id"` // uploader
	Title       string    `bson:"title" json:"title"`
	Description string    `bson:"description,omitempty" json:"description,omitempty"`
	Tags        []string  `bson:"tags,omitempty" json:"tags,omitempty"`
	ViewCount   int64     `bson:"view_count" json:"view_count"`
	LikeCount   int64     `bson:"like_count" json:"like_count"`
	DurationSec int       `bson:"duration_sec,omitempty" json:"duration_sec,omitempty"`
	UploadedAt  time.Time `bson:"uploaded_at" json:"uploaded_at"`
	UpdatedAt   time.Time `bson:"updated_at" json:"updated_at"`
	// Denormalized fields for fast reads: username, avatar, etc., can be added if desired.
}
