package cockroachdb

import (
	"context"
	"database/sql"
	"time"
)

// Video represents a video entity
type Video struct {
	ID          string
	UserID      string
	Title       string
	Description string
	ViewCount   int
	UploadedAt  time.Time
}

// VideoRepository handles CRUD for videos
type VideoRepository struct {
	db *sql.DB
}

func NewVideoRepository(db *sql.DB) *VideoRepository {
	return &VideoRepository{db: db}
}

func (r *VideoRepository) Create(ctx context.Context, v *Video) error {
	query := `INSERT INTO videos (user_id, title, description) VALUES ($1, $2, $3) RETURNING id, uploaded_at`
	return r.db.QueryRowContext(ctx, query, v.UserID, v.Title, v.Description).
		Scan(&v.ID, &v.UploadedAt)
}

func (r *VideoRepository) IncrementViews(ctx context.Context, videoID string) error {
	query := `UPDATE videos SET view_count = view_count + 1 WHERE id=$1`
	_, err := r.db.ExecContext(ctx, query, videoID)
	return err
}
