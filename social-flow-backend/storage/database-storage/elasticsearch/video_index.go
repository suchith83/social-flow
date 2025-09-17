package elasticsearch

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"time"

	"github.com/google/uuid"
)

const VideoIndex = "videos"

type Video struct {
	ID          string    `json:"id"`
	UserID      string    `json:"user_id"`
	Title       string    `json:"title"`
	Description string    `json:"description"`
	Tags        []string  `json:"tags"`
	ViewCount   int       `json:"view_count"`
	UploadedAt  time.Time `json:"uploaded_at"`
}

// VideoRepository handles indexing and searching videos
type VideoRepository struct{}

func NewVideoRepository() *VideoRepository {
	return &VideoRepository{}
}

func (r *VideoRepository) IndexVideo(ctx context.Context, v *Video) error {
	if v.ID == "" {
		v.ID = uuid.New().String()
	}
	body, _ := json.Marshal(v)

	res, err := Client.Index(
		VideoIndex,
		bytes.NewReader(body),
		Client.Index.WithDocumentID(v.ID),
		Client.Index.WithContext(ctx),
	)
	if err != nil {
		return err
	}
	defer res.Body.Close()

	if res.IsError() {
		return fmt.Errorf("failed to index video: %s", res.String())
	}
	return nil
}
