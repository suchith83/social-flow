package elasticsearch

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"time"

	"github.com/google/uuid"
)

const UserIndex = "users"

type User struct {
	ID        string    `json:"id"`
	Username  string    `json:"username"`
	Email     string    `json:"email"`
	JoinedAt  time.Time `json:"joined_at"`
}

// UserRepository handles indexing and searching users
type UserRepository struct{}

func NewUserRepository() *UserRepository {
	return &UserRepository{}
}

func (r *UserRepository) IndexUser(ctx context.Context, u *User) error {
	if u.ID == "" {
		u.ID = uuid.New().String()
	}
	body, _ := json.Marshal(u)

	res, err := Client.Index(
		UserIndex,
		bytes.NewReader(body),
		Client.Index.WithDocumentID(u.ID),
		Client.Index.WithContext(ctx),
	)
	if err != nil {
		return err
	}
	defer res.Body.Close()

	if res.IsError() {
		return fmt.Errorf("failed to index user: %s", res.String())
	}
	return nil
}
