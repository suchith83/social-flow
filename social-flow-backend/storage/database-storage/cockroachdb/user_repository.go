package cockroachdb

import (
	"context"
	"database/sql"
	"time"
)

// User represents a user in the system
type User struct {
	ID        string
	Username  string
	Email     string
	CreatedAt time.Time
}

// UserRepository handles CRUD for users
type UserRepository struct {
	db *sql.DB
}

func NewUserRepository(db *sql.DB) *UserRepository {
	return &UserRepository{db: db}
}

func (r *UserRepository) Create(ctx context.Context, u *User) error {
	query := `INSERT INTO users (username, email) VALUES ($1, $2) RETURNING id, created_at`
	return r.db.QueryRowContext(ctx, query, u.Username, u.Email).Scan(&u.ID, &u.CreatedAt)
}

func (r *UserRepository) FindByEmail(ctx context.Context, email string) (*User, error) {
	query := `SELECT id, username, email, created_at FROM users WHERE email=$1`
	row := r.db.QueryRowContext(ctx, query, email)

	u := &User{}
	if err := row.Scan(&u.ID, &u.Username, &u.Email, &u.CreatedAt); err != nil {
		return nil, err
	}
	return u, nil
}
