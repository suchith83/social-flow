// ========================================
// File: auth.go
// ========================================
package auth

import (
	"context"
	"errors"
	"time"
)

// UserClaims represents custom claims for a user
type UserClaims struct {
	UserID   string   `json:"uid"`
	Username string   `json:"username"`
	Roles    []string `json:"roles"`
}

// AuthManager defines the main authentication interface
type AuthManager interface {
	// Password operations
	HashPassword(password string) (string, error)
	VerifyPassword(password, hash string) bool

	// JWT operations
	GenerateToken(claims UserClaims, ttl time.Duration) (string, error)
	GenerateRefreshToken(userID string, ttl time.Duration) (string, error)
	ValidateToken(token string) (*UserClaims, error)
	RevokeToken(token string) error

	// Session operations
	SaveSession(ctx context.Context, sessionID string, data map[string]interface{}, ttl time.Duration) error
	GetSession(ctx context.Context, sessionID string) (map[string]interface{}, error)
	DestroySession(ctx context.Context, sessionID string) error

	// OAuth login
	GetOAuthLoginURL(provider, state string) (string, error)
	ExchangeOAuthCode(ctx context.Context, provider, code string) (*UserClaims, error)
}

// DefaultAuthManager is a concrete implementation
type DefaultAuthManager struct {
	pwHasher   PasswordHasher
	jwtManager *JWTManager
	sessions   SessionStore
	oauth      *OAuthManager
}

// NewAuthManager creates an instance of DefaultAuthManager
func NewAuthManager(secret string, store SessionStore, oauthCfg *OAuthConfig) (*DefaultAuthManager, error) {
	jwtMgr := NewJWTManager(secret)
	pw := NewArgon2Hasher()
	oauthMgr := NewOAuthManager(oauthCfg)

	return &DefaultAuthManager{
		pwHasher:   pw,
		jwtManager: jwtMgr,
		sessions:   store,
		oauth:      oauthMgr,
	}, nil
}

// Implement AuthManager methods
func (a *DefaultAuthManager) HashPassword(password string) (string, error) {
	return a.pwHasher.Hash(password)
}

func (a *DefaultAuthManager) VerifyPassword(password, hash string) bool {
	return a.pwHasher.Verify(password, hash)
}

func (a *DefaultAuthManager) GenerateToken(claims UserClaims, ttl time.Duration) (string, error) {
	return a.jwtManager.GenerateAccessToken(claims, ttl)
}

func (a *DefaultAuthManager) GenerateRefreshToken(userID string, ttl time.Duration) (string, error) {
	return a.jwtManager.GenerateRefreshToken(userID, ttl)
}

func (a *DefaultAuthManager) ValidateToken(token string) (*UserClaims, error) {
	return a.jwtManager.ValidateToken(token)
}

func (a *DefaultAuthManager) RevokeToken(token string) error {
	return a.jwtManager.RevokeToken(token)
}

func (a *DefaultAuthManager) SaveSession(ctx context.Context, sessionID string, data map[string]interface{}, ttl time.Duration) error {
	return a.sessions.Save(ctx, sessionID, data, ttl)
}

func (a *DefaultAuthManager) GetSession(ctx context.Context, sessionID string) (map[string]interface{}, error) {
	return a.sessions.Get(ctx, sessionID)
}

func (a *DefaultAuthManager) DestroySession(ctx context.Context, sessionID string) error {
	return a.sessions.Destroy(ctx, sessionID)
}

func (a *DefaultAuthManager) GetOAuthLoginURL(provider, state string) (string, error) {
	if a.oauth == nil {
		return "", errors.New("OAuth not configured")
	}
	return a.oauth.GetLoginURL(provider, state)
}

func (a *DefaultAuthManager) ExchangeOAuthCode(ctx context.Context, provider, code string) (*UserClaims, error) {
	if a.oauth == nil {
		return nil, errors.New("OAuth not configured")
	}
	return a.oauth.ExchangeCode(ctx, provider, code)
}
