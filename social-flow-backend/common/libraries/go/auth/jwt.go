// ========================================
// File: jwt.go
// ========================================
package auth

import (
	"errors"
	"sync"
	"time"

	"github.com/golang-jwt/jwt/v5"
)

// JWTManager manages JWT token creation and validation
type JWTManager struct {
	secret     []byte
	blacklist  map[string]struct{}
	mu         sync.RWMutex
}

// NewJWTManager creates new JWT manager
func NewJWTManager(secret string) *JWTManager {
	return &JWTManager{
		secret:    []byte(secret),
		blacklist: make(map[string]struct{}),
	}
}

func (j *JWTManager) GenerateAccessToken(claims UserClaims, ttl time.Duration) (string, error) {
	token := jwt.NewWithClaims(jwt.SigningMethodHS256, jwt.MapClaims{
		"uid":      claims.UserID,
		"username": claims.Username,
		"roles":    claims.Roles,
		"exp":      time.Now().Add(ttl).Unix(),
	})
	return token.SignedString(j.secret)
}

func (j *JWTManager) GenerateRefreshToken(userID string, ttl time.Duration) (string, error) {
	token := jwt.NewWithClaims(jwt.SigningMethodHS256, jwt.MapClaims{
		"uid": userID,
		"typ": "refresh",
		"exp": time.Now().Add(ttl).Unix(),
	})
	return token.SignedString(j.secret)
}

func (j *JWTManager) ValidateToken(tokenStr string) (*UserClaims, error) {
	j.mu.RLock()
	if _, revoked := j.blacklist[tokenStr]; revoked {
		j.mu.RUnlock()
		return nil, errors.New("token revoked")
	}
	j.mu.RUnlock()

	token, err := jwt.Parse(tokenStr, func(t *jwt.Token) (interface{}, error) {
		return j.secret, nil
	})
	if err != nil || !token.Valid {
		return nil, errors.New("invalid token")
	}

	claims := token.Claims.(jwt.MapClaims)
	return &UserClaims{
		UserID:   claims["uid"].(string),
		Username: claims["username"].(string),
		Roles:    toStringSlice(claims["roles"]),
	}, nil
}

func (j *JWTManager) RevokeToken(tokenStr string) error {
	j.mu.Lock()
	defer j.mu.Unlock()
	j.blacklist[tokenStr] = struct{}{}
	return nil
}

func toStringSlice(input interface{}) []string {
	if arr, ok := input.([]interface{}); ok {
		out := []string{}
		for _, v := range arr {
			if s, ok := v.(string); ok {
				out = append(out, s)
			}
		}
		return out
	}
	return []string{}
}
