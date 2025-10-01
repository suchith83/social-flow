package security

import (
	"crypto/sha256"
	"encoding/hex"
	"golang.org/x/crypto/argon2"
	"golang.org/x/crypto/bcrypt"
)

// HashSHA256 returns SHA-256 hash of input as hex.
func HashSHA256(input string) string {
	h := sha256.Sum256([]byte(input))
	return hex.EncodeToString(h[:])
}

// HashPasswordBcrypt hashes password with bcrypt.
func HashPasswordBcrypt(password string) (string, error) {
	hash, err := bcrypt.GenerateFromPassword([]byte(password), bcrypt.DefaultCost)
	return string(hash), err
}

// VerifyPasswordBcrypt verifies bcrypt hash.
func VerifyPasswordBcrypt(password, hash string) bool {
	return bcrypt.CompareHashAndPassword([]byte(hash), []byte(password)) == nil
}

// HashPasswordArgon2 hashes password using Argon2id.
func HashPasswordArgon2(password, salt string) string {
	hash := argon2.IDKey([]byte(password), []byte(salt), 3, 32*1024, 4, 32)
	return hex.EncodeToString(hash)
}
