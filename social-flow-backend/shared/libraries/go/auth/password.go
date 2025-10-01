// ========================================
// File: password.go
// ========================================
package auth

import (
	"crypto/rand"
	"crypto/subtle"
	"encoding/base64"
	"fmt"

	"golang.org/x/crypto/argon2"
	"golang.org/x/crypto/bcrypt"
)

// PasswordHasher defines an interface for hashing and verifying passwords
type PasswordHasher interface {
	Hash(password string) (string, error)
	Verify(password, hash string) bool
}

// =================== Bcrypt ===================
type BcryptHasher struct{}

func NewBcryptHasher() *BcryptHasher {
	return &BcryptHasher{}
}

func (b *BcryptHasher) Hash(password string) (string, error) {
	bytes, err := bcrypt.GenerateFromPassword([]byte(password), bcrypt.DefaultCost)
	return string(bytes), err
}

func (b *BcryptHasher) Verify(password, hash string) bool {
	err := bcrypt.CompareHashAndPassword([]byte(hash), []byte(password))
	return err == nil
}

// =================== Argon2id ===================
type Argon2Hasher struct {
	time    uint32
	memory  uint32
	threads uint8
	keyLen  uint32
}

func NewArgon2Hasher() *Argon2Hasher {
	return &Argon2Hasher{
		time:    1,
		memory:  64 * 1024,
		threads: 4,
		keyLen:  32,
	}
}

func (a *Argon2Hasher) Hash(password string) (string, error) {
	salt := make([]byte, 16)
	if _, err := rand.Read(salt); err != nil {
		return "", err
	}
	hash := argon2.IDKey([]byte(password), salt, a.time, a.memory, a.threads, a.keyLen)
	b64Salt := base64.RawStdEncoding.EncodeToString(salt)
	b64Hash := base64.RawStdEncoding.EncodeToString(hash)
	return fmt.Sprintf("argon2id$%s$%s", b64Salt, b64Hash), nil
}

func (a *Argon2Hasher) Verify(password, encoded string) bool {
	var b64Salt, b64Hash string
	_, err := fmt.Sscanf(encoded, "argon2id$%s$%s", &b64Salt, &b64Hash)
	if err != nil {
		return false
	}
	salt, _ := base64.RawStdEncoding.DecodeString(b64Salt)
	expectedHash, _ := base64.RawStdEncoding.DecodeString(b64Hash)

	hash := argon2.IDKey([]byte(password), salt, a.time, a.memory, a.threads, uint32(len(expectedHash)))
	return subtle.ConstantTimeCompare(hash, expectedHash) == 1
}
