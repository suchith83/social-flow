package utils

import (
	"crypto/rand"
	"encoding/base64"
	"fmt"
	"math/big"
)

var letters = []rune("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")

// SecureRandomBytes returns n cryptographically secure random bytes.
func SecureRandomBytes(n int) ([]byte, error) {
	if n <= 0 {
		return nil, nil
	}
	b := make([]byte, n)
	_, err := rand.Read(b)
	return b, err
}

// RandomBase64 creates a URL-safe base64 string of n bytes.
func RandomBase64(n int) (string, error) {
	b, err := SecureRandomBytes(n)
	if err != nil {
		return "", err
	}
	return base64.RawURLEncoding.EncodeToString(b), nil
}

// RandomString generates a secure random string of length n using the alphanumeric alphabet.
func RandomString(n int) (string, error) {
	if n <= 0 {
		return "", nil
	}
	res := make([]rune, n)
	max := big.NewInt(int64(len(letters)))
	for i := 0; i < n; i++ {
		idx, err := rand.Int(rand.Reader, max)
		if err != nil {
			return "", err
		}
		res[i] = letters[idx.Int64()]
	}
	return string(res), nil
}

// RandomInt generates a cryptographically secure integer between [0, max).
func RandomInt(max int64) (int64, error) {
	if max <= 0 {
		return 0, nil
	}
	n, err := rand.Int(rand.Reader, big.NewInt(max))
	if err != nil {
		return 0, err
	}
	return n.Int64(), nil
}

// FriendlyID returns a short, safe ID suitable for URLs (uses 12 bytes by default).
func FriendlyID() (string, error) {
	return RandomBase64(12)
}
