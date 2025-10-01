package security

import (
	"bytes"
	"crypto/aes"
	"crypto/cipher"
	"crypto/rand"
	"crypto/rsa"
	"crypto/sha256"
	"crypto/x509"
	"encoding/pem"
	"errors"
	"io"
)

// ======================= AES-GCM ENCRYPTION =======================

// EncryptAESGCM encrypts plaintext using AES-GCM with a random nonce.
func EncryptAESGCM(key, plaintext []byte) (ciphertext []byte, nonce []byte, err error) {
	block, err := aes.NewCipher(key)
	if err != nil {
		return nil, nil, err
	}

	nonce = make([]byte, 12) // Standard nonce size for GCM
	if _, err := io.ReadFull(rand.Reader, nonce); err != nil {
		return nil, nil, err
	}

	aesgcm, err := cipher.NewGCM(block)
	if err != nil {
		return nil, nil, err
	}

	ciphertext = aesgcm.Seal(nil, nonce, plaintext, nil)
	return ciphertext, nonce, nil
}

// DecryptAESGCM decrypts ciphertext using AES-GCM.
func DecryptAESGCM(key, ciphertext, nonce []byte) ([]byte, error) {
	block, err := aes.NewCipher(key)
	if err != nil {
		return nil, err
	}

	aesgcm, err := cipher.NewGCM(block)
	if err != nil {
		return nil, err
	}

	return aesgcm.Open(nil, nonce, ciphertext, nil)
}

// ======================= RSA ENCRYPTION =======================

// GenerateRSAKeyPair generates an RSA key pair.
func GenerateRSAKeyPair(bits int) (*rsa.PrivateKey, error) {
	return rsa.GenerateKey(rand.Reader, bits)
}

// ExportRSAPrivateKey exports RSA private key in PEM format.
func ExportRSAPrivateKey(priv *rsa.PrivateKey) ([]byte, error) {
	privBytes := x509.MarshalPKCS1PrivateKey(priv)
	return pem.EncodeToMemory(&pem.Block{Type: "RSA PRIVATE KEY", Bytes: privBytes}), nil
}

// ExportRSAPublicKey exports RSA public key in PEM format.
func ExportRSAPublicKey(pub *rsa.PublicKey) ([]byte, error) {
	pubBytes, err := x509.MarshalPKIXPublicKey(pub)
	if err != nil {
		return nil, err
	}
	return pem.EncodeToMemory(&pem.Block{Type: "PUBLIC KEY", Bytes: pubBytes}), nil
}

// EncryptRSA encrypts data with RSA public key.
func EncryptRSA(pub *rsa.PublicKey, data []byte) ([]byte, error) {
	return rsa.EncryptOAEP(sha256.New(), rand.Reader, pub, data, nil)
}

// DecryptRSA decrypts RSA ciphertext with private key.
func DecryptRSA(priv *rsa.PrivateKey, ciphertext []byte) ([]byte, error) {
	return rsa.DecryptOAEP(sha256.New(), rand.Reader, priv, ciphertext, nil)
}

// ======================= HYBRID ENCRYPTION =======================

// HybridEncrypt encrypts data using AES for payload + RSA for AES key.
func HybridEncrypt(pub *rsa.PublicKey, data []byte) (encryptedKey, nonce, ciphertext []byte, err error) {
	// Generate random AES-256 key
	aesKey := make([]byte, 32)
	if _, err := rand.Read(aesKey); err != nil {
		return nil, nil, nil, err
	}

	// Encrypt AES key with RSA
	encryptedKey, err = EncryptRSA(pub, aesKey)
	if err != nil {
		return nil, nil, nil, err
	}

	// Encrypt payload with AES
	ciphertext, nonce, err = EncryptAESGCM(aesKey, data)
	return encryptedKey, nonce, ciphertext, err
}

// HybridDecrypt decrypts hybrid encrypted payload.
func HybridDecrypt(priv *rsa.PrivateKey, encryptedKey, nonce, ciphertext []byte) ([]byte, error) {
	aesKey, err := DecryptRSA(priv, encryptedKey)
	if err != nil {
		return nil, err
	}
	return DecryptAESGCM(aesKey, ciphertext, nonce)
}
