package security

import (
	"context"
	"encoding/base64"
	"errors"

	"github.com/aws/aws-sdk-go-v2/config"
	"github.com/aws/aws-sdk-go-v2/service/kms"
)

// EncryptWithKMS encrypts plaintext using AWS KMS key.
func EncryptWithKMS(ctx context.Context, keyID string, plaintext []byte) (string, error) {
	cfg, err := config.LoadDefaultConfig(ctx)
	if err != nil {
		return "", err
	}
	client := kms.NewFromConfig(cfg)

	out, err := client.Encrypt(ctx, &kms.EncryptInput{
		KeyId:     &keyID,
		Plaintext: plaintext,
	})
	if err != nil {
		return "", err
	}
	return base64.StdEncoding.EncodeToString(out.CiphertextBlob), nil
}

// DecryptWithKMS decrypts ciphertext using AWS KMS.
func DecryptWithKMS(ctx context.Context, ciphertextB64 string) ([]byte, error) {
	cfg, err := config.LoadDefaultConfig(ctx)
	if err != nil {
		return nil, err
	}
	client := kms.NewFromConfig(cfg)

	ciphertext, err := base64.StdEncoding.DecodeString(ciphertextB64)
	if err != nil {
		return nil, err
	}

	out, err := client.Decrypt(ctx, &kms.DecryptInput{
		CiphertextBlob: ciphertext,
	})
	if err != nil {
		return nil, err
	}
	return out.Plaintext, nil
}
