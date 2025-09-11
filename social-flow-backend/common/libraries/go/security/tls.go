package security

import (
	"crypto/tls"
	"crypto/x509"
	"errors"
	"os"
)

// LoadTLSConfig loads TLS configuration with cert and CA validation.
func LoadTLSConfig(certFile, keyFile, caFile string) (*tls.Config, error) {
	cert, err := tls.LoadX509KeyPair(certFile, keyFile)
	if err != nil {
		return nil, err
	}

	caCert, err := os.ReadFile(caFile)
	if err != nil {
		return nil, err
	}

	caPool := x509.NewCertPool()
	if ok := caPool.AppendCertsFromPEM(caCert); !ok {
		return nil, errors.New("failed to append CA cert")
	}

	return &tls.Config{
		Certificates: []tls.Certificate{cert},
		ClientCAs:    caPool,
		MinVersion:   tls.VersionTLS13,
	}, nil
}
