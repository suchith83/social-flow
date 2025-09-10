// ========================================
// File: oauth.go
// ========================================
package auth

import (
	"context"
	"errors"
	"golang.org/x/oauth2"
	"golang.org/x/oauth2/github"
	"golang.org/x/oauth2/google"
)

// OAuthConfig stores credentials for providers
type OAuthConfig struct {
	GoogleClientID     string
	GoogleClientSecret string
	GoogleRedirectURL  string

	GitHubClientID     string
	GitHubClientSecret string
	GitHubRedirectURL  string
}

type OAuthManager struct {
	google *oauth2.Config
	github *oauth2.Config
}

func NewOAuthManager(cfg *OAuthConfig) *OAuthManager {
	return &OAuthManager{
		google: &oauth2.Config{
			ClientID:     cfg.GoogleClientID,
			ClientSecret: cfg.GoogleClientSecret,
			RedirectURL:  cfg.GoogleRedirectURL,
			Scopes:       []string{"email", "profile"},
			Endpoint:     google.Endpoint,
		},
		github: &oauth2.Config{
			ClientID:     cfg.GitHubClientID,
			ClientSecret: cfg.GitHubClientSecret,
			RedirectURL:  cfg.GitHubRedirectURL,
			Scopes:       []string{"user:email"},
			Endpoint:     github.Endpoint,
		},
	}
}

func (o *OAuthManager) GetLoginURL(provider, state string) (string, error) {
	switch provider {
	case "google":
		return o.google.AuthCodeURL(state), nil
	case "github":
		return o.github.AuthCodeURL(state), nil
	default:
		return "", errors.New("unsupported provider")
	}
}

func (o *OAuthManager) ExchangeCode(ctx context.Context, provider, code string) (*UserClaims, error) {
	var conf *oauth2.Config
	switch provider {
	case "google":
		conf = o.google
	case "github":
		conf = o.github
	default:
		return nil, errors.New("unsupported provider")
	}

	token, err := conf.Exchange(ctx, code)
	if err != nil {
		return nil, err
	}

	// Normally, you would call provider’s userinfo endpoint.
	// For brevity, we’ll fake user extraction.
	return &UserClaims{
		UserID:   "oauth-user",
		Username: provider + "_user",
		Roles:    []string{"user"},
	}, nil
}
