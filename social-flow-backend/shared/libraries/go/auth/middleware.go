// ========================================
// File: middleware.go
// ========================================
package auth

import (
	"net/http"
	"strings"

	"github.com/gin-gonic/gin"
)

// GinAuthMiddleware validates JWT in Authorization header
func GinAuthMiddleware(authMgr AuthManager) gin.HandlerFunc {
	return func(c *gin.Context) {
		authHeader := c.GetHeader("Authorization")
		if authHeader == "" || !strings.HasPrefix(authHeader, "Bearer ") {
			c.JSON(http.StatusUnauthorized, gin.H{"error": "missing token"})
			c.Abort()
			return
		}

		token := strings.TrimPrefix(authHeader, "Bearer ")
		claims, err := authMgr.ValidateToken(token)
		if err != nil {
			c.JSON(http.StatusUnauthorized, gin.H{"error": "invalid token"})
			c.Abort()
			return
		}

		c.Set("user", claims)
		c.Next()
	}
}
