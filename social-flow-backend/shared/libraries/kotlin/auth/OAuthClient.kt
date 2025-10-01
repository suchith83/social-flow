package common.libraries.kotlin.auth

import java.time.Instant

/**
 * Lightweight OAuth client abstraction for social login / external OIDC providers.
 *
 * This file contains a minimal interface + sample DTOs. Implement provider-specific clients
 * that call the provider endpoints and map returned profile to an internal User.
 *
 * NOTE: In production prefer a well-tested OIDC/OAuth library (e.g., Nimbus, Spring Security OAuth)
 * but this abstraction can help for small custom integrations.
 */

data class ExternalAuthToken(
    val accessToken: String,
    val refreshToken: String?,
    val expiresAt: Instant?
)

data class ExternalProfile(
    val id: String,
    val email: String?,
    val username: String?,
    val displayName: String?,
    val raw: Map<String, Any> = emptyMap()
)

interface OAuthClient {
    /**
     * Exchange code for access token (Authorization code flow).
     */
    fun exchangeCode(code: String, redirectUri: String): ExternalAuthToken

    /**
     * Fetch user profile using provider token.
     */
    fun fetchProfile(accessToken: String): ExternalProfile

    /**
     * Optionally refresh access token if provider supports it.
     */
    fun refreshToken(refreshToken: String): ExternalAuthToken?
}
