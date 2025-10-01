package common.libraries.kotlin.auth

import java.time.Instant
import java.util.*

/**
 * Domain models used by the auth library.
 */

data class User(
    val id: UUID,
    val username: String,
    val email: String,
    val passwordHash: String,     // stored password hash
    val salt: String? = null,     // used by some hash algorithms (optional)
    val enabled: Boolean = true,
    val createdAt: Instant = Instant.now(),
    val metadata: Map<String, String> = emptyMap()
)

data class AuthResult(
    val accessToken: String,
    val refreshToken: String?,
    val expiresAt: Instant
)

data class RefreshToken(
    val token: String,
    val userId: UUID,
    val expiresAt: Instant,
    val issuedAt: Instant = Instant.now(),
    val metadata: Map<String, String> = emptyMap()
)

data class JwtClaims(
    val sub: String,
    val jti: String,
    val iat: Long,
    val exp: Long,
    val additional: Map<String, Any> = emptyMap()
)
