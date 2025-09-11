package common.libraries.kotlin.auth

import java.time.Instant
import java.util.*
import kotlin.time.Duration
import kotlin.time.Duration.Companion.seconds

/**
 * High-level authentication service.
 *
 * Responsibilities:
 *  - validate credentials
 *  - issue access and refresh tokens
 *  - rotate refresh tokens
 *  - support logout / revoke
 *
 * This class depends on repository interfaces and JwtManager + PasswordHasher.
 */
class AuthService(
    private val userRepository: UserRepository,
    private val refreshTokenRepository: RefreshTokenRepository,
    private val passwordHasher: PasswordHasher,
    private val jwtManager: JwtManager,
    private val accessTtlSeconds: Long = 900,   // 15 minutes default
    private val refreshTtlSeconds: Long = 60 * 60 * 24 * 30 // 30 days
) {

    /**
     * Authenticate user, return AuthResult (access + refresh tokens).
     */
    fun authenticate(identifier: String, password: CharArray): AuthResult {
        val user = userRepository.findByUsernameOrEmail(identifier) ?: throw UserNotFoundException(identifier)
        if (!user.enabled) throw AccountDisabledException()

        // verify
        val ok = passwordHasher.verify(password, user.passwordHash, user.salt)
        if (!ok) throw InvalidCredentialsException()

        // create access token
        val access = jwtManager.generateToken(user.id.toString(), accessTtlSeconds, mapOf(
            "username" to user.username,
            "email" to user.email
        ))

        // create refresh token (random)
        val refresh = generateRefreshToken(user.id)

        return AuthResult(
            accessToken = access,
            refreshToken = refresh.token,
            expiresAt = Instant.now().plusSeconds(accessTtlSeconds)
        )
    }

    private fun generateRefreshToken(userId: UUID): RefreshToken {
        val tokenBytes = SaltUtil.randomSaltBytes(32) // cryptographically random token
        val tokenStr = Base64.getUrlEncoder().withoutPadding().encodeToString(tokenBytes)
        val expiresAt = Instant.now().plusSeconds(refreshTtlSeconds)
        val r = RefreshToken(token = tokenStr, userId = userId, expiresAt = expiresAt)
        refreshTokenRepository.save(r)
        return r
    }

    /**
     * Use refresh token to get a new access token (and optionally rotate refresh token).
     * Rotation: revoke old token and issue a new one.
     */
    fun refreshAccess(refreshTokenStr: String, rotate: Boolean = true): AuthResult {
        val existing = refreshTokenRepository.findByToken(refreshTokenStr) ?: throw TokenNotFoundException()
        if (existing.expiresAt.isBefore(Instant.now())) {
            refreshTokenRepository.revoke(refreshTokenStr)
            throw TokenExpiredException()
        }

        val user = userRepository.findById(existing.userId) ?: throw UserNotFoundException(existing.userId.toString())
        if (!user.enabled) throw AccountDisabledException()

        // issue access token
        val access = jwtManager.generateToken(user.id.toString(), accessTtlSeconds, mapOf(
            "username" to user.username
        ))

        var newRefreshTokenStr: String? = refreshTokenStr
        if (rotate) {
            // revoke old and issue new
            refreshTokenRepository.revoke(refreshTokenStr)
            val newRt = generateRefreshToken(user.id)
            newRefreshTokenStr = newRt.token
        }

        return AuthResult(accessToken = access, refreshToken = newRefreshTokenStr, expiresAt = Instant.now().plusSeconds(accessTtlSeconds))
    }

    /**
     * Revoke refresh token
     */
    fun revokeRefresh(refreshTokenStr: String) {
        val existing = refreshTokenRepository.findByToken(refreshTokenStr) ?: throw TokenNotFoundException()
        refreshTokenRepository.revoke(existing.token)
    }

    /**
     * Change password flow (verifies old password and rotates tokens)
     */
    fun changePassword(userId: UUID, currentPassword: CharArray, newPassword: CharArray) {
        val user = userRepository.findById(userId) ?: throw UserNotFoundException(userId.toString())
        if (!passwordHasher.verify(currentPassword, user.passwordHash, user.salt)) throw InvalidCredentialsException()

        val (newHash, newSalt) = passwordHasher.hash(newPassword)
        val ok = userRepository.updatePassword(userId, newHash, newSalt)
        if (!ok) throw AuthException("failed to update password")
        // revoke existing refresh tokens
        refreshTokenRepository.revokeAllForUser(userId)
    }
}
