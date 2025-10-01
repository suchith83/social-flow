package common.libraries.kotlin.auth

import java.util.*

/**
 * Repository interfaces to decouple storage from logic.
 *
 * Implement these with your DB (Postgres, DynamoDB, etc.) in the application.
 */

interface UserRepository {
    /**
     * Find user by username or email (case-insensitive typically).
     */
    fun findByUsernameOrEmail(identifier: String): User?

    /**
     * Find user by id.
     */
    fun findById(id: UUID): User?

    /**
     * Save or update a user (returns saved user).
     */
    fun save(user: User): User

    /**
     * Update password hash and salt atomically.
     */
    fun updatePassword(userId: UUID, newHash: String, newSalt: String?): Boolean
}

interface RefreshTokenRepository {
    /**
     * Persist refresh token
     */
    fun save(token: RefreshToken)

    /**
     * Find refresh token
     */
    fun findByToken(token: String): RefreshToken?

    /**
     * Revoke or delete token
     */
    fun revoke(token: String)

    /**
     * Revoke all tokens for a user (e.g. on password change)
     */
    fun revokeAllForUser(userId: UUID)
}
