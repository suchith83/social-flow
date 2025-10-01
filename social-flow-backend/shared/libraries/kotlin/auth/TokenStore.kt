package common.libraries.kotlin.auth

import java.time.Instant
import java.util.*
import java.util.concurrent.ConcurrentHashMap

/**
 * Simple in-memory token store with optional persistence-backed repository wiring.
 * The library provides an interface (RefreshTokenRepository) above; this file provides
 * a convenient in-memory implementation for testing / small deployments.
 */

class InMemoryRefreshTokenRepository : RefreshTokenRepository {
    private val tokens = ConcurrentHashMap<String, RefreshToken>()

    override fun save(token: RefreshToken) {
        tokens[token.token] = token
    }

    override fun findByToken(token: String): RefreshToken? = tokens[token]

    override fun revoke(token: String) {
        tokens.remove(token)
    }

    override fun revokeAllForUser(userId: UUID) {
        tokens.entries.removeIf { it.value.userId == userId }
    }

    // Helper to cleanup expired tokens (call periodically)
    fun cleanupExpired(now: Instant = Instant.now()) {
        tokens.entries.removeIf { it.value.expiresAt.isBefore(now) }
    }
}
