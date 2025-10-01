package common.libraries.kotlin.database

import java.time.Instant
import java.util.*

/**
 * Domain models used by the database package.
 * Keep POJOs / data classes here so repositories can return typed objects.
 */

data class DBUser(
    val id: UUID,
    val username: String,
    val email: String,
    val passwordHash: String,
    val salt: String?,
    val enabled: Boolean,
    val createdAt: Instant,
    val updatedAt: Instant?
)
