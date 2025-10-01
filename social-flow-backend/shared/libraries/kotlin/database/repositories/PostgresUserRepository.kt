package common.libraries.kotlin.database.repositories

import common.libraries.kotlin.database.*
import java.sql.Connection
import java.sql.ResultSet
import java.time.Instant
import java.util.*

/**
 * Postgres implementation of UserRepository using plain JDBC helpers from jdbcUtils.
 *
 * SQL uses named columns and returns typed values. This implementation focuses on clarity,
 * SQL correctness, and protection against SQL injection via prepared statements.
 *
 * NOTE: use LOWER(...) on lookups if your DB collation requires case-insensitive search.
 */

class PostgresUserRepository : UserRepository {

    private fun mapUser(rs: ResultSet): DBUser {
        return DBUser(
            id = rs.getUUID("id")!!,
            username = rs.getString("username"),
            email = rs.getString("email"),
            passwordHash = rs.getString("password_hash"),
            salt = rs.getString("salt"),
            enabled = rs.getBoolean("enabled"),
            createdAt = rs.getInstant("created_at") ?: Instant.now(),
            updatedAt = rs.getInstant("updated_at")
        )
    }

    override fun findById(conn: Connection, id: UUID): DBUser? {
        val sql = "SELECT id, username, email, password_hash, salt, enabled, created_at, updated_at FROM app_users WHERE id = ?"
        return queryOne(conn, sql, listOf(id)) { rs -> mapUser(rs) }
    }

    override fun findByUsernameOrEmail(conn: Connection, identifier: String): DBUser? {
        val sql = """
            SELECT id, username, email, password_hash, salt, enabled, created_at, updated_at
            FROM app_users
            WHERE username = ? OR email = ?
            LIMIT 1
        """.trimIndent()
        return queryOne(conn, sql, listOf(identifier, identifier)) { rs -> mapUser(rs) }
    }

    override fun create(conn: Connection, user: DBUser): DBUser {
        // We'll use UUID as primary key passed from application
        val sql = """
            INSERT INTO app_users (id, username, email, password_hash, salt, enabled, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """.trimIndent()
        val createdAt = java.sql.Timestamp.from(user.createdAt)
        executeUpdate(conn, sql, listOf(user.id, user.username, user.email, user.passwordHash, user.salt, user.enabled, createdAt))
        // Return the persisted user (updated_at null initially)
        return user.copy()
    }

    override fun updatePassword(conn: Connection, userId: UUID, newHash: String, newSalt: String?): Boolean {
        val sql = "UPDATE app_users SET password_hash = ?, salt = ?, updated_at = ? WHERE id = ?"
        val updatedAt = java.sql.Timestamp.from(Instant.now())
        val rows = executeUpdate(conn, sql, listOf(newHash, newSalt, updatedAt, userId))
        return rows > 0
    }

    override fun disableUser(conn: Connection, userId: UUID): Boolean {
        val sql = "UPDATE app_users SET enabled = false, updated_at = ? WHERE id = ?"
        val updatedAt = java.sql.Timestamp.from(Instant.now())
        return executeUpdate(conn, sql, listOf(updatedAt, userId)) > 0
    }
}
