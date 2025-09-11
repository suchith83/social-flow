package common.libraries.kotlin.database

import java.sql.Connection
import java.sql.PreparedStatement
import java.sql.ResultSet
import java.sql.Timestamp
import java.time.Instant
import java.util.*
import kotlin.math.max

/**
 * Lightweight helpers to work with JDBC without introducing an extra library.
 * These helpers focus on common patterns: executing queries, scanning results, and safe reads.
 *
 * If you prefer, you can replace these with kotliquery, jdbi, or an ORM like Exposed.
 */

/** Safely close result set ignoring exceptions */
fun safeClose(rs: ResultSet?) { try { rs?.close() } catch (_: Exception) {} }
fun safeClose(ps: PreparedStatement?) { try { ps?.close() } catch (_: Exception) {} }

/**
 * Prepare statement and bind parameters. Caller should close PreparedStatement.
 * `params` may contain java.sql.Types.* wrappers or plain primitives.
 */
fun Connection.prepareAndBind(sql: String, params: List<Any?> = emptyList()): PreparedStatement {
    val ps = this.prepareStatement(sql)
    params.forEachIndexed { idx, value ->
        val i = idx + 1
        when (value) {
            null -> ps.setObject(i, null)
            is String -> ps.setString(i, value)
            is Int -> ps.setInt(i, value)
            is Long -> ps.setLong(i, value)
            is Boolean -> ps.setBoolean(i, value)
            is java.sql.Timestamp -> ps.setTimestamp(i, value)
            is java.sql.Date -> ps.setDate(i, value)
            is java.util.Date -> ps.setTimestamp(i, Timestamp(value.time))
            is UUID -> ps.setObject(i, value)
            else -> ps.setObject(i, value)
        }
    }
    return ps
}

/** Helper to convert SQL timestamp to Instant */
fun tsToInstant(ts: Timestamp?): Instant? = ts?.toInstant()

/** Helper: read UUID from ResultSet by column name */
fun ResultSet.getUUID(column: String): UUID? {
    val obj = this.getObject(column)
    return when (obj) {
        is UUID -> obj
        is String -> UUID.fromString(obj)
        else -> null
    }
}

/** Helper: read Instant from ResultSet by column name */
fun ResultSet.getInstant(column: String): Instant? = tsToInstant(this.getTimestamp(column))

/** Execute update and return affected rows */
fun executeUpdate(conn: Connection, sql: String, params: List<Any?> = emptyList()): Int {
    val ps = conn.prepareAndBind(sql, params)
    ps.use { return it.executeUpdate() }
}

/** Query helper that maps a single row (or returns null) */
fun <T> queryOne(conn: Connection, sql: String, params: List<Any?> = emptyList(), mapper: (ResultSet) -> T): T? {
    val ps = conn.prepareAndBind(sql, params)
    ps.use {
        val rs = it.executeQuery()
        rs.use {
            return if (rs.next()) mapper(rs) else null
        }
    }
}

/** Query helper that maps multiple rows */
fun <T> queryAll(conn: Connection, sql: String, params: List<Any?> = emptyList(), mapper: (ResultSet) -> T): List<T> {
    val ps = conn.prepareAndBind(sql, params)
    ps.use {
        val rs = it.executeQuery()
        val out = ArrayList<T>()
        rs.use {
            while (rs.next()) {
                out.add(mapper(rs))
            }
        }
        return out
    }
}

/** Insert and return generated key (assumes single key of type long) */
fun insertAndReturnId(conn: Connection, sql: String, params: List<Any?> = emptyList()): Long {
    val ps = conn.prepareStatement(sql, java.sql.Statement.RETURN_GENERATED_KEYS)
    try {
        params.forEachIndexed { idx, v -> when (v) {
            null -> ps.setObject(idx+1, null)
            is String -> ps.setString(idx+1, v)
            is Int -> ps.setInt(idx+1, v)
            is Long -> ps.setLong(idx+1, v)
            is Boolean -> ps.setBoolean(idx+1, v)
            is UUID -> ps.setObject(idx+1, v)
            else -> ps.setObject(idx+1, v)
        } }
        ps.executeUpdate()
        val rs = ps.generatedKeys
        rs.use {
            return if (rs.next()) rs.getLong(1) else -1L
        }
    } finally {
        safeClose(ps)
    }
}
