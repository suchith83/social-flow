package common.libraries.kotlin.database.repositories

import common.libraries.kotlin.database.DBUser
import java.sql.Connection
import java.util.*

/**
 * Repository interface for users. Implementations should be transaction-aware (accept a Connection).
 */
interface UserRepository {
    fun findById(conn: Connection, id: UUID): DBUser?
    fun findByUsernameOrEmail(conn: Connection, identifier: String): DBUser?
    fun create(conn: Connection, user: DBUser): DBUser
    fun updatePassword(conn: Connection, userId: UUID, newHash: String, newSalt: String?): Boolean
    fun disableUser(conn: Connection, userId: UUID): Boolean
    // add paginated listing, filters, etc. as needed
}
