package common.libraries.kotlin.database

import java.sql.Connection

/**
 * Small transaction-scoped interface that repositories can use to accept a Connection.
 * This encourages repositories to be transaction-agnostic and use the supplied Connection.
 *
 * Example:
 *   db.transaction { conn -> userRepo.findById(conn, id) }
 */

typealias Tx = Connection
