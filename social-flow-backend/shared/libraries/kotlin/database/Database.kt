package common.libraries.kotlin.database

import java.sql.Connection
import javax.sql.DataSource
import kotlin.concurrent.thread

/**
 * High-level Database helper that wires DataSource + migrations and provides a convenient
 * transaction function. This can be replaced by a more feature-rich repository manager.
 *
 * Example usage:
 * val ds = DataSourceProvider.createHikariDataSource(cfg)
 * val db = Database(ds)
 * val user = db.transaction { tx -> userRepo.findById(tx, uuid) }
 */

class Database(private val dataSource: DataSource, private val cfg: DatabaseConfig? = null) {

    init {
        // if cfg requested flyway, run migrations in current thread. Alternatively run in background.
        cfg?.let { c ->
            if (c.enableFlyway) {
                // run migrations synchronously to avoid serving before schema ready
                Migrations.runMigrations(dataSource, c)
            }
        }
    }

    /**
     * Execute a block inside a JDBC transaction and commit/rollback automatically.
     * The block receives an active Connection. The Connection will be closed automatically.
     */
    fun <T> transaction(block: (Connection) -> T): T {
        return DataSourceProvider.withConnection(dataSource) { conn ->
            val autoCommit = conn.autoCommit
            try {
                if (autoCommit) conn.autoCommit = false
                val res = block(conn)
                conn.commit()
                res
            } catch (ex: Exception) {
                try { conn.rollback() } catch (ignored: Exception) {}
                throw ex
            } finally {
                try { conn.autoCommit = autoCommit } catch (ignored: Exception) {}
            }
        }
    }

    /**
     * Returns the raw DataSource (for low-level operations).
     */
    fun datasource(): DataSource = dataSource
}
