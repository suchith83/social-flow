package common.libraries.kotlin.database

import com.zaxxer.hikari.HikariConfig
import com.zaxxer.hikari.HikariDataSource
import java.sql.Connection
import javax.sql.DataSource

/**
 * Simple provider that returns a configured HikariCP DataSource.
 * You can wrap this in your DI container and share the DataSource as a singleton.
 *
 * NOTE: Close the datasource at application shutdown by calling datasource.close()
 */

object DataSourceProvider {

    fun createHikariDataSource(cfg: DatabaseConfig): HikariDataSource {
        val hc = HikariConfig().apply {
            jdbcUrl = cfg.jdbcUrl
            username = cfg.username
            password = cfg.password
            maximumPoolSize = cfg.maximumPoolSize
            minimumIdle = cfg.minimumIdle
            connectionTimeout = cfg.connectionTimeoutMs
            idleTimeout = cfg.idleTimeoutMs
            maxLifetime = cfg.maxLifetimeMs
            poolName = cfg.poolName
            isAutoCommit = false // explicit transactions by default
            addDataSourceProperty("cachePrepStmts", "true")
            addDataSourceProperty("prepStmtCacheSize", "250")
            addDataSourceProperty("prepStmtCacheSqlLimit", "2048")
            // Postgres-specific tuning may be added by callers
        }
        return HikariDataSource(hc)
    }

    /**
     * Utility to only fetch a connection (for manual usage) and ensure auto-close.
     */
    fun <T> withConnection(ds: DataSource, block: (Connection) -> T): T {
        ds.connection.use { conn ->
            return block(conn)
        }
    }
}
