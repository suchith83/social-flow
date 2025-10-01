package common.libraries.kotlin.database

import org.flywaydb.core.Flyway
import javax.sql.DataSource

/**
 * Flyway migration runner. Call `runMigrations` during application startup if enabled.
 *
 * This is optional but highly recommended. Migrations should be stored in resources under
 * db/migration (Flyway default) or configured `flywayLocations`.
 */

object Migrations {
    fun runMigrations(ds: DataSource, cfg: DatabaseConfig) {
        if (!cfg.enableFlyway) return
        val flyway = Flyway.configure()
            .dataSource(ds)
            .locations(*cfg.flywayLocations.toTypedArray())
            .baselineOnMigrate(true)
            .load()
        val result = flyway.migrate()
        // optionally log result: result.migrationsExecuted etc
    }
}
