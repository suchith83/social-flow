package common.libraries.kotlin.database

/**
 * Configuration holder for DB connection. Fill from environment/config loader in your app.
 */
data class DatabaseConfig(
    val jdbcUrl: String,
    val username: String?,
    val password: String?,
    val maximumPoolSize: Int = 10,
    val minimumIdle: Int = 1,
    val connectionTimeoutMs: Long = 30_000,
    val idleTimeoutMs: Long = 600_000,
    val maxLifetimeMs: Long = 1_800_000, // 30 minutes
    val poolName: String = "app-pool",
    val enableFlyway: Boolean = true,
    val flywayLocations: List<String> = listOf("classpath:db/migration")
)
