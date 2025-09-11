package common.libraries.kotlin.utils

/**
 * Centralized configuration for utilities.
 * Allows tests or apps to override defaults.
 */
object UtilsConfig {
    // Default coroutine timeout for utilities (ms)
    @Volatile
    var defaultTimeoutMs: Long = 10_000L

    // Default dispatcher for blocking IO
    @Volatile
    var blockingDispatcher = kotlinx.coroutines.Dispatchers.IO

    // Retry defaults
    @Volatile
    var defaultMaxRetries: Int = 3
    @Volatile
    var defaultInitialBackoffMs: Long = 200L
    @Volatile
    var defaultMaxBackoffMs: Long = 5_000L

    // Circuit breaker defaults
    @Volatile
    var defaultCbFailureThreshold: Int = 5
    @Volatile
    var defaultCbResetTimeoutMs: Long = 60_000L
}
