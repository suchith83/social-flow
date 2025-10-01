package common.libraries.kotlin.utils

import kotlinx.coroutines.delay
import kotlinx.coroutines.withContext
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlin.math.min
import kotlin.random.Random

/**
 * Robust retry helpers with exponential backoff, full jitter, and optional predicate.
 *
 * Example usage:
 * val result = retry(backoffMs = 200, maxRetries = 5) { doNetworkCall() }
 *
 * This function is suspendable and respects coroutines cancellation.
 */
suspend fun <T> retry(
    maxRetries: Int = UtilsConfig.defaultMaxRetries,
    initialBackoffMs: Long = UtilsConfig.defaultInitialBackoffMs,
    maxBackoffMs: Long = UtilsConfig.defaultMaxBackoffMs,
    factor: Double = 2.0,
    onRetry: suspend (attempt: Int, waitMs: Long, lastError: Throwable?) -> Unit = { _, _, _ -> },
    shouldRetry: (Throwable) -> Boolean = { true },
    block: suspend () -> T
): T {
    var attempt = 0
    var lastError: Throwable? = null

    while (true) {
        try {
            // run on blocking dispatcher if necessary for IO heavy work
            return withContext(UtilsConfig.blockingDispatcher) { block() }
        } catch (e: Throwable) {
            lastError = e
            attempt++
            if (attempt > maxRetries || !shouldRetry(e)) {
                throw e
            }

            val exp = initialBackoffMs * Math.pow(factor, (attempt - 1).toDouble())
            val wait = min(maxBackoffMs.toDouble(), exp).toLong()
            // full jitter: random between 0 and wait
            val jittered = Random.nextLong(0, wait + 1)
            onRetry(attempt, jittered, e)
            delay(jittered)
        }
    }
}

/**
 * A helper for safe retries with thread-safety and only one concurrent attempt for a given key.
 * Useful for caching or one-time refresh operations.
 */
class SingleFlight {
    private val locks = mutableMapOf<Any, Mutex>()

    private fun mutexFor(key: Any): Mutex = synchronized(locks) {
        locks.getOrPut(key) { Mutex() }
    }

    suspend fun <T> doOnce(key: Any, block: suspend () -> T): T {
        val m = mutexFor(key)
        m.withLock { return block() }
    }
}
