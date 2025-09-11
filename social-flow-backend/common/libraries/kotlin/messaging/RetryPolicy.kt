package common.libraries.kotlin.messaging

import kotlin.math.min
import kotlin.random.Random
import kotlin.time.Duration
import kotlin.time.DurationUnit
import kotlin.time.toDuration

/**
 * Simple retry policy that works with messaging handlers.
 */
data class RetryPolicy(
    val maxAttempts: Int = 5,
    val baseDelayMillis: Long = 200,
    val maxDelayMillis: Long = 10_000,
    val jitterFactor: Double = 0.2 // +/- jitter
) {
    fun nextDelay(attempt: Int): Long {
        val exp = (baseDelayMillis * Math.pow(2.0, (attempt - 1).toDouble())).toLong()
        val capped = min(exp, maxDelayMillis)
        val jitter = (capped * jitterFactor).toLong()
        return capped - jitter + Random.nextLong(jitter * 2 + 1)
    }
}
