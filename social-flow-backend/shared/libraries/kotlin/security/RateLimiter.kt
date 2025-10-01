package common.libraries.kotlin.security

import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.atomic.AtomicInteger

/**
 * In-memory fixed window rate limiter (can be swapped with Redis).
 */
object RateLimiter {
    private val requestCounts: ConcurrentHashMap<String, Pair<Long, AtomicInteger>> = ConcurrentHashMap()

    fun allowRequest(clientId: String): Boolean {
        val currentMinute = System.currentTimeMillis() / 60000
        val entry = requestCounts.compute(clientId) { _, existing ->
            if (existing == null || existing.first != currentMinute) {
                currentMinute to AtomicInteger(1)
            } else {
                existing.first to AtomicInteger(existing.second.incrementAndGet())
            }
        }

        return entry!!.second.get() <= SecurityConfig.maxRequestsPerMinute
    }
}
