package common.libraries.kotlin.messaging

import kotlinx.coroutines.delay

suspend fun backoffRetry(policy: RetryPolicy, op: suspend (attempt: Int) -> Boolean): Boolean {
    var attempt = 1
    while (attempt <= policy.maxAttempts) {
        val ok = try { op(attempt) } catch (e: Throwable) { false }
        if (ok) return true
        if (attempt == policy.maxAttempts) break
        val wait = policy.nextDelay(attempt)
        delay(wait)
        attempt++
    }
    return false
}
