package common.libraries.kotlin.utils

import kotlinx.coroutines.delay
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import java.util.concurrent.atomic.AtomicInteger
import kotlin.system.measureTimeMillis

/**
 * Lightweight circuit breaker.
 * - counts failures in rolling window (simple counter)
 * - when threshold exceeded, circuit transitions to OPEN
 * - after resetTimeoutMs it becomes HALF_OPEN and allows one probe
 * - if probe success, closed; else, open again.
 *
 * This is intentionally simple and in-memory (process-local). For distributed systems,
 * integrate with Redis or other shared stores.
 */
class CircuitBreaker(
    private val failureThreshold: Int = UtilsConfig.defaultCbFailureThreshold,
    private val resetTimeoutMs: Long = UtilsConfig.defaultCbResetTimeoutMs
) {
    private val failures = AtomicInteger(0)
    @Volatile
    private var state: State = State.CLOSED
    @Volatile
    private var openedAt: Long = 0
    private val mutex = Mutex()

    enum class State { CLOSED, OPEN, HALF_OPEN }

    fun currentState(): State = state

    private fun recordFailure() {
        val f = failures.incrementAndGet()
        if (f >= failureThreshold && state == State.CLOSED) {
            openCircuit()
        }
    }

    private fun recordSuccess() {
        failures.set(0)
        state = State.CLOSED
    }

    private fun openCircuit() {
        state = State.OPEN
        openedAt = System.currentTimeMillis()
    }

    private fun canAttempt(): Boolean {
        return when (state) {
            State.CLOSED -> true
            State.OPEN -> {
                if (System.currentTimeMillis() - openedAt >= resetTimeoutMs) {
                    state = State.HALF_OPEN
                    true
                } else false
            }
            State.HALF_OPEN -> true
        }
    }

    /**
     * Execute [block] under circuit-breaker protection.
     * Throws CircuitOpenException if circuit does not allow attempts.
     */
    suspend fun <T> execute(block: suspend () -> T): T {
        if (!canAttempt()) throw CircuitOpenException("Circuit is open")

        return mutex.withLock {
            // Re-check after acquiring lock
            if (!canAttempt()) throw CircuitOpenException("Circuit is open")

            try {
                val result = block()
                recordSuccess()
                result
            } catch (e: Throwable) {
                recordFailure()
                throw e
            }
        }
    }

    class CircuitOpenException(message: String) : RuntimeException(message)
}
