package common.libraries.kotlin.utils

import kotlinx.coroutines.withTimeout
import java.io.Closeable

/**
 * Common Kotlin extension helpers.
 */

/** Execute [block] with a default timeout (from UtilsConfig). */
suspend fun <T> withDefaultTimeout(block: suspend () -> T): T =
    withTimeout(UtilsConfig.defaultTimeoutMs) { block() }

/** Safely close a Closeable ignoring exceptions. */
fun Closeable?.closeSafe() {
    try {
        this?.close()
    } catch (ignored: Exception) {
    }
}

/** Null-safe apply: only run block when receiver is not null. */
inline fun <T : Any, R> T?.ifPresent(block: (T) -> R?): R? = this?.let(block)
