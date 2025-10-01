package common.libraries.kotlin.messaging

import java.time.Instant

/**
 * Basic message model used by the messaging library.
 * Keep this minimal and let callers attach payload as bytes or JSON.
 */
data class Message<T>(
    val id: String,
    val topic: String,
    val key: String? = null,
    val payload: T,
    val headers: Map<String, String> = emptyMap(),
    val createdAt: Instant = Instant.now()
)

/**
 * Delivery result returned from producers.
 */
sealed class DeliveryResult {
    data class Success(val metadata: Map<String, Any> = emptyMap()) : DeliveryResult()
    data class Failure(val cause: Throwable) : DeliveryResult()
}
