package common.libraries.kotlin.messaging

import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

/**
 * Producer interface. Implementations should be safe for concurrent use.
 * send() is a suspending call that returns DeliveryResult.
 */
interface Producer {
    suspend fun <T> send(message: Message<T>): DeliveryResult
}

/**
 * Simple helper to send JSON-serializable message using MessageSerializer.
 */
suspend fun <T> Producer.sendJson(messageId: String, topic: String, key: String?, payload: T, serializer: MessageSerializer, headers: Map<String, String> = emptyMap()): DeliveryResult {
    val bytes = withContext(Dispatchers.Default) { serializer.serialize(payload) }
    val msg = Message(id = messageId, topic = topic, key = key, payload = bytes, headers = headers)
    @Suppress("UNCHECKED_CAST")
    return send(msg as Message<Any>)
}
