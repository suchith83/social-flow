package common.libraries.kotlin.messaging.testing

import common.libraries.kotlin.messaging.*
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.launch
import kotlinx.coroutines.runBlocking
import java.util.concurrent.ConcurrentHashMap

/**
 * Lightweight in-memory broker for unit tests. Not a full broker — intended for testing producers and consumers.
 *
 * Usage:
 * val b = InMemoryBroker()
 * val p = b.producerFor("topic")
 * val c = b.consumerFor("topic")
 */
class InMemoryBroker {
    private val topics = ConcurrentHashMap<String, Channel<Message<ByteArray>>>()

    fun producerFor(topic: String): Producer {
        val ch = topics.computeIfAbsent(topic) { Channel(Channel.UNLIMITED) }
        return object : Producer {
            override suspend fun <T> send(message: Message<T>): DeliveryResult {
                val bytes = when (val p = message.payload) {
                    is ByteArray -> p
                    else -> throw IllegalArgumentException("InMemoryBroker expects ByteArray payload")
                }
                ch.send(Message(id = message.id, topic = topic, key = message.key, payload = bytes, headers = message.headers))
                return DeliveryResult.Success()
            }
        }
    }

    fun consumerFor(topic: String): Pair<Consumer, suspend (Message<ByteArray>) -> Unit> {
        val ch = topics.computeIfAbsent(topic) { Channel(Channel.UNLIMITED) }
        val consumer = object : Consumer {
            private var running = true
            override fun start(scope: kotlinx.coroutines.CoroutineScope, handler: MessageHandler<ByteArray>) =
                scope.launch {
                    while (running) {
                        val msg = ch.receive()
                        handler.handle(msg)
                    }
                }

            override fun stop() { running = false }
        }
        // helper to publish directly for tests
        val publisher: suspend (Message<ByteArray>) -> Unit = { m -> ch.send(m) }
        return Pair(consumer, publisher)
    }
}
