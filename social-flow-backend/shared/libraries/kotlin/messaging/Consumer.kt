package common.libraries.kotlin.messaging

import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Job

/**
 * Message handler to process inbound messages.
 * Return true if processing succeeded and message can be committed/acknowledged; false to retry.
 */
fun interface MessageHandler<T> {
    suspend fun handle(msg: Message<T>): Boolean
}

/**
 * Consumer interface. Implementations should provide start/stop lifecycle.
 */
interface Consumer {
    /**
     * Start consumer. Returns a Job representing the consumer lifecycle (so caller can cancel).
     */
    fun start(scope: CoroutineScope, handler: MessageHandler<ByteArray>): Job

    /**
     * Stop the consumer gracefully.
     */
    fun stop()
}
