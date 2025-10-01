package common.libraries.kotlin.messaging.kafka

import common.libraries.kotlin.messaging.*
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.apache.kafka.clients.producer.KafkaProducer
import org.apache.kafka.clients.producer.ProducerRecord
import org.apache.kafka.clients.producer.RecordMetadata
import java.util.Properties
import java.util.concurrent.CompletableFuture
import kotlin.coroutines.resume
import kotlin.coroutines.resumeWithException
import kotlin.coroutines.suspendCoroutine

/**
 * Kafka producer implementation wrapping org.apache.kafka.clients.producer.KafkaProducer
 *
 * The generic Message.payload can be bytes (ByteArray) or a typed object already serialized by caller.
 * This implementation expects Message.payload to be ByteArray for direct sending.
 */
class KafkaProducerImpl<K>(
    private val producer: KafkaProducer<K, ByteArray>,
    private val serializer: MessageSerializer
) : Producer, HealthCheck {

    companion object {
        fun buildProps(bootstrapServers: String, clientId: String = "app-producer"): Properties {
            return Properties().apply {
                put("bootstrap.servers", bootstrapServers)
                put("acks", "all")
                put("retries", "3")
                put("client.id", clientId)
                put("key.serializer", "org.apache.kafka.common.serialization.StringSerializer")
                put("value.serializer", "org.apache.kafka.common.serialization.ByteArraySerializer")
            }
        }
    }

    override suspend fun <T> send(message: Message<T>): DeliveryResult {
        // Expect message.payload to be ByteArray or serializable via serializer
        val value: ByteArray = when (val p = message.payload) {
            is ByteArray -> p
            else -> serializer.serialize(p)
        }

        val record = ProducerRecord<K, ByteArray>(message.topic, message.key as K?, value).also {
            for ((k, v) in message.headers) it.headers().add(k, v.toByteArray())
        }

        return withContext(Dispatchers.IO) {
            try {
                val metadata = sendAsync(record)
                DeliveryResult.Success(mapOf(
                    "partition" to metadata.partition(),
                    "offset" to metadata.offset(),
                    "topic" to metadata.topic()
                ))
            } catch (e: Throwable) {
                DeliveryResult.Failure(e)
            }
        }
    }

    private suspend fun sendAsync(record: ProducerRecord<K, ByteArray>): RecordMetadata =
        suspendCoroutine { cont ->
            producer.send(record) { metadata, exception ->
                if (exception != null) cont.resumeWithException(exception)
                else cont.resume(metadata)
            }
        }

    fun close() { producer.flush(); producer.close() }

    override fun health(): ComponentHealth {
        // KafkaProducer doesn't have a built-in health probe — we rely on metadata fetch
        return ComponentHealth(name = "kafka-producer", healthy = true)
    }
}
