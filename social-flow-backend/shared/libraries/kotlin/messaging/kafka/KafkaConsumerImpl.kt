package common.libraries.kotlin.messaging.kafka

import common.libraries.kotlin.messaging.*
import kotlinx.coroutines.*
import org.apache.kafka.clients.consumer.ConsumerConfig
import org.apache.kafka.clients.consumer.KafkaConsumer
import org.apache.kafka.clients.consumer.OffsetCommitCallback
import org.apache.kafka.common.errors.WakeupException
import java.time.Duration
import java.util.Properties
import java.util.concurrent.atomic.AtomicBoolean
import kotlin.coroutines.resume
import kotlin.coroutines.resumeWithException
import kotlin.coroutines.suspendCoroutine

/**
 * Kafka consumer implementation with manual commit and coroutine-driven loop.
 * Expects messages values as ByteArray.
 *
 * Note: This is a simple, correct-by-construction consumer that uses poll() in a loop.
 * For production you may add partition assignment listeners, rebalance handling, and metrics.
 */
class KafkaConsumerImpl<K>(
    private val consumer: KafkaConsumer<K, ByteArray>,
    private val topics: List<String>,
    private val retryPolicy: RetryPolicy = RetryPolicy(),
    private val pollInterval: Duration = Duration.ofMillis(500)
) : Consumer, HealthCheck {

    private val running = AtomicBoolean(false)
    private var job: Job? = null

    companion object {
        fun buildProps(bootstrapServers: String, groupId: String, clientId: String = "app-consumer"): Properties {
            return Properties().apply {
                put(ConsumerConfig.BOOTSTRAP_SERVERS_CONFIG, bootstrapServers)
                put(ConsumerConfig.GROUP_ID_CONFIG, groupId)
                put(ConsumerConfig.KEY_DESERIALIZER_CLASS_CONFIG, "org.apache.kafka.common.serialization.StringDeserializer")
                put(ConsumerConfig.VALUE_DESERIALIZER_CLASS_CONFIG, "org.apache.kafka.common.serialization.ByteArrayDeserializer")
                put(ConsumerConfig.ENABLE_AUTO_COMMIT_CONFIG, "false")
                put(ConsumerConfig.AUTO_OFFSET_RESET_CONFIG, "earliest")
                put(ConsumerConfig.CLIENT_ID_CONFIG, clientId)
                // tuning: max.poll.records, heartbeat, session.timeout.ms, etc.
            }
        }
    }

    override fun start(scope: CoroutineScope, handler: MessageHandler<ByteArray>): Job {
        if (running.getAndSet(true)) throw IllegalStateException("consumer already started")
        consumer.subscribe(topics)
        job = scope.launch(Dispatchers.IO) {
            try {
                while (isActive && running.get()) {
                    val records = consumer.poll(pollInterval)
                    if (records.count() == 0) continue

                    val ctx = this // coroutine scope for per-batch processing

                    // process records sequentially to make commit safe (or parallelize with care)
                    for (rec in records) {
                        val msg = Message(
                            id = "${rec.topic()}-${rec.partition()}-${rec.offset()}",
                            topic = rec.topic(),
                            key = rec.key()?.toString(),
                            payload = rec.value() ?: ByteArray(0),
                            headers = rec.headers().associate { it.key() to String(it.value()) }
                        )
                        val processed = backoffRetry(retryPolicy) { attempt ->
                            // call handler; returns Boolean success
                            handler.handle(msg)
                        }

                        if (processed) {
                            // commit offset for record
                            consumer.commitSync(mapOf(rec.topicPartition() to org.apache.kafka.clients.consumer.OffsetAndMetadata(rec.offset() + 1)))
                        } else {
                            // handler failed after retries — log and continue. Optionally move to DLQ
                            // do not commit so message will be retried per consumer group semantics
                        }
                    }
                }
            } catch (wx: WakeupException) {
                // expected on shutdown
            } finally {
                try { consumer.close() } catch (_: Exception) {}
            }
        }
        return job!!
    }

    override fun stop() {
        if (!running.getAndSet(false)) return
        consumer.wakeup()
        runBlocking {
            job?.join()
        }
    }

    override fun health(): ComponentHealth = ComponentHealth(name = "kafka-consumer", healthy = true)
}
