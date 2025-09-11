package common.libraries.kotlin.messaging.sqs

import common.libraries.kotlin.messaging.*
import kotlinx.coroutines.*
import software.amazon.awssdk.auth.credentials.DefaultCredentialsProvider
import software.amazon.awssdk.core.SdkBytes
import software.amazon.awssdk.services.sqs.SqsAsyncClient
import software.amazon.awssdk.services.sqs.model.*
import java.net.URI
import java.time.Duration
import java.util.concurrent.CompletableFuture
import java.util.concurrent.atomic.AtomicBoolean
import kotlin.coroutines.resume
import kotlin.coroutines.resumeWithException
import kotlin.coroutines.suspendCoroutine

/**
 * SQS client wrapper providing Producer and Consumer behavior.
 * This example uses SqsAsyncClient; you can swap to sync client if preferred.
 *
 * It expects the queueUrl to be provided (long form).
 */
class SqsClientImpl(
    private val sqs: SqsAsyncClient,
    private val queueUrl: String,
    private val serializer: common.libraries.kotlin.messaging.MessageSerializer,
    private val waitTimeSeconds: Int = 10,
    private val visibilityTimeoutSeconds: Int = 30
) : common.libraries.kotlin.messaging.Producer, common.libraries.kotlin.messaging.Consumer, common.libraries.kotlin.messaging.HealthCheck {

    private val running = AtomicBoolean(false)
    private var job: Job? = null

    companion object {
        fun builder(endpoint: String? = null): SqsAsyncClient {
            val b = SqsAsyncClient.builder().credentialsProvider(DefaultCredentialsProvider.create())
            endpoint?.let { b.endpointOverride(URI.create(it)) }
            return b.build()
        }
    }

    override suspend fun <T> send(message: common.libraries.kotlin.messaging.Message<T>): common.libraries.kotlin.messaging.DeliveryResult {
        val value: ByteArray = when (val p = message.payload) {
            is ByteArray -> p
            else -> serializer.serialize(p)
        }
        val request = SendMessageRequest.builder()
            .queueUrl(queueUrl)
            .messageBody(String(value))
            .messageAttributes(message.headers.mapValues { (k, v) ->
                MessageAttributeValue.builder().dataType("String").stringValue(v).build()
            })
            .build()

        return try {
            val res = suspendCoroutine<SendMessageResponse> { cont ->
                sqs.sendMessage(request).whenComplete { r, ex ->
                    if (ex != null) cont.resumeWithException(ex) else cont.resume(r)
                }
            }
            DeliveryResult.Success(mapOf("messageId" to res.messageId()))
        } catch (e: Throwable) {
            DeliveryResult.Failure(e)
        }
    }

    override fun start(scope: CoroutineScope, handler: common.libraries.kotlin.messaging.MessageHandler<ByteArray>) : Job {
        if (running.getAndSet(true)) throw IllegalStateException("already started")
        job = scope.launch(Dispatchers.IO) {
            try {
                while (isActive && running.get()) {
                    val req = ReceiveMessageRequest.builder()
                        .queueUrl(queueUrl)
                        .maxNumberOfMessages(10)
                        .waitTimeSeconds(waitTimeSeconds)
                        .visibilityTimeout(visibilityTimeoutSeconds)
                        .messageAttributeNames("All")
                        .build()
                    val res = suspendCoroutine<ReceiveMessageResponse> { cont ->
                        sqs.receiveMessage(req).whenComplete { r, ex ->
                            if (ex != null) cont.resumeWithException(ex) else cont.resume(r)
                        }
                    }
                    val msgs = res.messages()
                    if (msgs.isEmpty()) continue
                    for (m in msgs) {
                        val body = m.body().toByteArray()
                        val msg = common.libraries.kotlin.messaging.Message(
                            id = m.messageId(),
                            topic = queueUrl,
                            payload = body,
                            headers = m.messageAttributes().mapValues { it.value().stringValue() ?: "" }
                        )
                        val processed = backoffRetry(common.libraries.kotlin.messaging.RetryPolicy(), { attempt ->
                            handler.handle(msg)
                        })
                        if (processed) {
                            // delete message
                            val delReq = DeleteMessageRequest.builder().queueUrl(queueUrl).receiptHandle(m.receiptHandle()).build()
                            suspendCoroutine<Unit> { cont ->
                                sqs.deleteMessage(delReq).whenComplete { _, ex ->
                                    if (ex != null) cont.resumeWithException(ex) else cont.resume(Unit)
                                }
                            }
                        } else {
                            // optionally change visibility or send to DLQ
                        }
                    }
                }
            } finally {
                // cleanup
            }
        }
        return job!!
    }

    override fun stop() {
        if (!running.getAndSet(false)) return
        runBlocking {
            job?.join()
        }
    }

    override fun health(): common.libraries.kotlin.messaging.ComponentHealth {
        return common.libraries.kotlin.messaging.ComponentHealth("sqs-client", healthy = true)
    }
}
