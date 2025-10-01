package common.libraries.kotlin.monitoring

import java.util.concurrent.ConcurrentHashMap
import java.util.*

/**
 * Distributed tracing implementation.
 * Supports parent-child spans for request flows.
 */
object Tracing {
    data class Span(
        val id: String = UUID.randomUUID().toString(),
        val name: String,
        val parentId: String? = null,
        val startTime: Long = System.currentTimeMillis(),
        var endTime: Long? = null,
        val tags: MutableMap<String, Any> = mutableMapOf()
    )

    private val activeSpans: ConcurrentHashMap<String, Span> = ConcurrentHashMap()

    fun startSpan(name: String, parentId: String? = null): Span {
        if (!MonitoringConfig.enableTracing) return Span(name = name, parentId = parentId)
        val span = Span(name = name, parentId = parentId)
        activeSpans[span.id] = span
        return span
    }

    fun endSpan(span: Span) {
        span.endTime = System.currentTimeMillis()
        activeSpans.remove(span.id)
        Logger.info("Trace span finished", mapOf("span" to span))
    }

    fun addTag(span: Span, key: String, value: Any) {
        span.tags[key] = value
    }
}
