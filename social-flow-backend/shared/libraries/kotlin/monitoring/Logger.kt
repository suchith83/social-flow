package common.libraries.kotlin.monitoring

import java.time.Instant
import java.util.*

/**
 * Structured JSON Logger with correlation IDs and severity levels.
 */
object Logger {
    enum class Level { DEBUG, INFO, WARN, ERROR, FATAL }

    private fun log(level: Level, message: String, context: Map<String, Any?> = emptyMap()) {
        if (!MonitoringConfig.enableLogging) return

        val logEvent = mapOf(
            "timestamp" to Instant.now().toString(),
            "level" to level.name,
            "message" to message,
            "correlationId" to (context["correlationId"] ?: UUID.randomUUID().toString()),
            "context" to context
        )

        println(logEvent.toString()) // Replace with JSON serializer for real system
    }

    fun debug(msg: String, ctx: Map<String, Any?> = emptyMap()) = log(Level.DEBUG, msg, ctx)
    fun info(msg: String, ctx: Map<String, Any?> = emptyMap()) = log(Level.INFO, msg, ctx)
    fun warn(msg: String, ctx: Map<String, Any?> = emptyMap()) = log(Level.WARN, msg, ctx)
    fun error(msg: String, ctx: Map<String, Any?> = emptyMap()) = log(Level.ERROR, msg, ctx)
    fun fatal(msg: String, ctx: Map<String, Any?> = emptyMap()) = log(Level.FATAL, msg, ctx)
}
