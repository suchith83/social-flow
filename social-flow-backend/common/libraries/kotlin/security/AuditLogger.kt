package common.libraries.kotlin.security

import java.time.Instant

/**
 * Security audit logging for compliance & forensics.
 */
object AuditLogger {
    enum class EventType { LOGIN_SUCCESS, LOGIN_FAILURE, TOKEN_ISSUE, TOKEN_EXPIRE, DATA_ACCESS }

    fun log(event: EventType, user: String?, details: Map<String, Any?> = emptyMap()) {
        val auditRecord = mapOf(
            "timestamp" to Instant.now().toString(),
            "event" to event.name,
            "user" to user,
            "details" to details
        )
        println("AUDIT: $auditRecord") // replace with persistent storage (DB, SIEM, etc.)
    }
}
