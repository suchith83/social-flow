package common.libraries.kotlin.messaging

/**
 * Simple health indicator interfaces for producers/consumers.
 */
data class ComponentHealth(val name: String, val healthy: Boolean, val details: Map<String, String> = emptyMap())

interface HealthCheck {
    fun health(): ComponentHealth
}
