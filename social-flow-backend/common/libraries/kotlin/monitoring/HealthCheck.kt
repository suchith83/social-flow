package common.libraries.kotlin.monitoring

/**
 * Provides health, readiness, and liveness checks for microservices.
 */
object HealthCheck {
    enum class Status { HEALTHY, UNHEALTHY }

    private val checks: MutableMap<String, () -> Boolean> = mutableMapOf()

    fun registerCheck(name: String, check: () -> Boolean) {
        checks[name] = check
    }

    fun runChecks(): Map<String, Status> {
        return checks.mapValues { (_, check) ->
            try {
                if (check()) Status.HEALTHY else Status.UNHEALTHY
            } catch (e: Exception) {
                Status.UNHEALTHY
            }
        }
    }

    fun overallStatus(): Status {
        return if (runChecks().all { it.value == Status.HEALTHY }) Status.HEALTHY else Status.UNHEALTHY
    }
}
