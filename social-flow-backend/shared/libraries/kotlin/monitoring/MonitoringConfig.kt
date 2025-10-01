package common.libraries.kotlin.monitoring

/**
 * Centralized configuration for monitoring components.
 * This allows dynamic reloading of config (via file/env/service discovery).
 */
object MonitoringConfig {
    var enableTracing: Boolean = true
    var enableMetrics: Boolean = true
    var enableLogging: Boolean = true
    var enableAlerts: Boolean = true

    // Metrics sampling rate (percentage of requests traced/logged)
    var samplingRate: Double = 1.0

    // Thresholds for alerting
    var cpuUsageThreshold: Double = 0.85
    var memoryUsageThreshold: Double = 0.90
    var requestLatencyThresholdMs: Long = 2000

    // Exporter endpoints
    var prometheusEndpoint: String = "http://localhost:9090/metrics"
    var tracingCollectorEndpoint: String = "http://localhost:4317"
    var alertWebhook: String = "http://localhost:8080/alerts"

    // Dynamic reloading hook
    fun reload(newConfig: Map<String, Any>) {
        newConfig["samplingRate"]?.let { samplingRate = it as Double }
        newConfig["cpuUsageThreshold"]?.let { cpuUsageThreshold = it as Double }
        newConfig["memoryUsageThreshold"]?.let { memoryUsageThreshold = it as Double }
        newConfig["requestLatencyThresholdMs"]?.let { requestLatencyThresholdMs = (it as Number).toLong() }
    }
}
