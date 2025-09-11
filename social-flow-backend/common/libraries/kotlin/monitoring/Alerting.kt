package common.libraries.kotlin.monitoring

import kotlinx.coroutines.*
import java.net.HttpURLConnection
import java.net.URL

/**
 * Simple alerting engine.
 * Triggers alerts when thresholds are exceeded.
 */
object Alerting {
    private val scope = CoroutineScope(Dispatchers.Default)

    fun startMonitoring() {
        if (!MonitoringConfig.enableAlerts) return

        scope.launch {
            while (isActive) {
                checkThresholds()
                delay(10_000)
            }
        }
    }

    private fun checkThresholds() {
        val metrics = MetricsCollector.export()

        metrics["system.load"]?.let {
            if (it > MonitoringConfig.cpuUsageThreshold) {
                sendAlert("High CPU Load Detected: $it")
            }
        }

        metrics["heap.used.mb"]?.let {
            if (it > MonitoringConfig.memoryUsageThreshold * 1024) {
                sendAlert("High Memory Usage Detected: $it MB")
            }
        }
    }

    private fun sendAlert(message: String) {
        Logger.warn("ALERT: $message")

        try {
            val url = URL(MonitoringConfig.alertWebhook)
            with(url.openConnection() as HttpURLConnection) {
                requestMethod = "POST"
                doOutput = true
                outputStream.write(message.toByteArray())
                Logger.info("Alert sent to webhook", mapOf("message" to message))
            }
        } catch (e: Exception) {
            Logger.error("Failed to send alert", mapOf("error" to e.message))
        }
    }
}
