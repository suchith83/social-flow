package common.libraries.kotlin.monitoring

import kotlinx.coroutines.*
import java.net.HttpURLConnection
import java.net.URL

/**
 * Exports collected metrics to Prometheus/Grafana-compatible endpoint.
 */
object DashboardExporter {
    private val scope = CoroutineScope(Dispatchers.IO)

    fun startExport() {
        if (!MonitoringConfig.enableMetrics) return

        scope.launch {
            while (isActive) {
                exportMetrics()
                delay(15_000) // every 15s
            }
        }
    }

    private fun exportMetrics() {
        val metrics = MetricsCollector.export()
        val payload = metrics.entries.joinToString("\n") { "${it.key} ${it.value}" }

        try {
            val url = URL(MonitoringConfig.prometheusEndpoint)
            with(url.openConnection() as HttpURLConnection) {
                requestMethod = "POST"
                doOutput = true
                outputStream.write(payload.toByteArray())
                Logger.debug("Metrics exported successfully")
            }
        } catch (e: Exception) {
            Logger.error("Metrics export failed", mapOf("error" to e.message))
        }
    }
}
