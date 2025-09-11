package common.libraries.kotlin.monitoring

import kotlinx.coroutines.*
import java.lang.management.ManagementFactory
import java.util.concurrent.ConcurrentHashMap
import kotlin.system.measureTimeMillis

/**
 * Collects system & application metrics.
 * Uses JVM's ManagementFactory and custom request timers.
 */
object MetricsCollector {
    private val metrics: ConcurrentHashMap<String, Double> = ConcurrentHashMap()

    init {
        // Schedule background system metric collection
        GlobalScope.launch(Dispatchers.Default) {
            while (true) {
                collectSystemMetrics()
                delay(5000) // every 5 seconds
            }
        }
    }

    fun recordLatency(name: String, block: suspend () -> Unit): Long {
        var duration = 0L
        runBlocking {
            duration = measureTimeMillis { runBlocking { block() } }
            metrics["latency.$name.ms"] = duration.toDouble()
        }
        return duration
    }

    fun incrementCounter(name: String, value: Int = 1) {
        metrics.compute(name) { _, old -> (old ?: 0.0) + value }
    }

    fun gauge(name: String, value: Double) {
        metrics[name] = value
    }

    private fun collectSystemMetrics() {
        val osBean = ManagementFactory.getOperatingSystemMXBean()
        val memoryBean = ManagementFactory.getMemoryMXBean()

        metrics["system.load"] = osBean.systemLoadAverage
        metrics["heap.used.mb"] = memoryBean.heapMemoryUsage.used / (1024.0 * 1024.0)
        metrics["nonheap.used.mb"] = memoryBean.nonHeapMemoryUsage.used / (1024.0 * 1024.0)
    }

    fun export(): Map<String, Double> = metrics.toMap()
}
