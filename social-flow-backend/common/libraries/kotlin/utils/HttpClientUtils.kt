package common.libraries.kotlin.utils

import io.ktor.client.*
import io.ktor.client.engine.cio.*
import io.ktor.client.features.*
import io.ktor.client.request.*
import io.ktor.client.statement.*
import kotlinx.coroutines.withTimeout
import kotlin.time.Duration
import kotlin.time.Duration.Companion.milliseconds

/**
 * Thin wrapper around Ktor HttpClient providing safe timeouts and JSON helpers.
 *
 * Add dependencies:
 * implementation("io.ktor:ktor-client-core:2.x")
 * implementation("io.ktor:ktor-client-cio:2.x")
 *
 * The client is reused (recommended) and configured with sensible timeouts.
 */
object HttpClientUtils {
    private val defaultTimeout = UtilsConfig.defaultTimeoutMs
    val client: HttpClient = HttpClient(CIO) {
        engine {
            // Default engine-level config
            requestTimeout = defaultTimeout
        }
        install(HttpTimeout) {
            requestTimeoutMillis = defaultTimeout
            connectTimeoutMillis = defaultTimeout
            socketTimeoutMillis = defaultTimeout
        }
        // Add additional features (auth, logging) as needed by caller
    }

    suspend inline fun getText(url: String, timeoutMs: Long = defaultTimeout): String {
        return withTimeout(timeoutMs) { client.get(url).readText() }
    }

    suspend inline fun getJson(url: String, timeoutMs: Long = defaultTimeout): String {
        return getText(url, timeoutMs)
    }

    suspend inline fun postJson(url: String, jsonBody: String, timeoutMs: Long = defaultTimeout): String {
        val response = withTimeout(timeoutMs) {
            client.post(url) {
                header("Content-Type", "application/json")
                body = jsonBody
            }
        }
        return response.readText()
    }
}
