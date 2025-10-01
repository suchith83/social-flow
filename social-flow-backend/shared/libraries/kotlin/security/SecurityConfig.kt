package common.libraries.kotlin.security

/**
 * Centralized configuration for all security components.
 * Secrets should be sourced from environment variables or secret managers.
 */
object SecurityConfig {
    var jwtSecret: String = System.getenv("JWT_SECRET") ?: "CHANGE_ME_SECRET"
    var jwtIssuer: String = "social-flow-backend"
    var jwtExpirationMs: Long = 1000L * 60 * 60 // 1 hour

    var argon2Memory: Int = 65536
    var argon2Iterations: Int = 3
    var argon2Parallelism: Int = 2

    var aesKey: String = System.getenv("AES_KEY") ?: "0123456789abcdef" // 16 bytes
    var rsaKeySize: Int = 2048

    var maxRequestsPerMinute: Int = 100
}
