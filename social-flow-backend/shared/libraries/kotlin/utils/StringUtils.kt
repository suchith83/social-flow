package common.libraries.kotlin.utils

import java.security.MessageDigest
import java.util.*
import kotlin.experimental.and

/**
 * String utilities: safe trimming, normalization, hashing, random id generation, templating helpers.
 */
object StringUtils {
    private val alphanum = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789".toCharArray()
    private val secureRandom = java.security.SecureRandom()

    /** Safely trim a nullable string and return null if result is empty. */
    fun safeTrimOrNull(value: String?): String? {
        return value?.trim()?.takeIf { it.isNotEmpty() }
    }

    /** Null-safe lowercase using locale-insensitive mapping. */
    fun safeLower(value: String?): String? = value?.lowercase()

    /** Generate a random alphanumeric id of specified length. */
    fun randomId(length: Int = 12): String {
        val buf = CharArray(length)
        for (i in 0 until length) buf[i] = alphanum[secureRandom.nextInt(alphanum.size)]
        return String(buf)
    }

    /** Compute SHA-256 hex digest of input. */
    fun sha256Hex(input: String): String {
        val md = MessageDigest.getInstance("SHA-256")
        val digest = md.digest(input.toByteArray(Charsets.UTF_8))
        return digest.joinToString("") { "%02x".format(it and 0xff.toByte()) }
    }

    /**
     * Simple templating: replace placeholders like {{name}} with values from map.
     * Does not evaluate code. Safe for simple templating.
     */
    fun renderTemplate(template: String, values: Map<String, Any?>): String {
        var out = template
        values.forEach { (k, v) ->
            out = out.replace("{{${k}}}", v?.toString() ?: "")
        }
        return out
    }
}
