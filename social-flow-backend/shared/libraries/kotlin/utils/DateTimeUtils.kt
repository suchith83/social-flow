package common.libraries.kotlin.utils

import java.time.*
import java.time.format.DateTimeFormatter
import java.time.temporal.ChronoUnit

/**
 * Date/time helper utilities using java.time (immutable and thread-safe).
 */
object DateTimeUtils {
    val ISO: DateTimeFormatter = DateTimeFormatter.ISO_OFFSET_DATE_TIME
    val ISO_DATE: DateTimeFormatter = DateTimeFormatter.ISO_DATE

    /** Parse ISO or epoch millis string tolerant */
    fun parseToInstant(value: String): Instant? {
        return try {
            when {
                value.matches(Regex("^\\d+\$")) -> Instant.ofEpochMilli(value.toLong())
                else -> Instant.parse(value)
            }
        } catch (e: Exception) {
            null
        }
    }

    fun nowUtc(): Instant = Instant.now()

    fun nowLocal(zoneId: ZoneId = ZoneId.systemDefault()): ZonedDateTime = ZonedDateTime.now(zoneId)

    fun formatIso(instant: Instant): String = ISO.format(instant.atOffset(ZoneOffset.UTC))

    fun formatLocal(zdt: ZonedDateTime): String = ISO.format(zdt)

    /** Helpful human-friendly duration (e.g., "3m 12s") */
    fun humanDuration(ms: Long): String {
        if (ms < 1000) return "${ms}ms"
        var remaining = ms
        val days = remaining / (24 * 3600 * 1000); remaining %= 24 * 3600 * 1000
        val hours = remaining / (3600 * 1000); remaining %= 3600 * 1000
        val minutes = remaining / (60 * 1000); remaining %= 60 * 1000
        val seconds = remaining / 1000
        val parts = mutableListOf<String>()
        if (days > 0) parts += "${days}d"
        if (hours > 0) parts += "${hours}h"
        if (minutes > 0) parts += "${minutes}m"
        if (seconds > 0) parts += "${seconds}s"
        return parts.joinToString(" ")
    }

    /** Truncate an instant to seconds precision (useful for caching keys). */
    fun truncSeconds(i: Instant) = i.truncatedTo(ChronoUnit.SECONDS)
}
