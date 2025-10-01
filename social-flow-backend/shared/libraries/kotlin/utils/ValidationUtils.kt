package common.libraries.kotlin.utils

import java.util.regex.Pattern

/**
 * Common validation helpers used across the codebase.
 */
object ValidationUtils {
    private val EMAIL_RE: Pattern = Pattern.compile(
        "^[A-Z0-9._%+-]+@[A-Z0-9.-]+\\.[A-Z]{2,}\$",
        Pattern.CASE_INSENSITIVE
    )

    private val UUID_RE: Pattern = Pattern.compile(
        "^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[1-5][0-9a-fA-F]{3}-[89abAB][0-9a-fA-F]{3}-[0-9a-fA-F]{12}\$"
    )

    fun isEmail(value: String?): Boolean = value != null && EMAIL_RE.matcher(value).matches()

    fun isUuid(value: String?): Boolean = value != null && UUID_RE.matcher(value).matches()

    fun isSafeFilename(name: String?): Boolean {
        if (name == null) return false
        // Reject path traversal and special characters
        if (name.contains("..") || name.contains('/') || name.contains('\\')) return false
        return name.matches(Regex("^[a-zA-Z0-9._-]{1,255}\$"))
    }

    /** Validate a password using configurable rules; returns pair(isValid, reason) */
    fun validatePassword(password: String?): Pair<Boolean, String> {
        if (password == null || password.length < 8) return false to "Password must be >= 8 characters"
        if (!password.any { it.isDigit() }) return false to "Password must contain a digit"
        if (!password.any { it.isLowerCase() }) return false to "Password must contain a lowercase letter"
        if (!password.any { it.isUpperCase() }) return false to "Password must contain an uppercase letter"
        if (!password.any { "!@#\$%^&*()-_+=[]{}|;:'\",.<>?/\\`~".contains(it) }) return false to "Password must contain a symbol"
        return true to "OK"
    }
}
