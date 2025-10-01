package common.libraries.kotlin.auth

import java.security.SecureRandom
import java.util.*
import kotlin.experimental.and

/**
 * Password hashing abstraction.
 *
 * Provide concrete implementations: BCryptPasswordHasher, Argon2PasswordHasher, or a hybrid.
 *
 * Example usage:
 * val hasher: PasswordHasher = BCryptPasswordHasher()
 * val (hash, salt) = hasher.hash("secret")
 */
interface PasswordHasher {
    /**
     * Hash password, returns pair(hash, salt) -- salt may be null for algorithms that embed salt.
     */
    fun hash(password: CharArray): Pair<String, String?>

    /**
     * Verify raw password against stored hash and optional salt.
     */
    fun verify(password: CharArray, hash: String, salt: String?): Boolean

    /**
     * Identifier for algorithm, e.g. "bcrypt", "argon2id".
     */
    fun id(): String
}

/**
 * Simple BCrypt implementation using `org.mindrot.jbcrypt.BCrypt`.
 * Add dependency: org.mindrot:jbcrypt
 */
class BCryptPasswordHasher(private val cost: Int = 12) : PasswordHasher {
    override fun hash(password: CharArray): Pair<String, String?> {
        val pwStr = password.concatToString()
        val hashed = org.mindrot.jbcrypt.BCrypt.hashpw(pwStr, org.mindrot.jbcrypt.BCrypt.gensalt(cost))
        // BCrypt stores salt internally; we return null salt.
        return Pair(hashed, null)
    }

    override fun verify(password: CharArray, hash: String, salt: String?): Boolean {
        val pwStr = password.concatToString()
        return try {
            org.mindrot.jbcrypt.BCrypt.checkpw(pwStr, hash)
        } catch (e: Exception) {
            false
        }
    }

    override fun id(): String = "bcrypt"
}

/**
 * Argon2id implementation using `de.mkammerer:argon2-jvm` (recommended).
 * Add dependency: at.favre.lib:argon2-jvm or de.mkammerer:argon2-jvm
 */
class Argon2PasswordHasher(
    private val iterations: Int = 3,
    private val memoryKiB: Int = 1 shl 16, // 65536 KiB
    private val parallelism: Int = 1,
    private val hashLength: Int = 32
) : PasswordHasher {
    private val argon2 = de.mkammerer.argon2.Argon2Factory.create()

    override fun hash(password: CharArray): Pair<String, String?> {
        val hash = argon2.hash(iterations, memoryKiB, parallelism, password, hashLength)
        return Pair(hash, null)
    }

    override fun verify(password: CharArray, hash: String, salt: String?): Boolean {
        return argon2.verify(hash, password)
    }

    override fun id(): String = "argon2id"
}

/**
 * Utility to generate cryptographic salt if needed by custom schemes.
 */
object SaltUtil {
    private val rnd = SecureRandom()
    fun randomSaltBytes(n: Int = 16): ByteArray {
        val b = ByteArray(n)
        rnd.nextBytes(b)
        return b
    }
    fun toHex(bytes: ByteArray): String = bytes.joinToString("") { "%02x".format(it) }
}
