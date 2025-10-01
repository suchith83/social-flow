package common.libraries.kotlin.security

import de.mkammerer.argon2.Argon2Factory

/**
 * Handles password hashing and verification with Argon2id (memory-hard, resistant to GPU cracking).
 */
object PasswordHasher {
    private val argon2 = Argon2Factory.create()

    fun hashPassword(password: String): String {
        return argon2.hash(
            SecurityConfig.argon2Iterations,
            SecurityConfig.argon2Memory,
            SecurityConfig.argon2Parallelism,
            password.toCharArray()
        )
    }

    fun verifyPassword(hash: String, password: String): Boolean {
        return argon2.verify(hash, password.toCharArray())
    }
}
