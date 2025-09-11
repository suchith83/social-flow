package common.libraries.kotlin.security

import java.security.KeyPairGenerator
import java.security.PrivateKey
import java.security.PublicKey
import javax.crypto.Cipher
import javax.crypto.spec.GCMParameterSpec
import javax.crypto.spec.SecretKeySpec
import java.util.*

/**
 * Provides AES/GCM encryption and RSA keypair generation.
 */
object EncryptionUtils {
    private const val AES_TRANSFORMATION = "AES/GCM/NoPadding"
    private const val RSA_ALGORITHM = "RSA"

    fun encryptAES(plainText: String): String {
        val keySpec = SecretKeySpec(SecurityConfig.aesKey.toByteArray(), "AES")
        val cipher = Cipher.getInstance(AES_TRANSFORMATION)
        val iv = ByteArray(12).also { SecureRandomUtils.secureRandom.nextBytes(it) }
        val gcmSpec = GCMParameterSpec(128, iv)

        cipher.init(Cipher.ENCRYPT_MODE, keySpec, gcmSpec)
        val encrypted = cipher.doFinal(plainText.toByteArray())

        return Base64.getEncoder().encodeToString(iv + encrypted)
    }

    fun decryptAES(cipherText: String): String {
        val decoded = Base64.getDecoder().decode(cipherText)
        val iv = decoded.copyOfRange(0, 12)
        val encrypted = decoded.copyOfRange(12, decoded.size)

        val keySpec = SecretKeySpec(SecurityConfig.aesKey.toByteArray(), "AES")
        val cipher = Cipher.getInstance(AES_TRANSFORMATION)
        val gcmSpec = GCMParameterSpec(128, iv)

        cipher.init(Cipher.DECRYPT_MODE, keySpec, gcmSpec)
        return String(cipher.doFinal(encrypted))
    }

    fun generateRSAKeyPair(): Pair<PublicKey, PrivateKey> {
        val keyGen = KeyPairGenerator.getInstance(RSA_ALGORITHM)
        keyGen.initialize(SecurityConfig.rsaKeySize)
        val pair = keyGen.generateKeyPair()
        return pair.public to pair.private
    }
}

/**
 * Utility for secure random number generation.
 */
object SecureRandomUtils {
    val secureRandom = java.security.SecureRandom()
}
