package common.libraries.kotlin.auth

import java.security.KeyFactory
import java.security.KeyPair
import java.security.KeyPairGenerator
import java.security.PrivateKey
import java.security.PublicKey
import java.security.interfaces.RSAPrivateKey
import java.security.interfaces.RSAPublicKey
import java.security.spec.PKCS8EncodedKeySpec
import java.security.spec.X509EncodedKeySpec
import javax.crypto.Cipher
import javax.crypto.KeyGenerator
import javax.crypto.SecretKey
import javax.crypto.spec.GCMParameterSpec

/**
 * Low-level crypto helpers.
 *
 * NOTE: Prefer to use high-level primitives (e.g., JWT libs, javax.crypto) and keep key usage simple.
 *
 * The helpers here are intentionally small and illustrative; adapt to your threat model.
 */

/** AES-GCM constants */
private const val AES_GCM_TAG_LEN_BITS = 128
private const val AES_ALGO = "AES"
private const val AES_GCM = "AES/GCM/NoPadding"

object CryptoUtils {
    fun generateAesKey(bits: Int = 256): SecretKey {
        val kg = KeyGenerator.getInstance(AES_ALGO)
        kg.init(bits)
        return kg.generateKey()
    }

    fun aesGcmEncrypt(key: SecretKey, plain: ByteArray, aad: ByteArray? = null): Pair<ByteArray, ByteArray> {
        val cipher = Cipher.getInstance(AES_GCM)
        val iv = ByteArray(12).also { java.security.SecureRandom().nextBytes(it) } // 96 bit IV recommended
        val spec = GCMParameterSpec(AES_GCM_TAG_LEN_BITS, iv)
        cipher.init(Cipher.ENCRYPT_MODE, key, spec)
        if (aad != null) cipher.updateAAD(aad)
        val ciphertext = cipher.doFinal(plain)
        return Pair(iv, ciphertext)
    }

    fun aesGcmDecrypt(key: SecretKey, iv: ByteArray, ciphertext: ByteArray, aad: ByteArray? = null): ByteArray {
        val cipher = Cipher.getInstance(AES_GCM)
        val spec = GCMParameterSpec(AES_GCM_TAG_LEN_BITS, iv)
        cipher.init(Cipher.DECRYPT_MODE, key, spec)
        if (aad != null) cipher.updateAAD(aad)
        return cipher.doFinal(ciphertext)
    }

    fun generateRsaKeyPair(bits: Int = 2048): KeyPair {
        val kpg = KeyPairGenerator.getInstance("RSA")
        kpg.initialize(bits)
        return kpg.generateKeyPair()
    }

    fun rsaEncrypt(pubKey: PublicKey, data: ByteArray): ByteArray {
        val cipher = Cipher.getInstance("RSA/ECB/OAEPWithSHA-256AndMGF1Padding")
        cipher.init(Cipher.ENCRYPT_MODE, pubKey)
        return cipher.doFinal(data)
    }

    fun rsaDecrypt(privKey: PrivateKey, data: ByteArray): ByteArray {
        val cipher = Cipher.getInstance("RSA/ECB/OAEPWithSHA-256AndMGF1Padding")
        cipher.init(Cipher.DECRYPT_MODE, privKey)
        return cipher.doFinal(data)
    }

    fun publicKeyFromBytes(pkBytes: ByteArray): PublicKey {
        val spec = X509EncodedKeySpec(pkBytes)
        val kf = KeyFactory.getInstance("RSA")
        return kf.generatePublic(spec)
    }
    fun privateKeyFromBytes(pkBytes: ByteArray): PrivateKey {
        val spec = PKCS8EncodedKeySpec(pkBytes)
        val kf = KeyFactory.getInstance("RSA")
        return kf.generatePrivate(spec)
    }
}
