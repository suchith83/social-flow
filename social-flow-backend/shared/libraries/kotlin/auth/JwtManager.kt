package common.libraries.kotlin.auth

import com.fasterxml.jackson.databind.ObjectMapper
import io.jsonwebtoken.*
import io.jsonwebtoken.security.Keys
import java.security.Key
import java.security.PrivateKey
import java.security.PublicKey
import java.time.Instant
import java.util.*
import kotlin.collections.HashMap

/**
 * JWT Manager supporting HMAC (shared secret) and RSA signing.
 *
 * This wrapper uses `io.jsonwebtoken` (jjwt). Configure dependencies:
 *   implementation("io.jsonwebtoken:jjwt-api:<ver>")
 *   runtimeOnly("io.jsonwebtoken:jjwt-impl:<ver>")
 *   runtimeOnly("io.jsonwebtoken:jjwt-jackson:<ver>")
 *
 * Accepts either a symmetric secret or an asymmetric key pair.
 */
class JwtManager private constructor(
    private val signingKey: Key,
    private val verifyKey: Key?,
    private val issuer: String?,
    private val objectMapper: ObjectMapper = ObjectMapper()
) {
    data class Config(val signingKey: Key, val verifyKey: Key? = null, val issuer: String? = null)

    companion object {
        /** Convenience: create manager from a shared secret (HMAC-SHA256). */
        fun fromHmac(secretBytes: ByteArray, issuer: String? = null): JwtManager {
            val key = Keys.hmacShaKeyFor(secretBytes)
            return JwtManager(key, key, issuer)
        }

        /** Create manager from RSA keys */
        fun fromRsa(privateKey: PrivateKey, publicKey: PublicKey, issuer: String? = null): JwtManager {
            return JwtManager(privateKey, publicKey, issuer)
        }
    }

    /**
     * Build and sign JWT.
     * additionalClaims: map of custom claims to include
     */
    fun generateToken(subject: String, ttlSeconds: Long, additionalClaims: Map<String, Any> = emptyMap()): String {
        val now = Date.from(Instant.now())
        val exp = Date.from(Instant.now().plusSeconds(ttlSeconds))
        val jti = UUID.randomUUID().toString()

        val builder = Jwts.builder()
            .setSubject(subject)
            .setId(jti)
            .setIssuedAt(now)
            .setExpiration(exp)

        issuer?.let { builder.setIssuer(it) }

        // add custom claims
        for ((k, v) in additionalClaims) builder.claim(k, v)

        return builder.signWith(signingKey, SignatureAlgorithm.forSigningKey(signingKey)).compact()
    }

    /**
     * Parse and validate token. Returns claims map on success or throws exception.
     */
    fun parseToken(token: String): Jws<Claims> {
        val parser = Jwts.parserBuilder().setSigningKey(verifyKey ?: signingKey).build()
        return parser.parseClaimsJws(token)
    }

    /**
     * Safe verify method returning domain-friendly JwtClaims.
     */
    fun verifyAndExtract(token: String): JwtClaims {
        try {
            val jws = parseToken(token)
            val c = jws.body
            val sub = c.subject ?: throw JwtException("missing sub")
            val jti = c.id ?: UUID.randomUUID().toString()
            val iat = c.issuedAt?.time?.div(1000) ?: (Instant.now().epochSecond)
            val exp = c.expiration?.time?.div(1000) ?: (Instant.now().plusSeconds(1).epochSecond)
            // extract additional claims excluding standard ones
            val additional = HashMap<String, Any>()
            for ((k, v) in c) {
                if (k !in listOf(Claims.SUBJECT, Claims.ID, Claims.EXPIRATION, Claims.ISSUED_AT, Claims.ISSUER)) {
                    additional[k] = v
                }
            }
            return JwtClaims(sub = sub, jti = jti, iat = iat, exp = exp, additional = additional)
        } catch (e: ExpiredJwtException) {
            throw TokenExpiredException()
        } catch (e: JwtException) {
            throw AuthException("invalid jwt: ${e.message}", e)
        }
    }
}
