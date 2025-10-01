package common.libraries.kotlin.security

import io.jsonwebtoken.Jwts
import io.jsonwebtoken.SignatureAlgorithm
import io.jsonwebtoken.security.Keys
import java.util.*

/**
 * Issues and validates JWTs with expiration and claims.
 */
object JwtProvider {
    private val signingKey = Keys.hmacShaKeyFor(SecurityConfig.jwtSecret.toByteArray())

    fun generateToken(subject: String, roles: List<String> = emptyList()): String {
        val now = Date()
        val expiry = Date(now.time + SecurityConfig.jwtExpirationMs)

        return Jwts.builder()
            .setSubject(subject)
            .setIssuer(SecurityConfig.jwtIssuer)
            .setIssuedAt(now)
            .setExpiration(expiry)
            .claim("roles", roles)
            .signWith(signingKey, SignatureAlgorithm.HS256)
            .compact()
    }

    fun validateToken(token: String): Boolean {
        return try {
            val claims = Jwts.parserBuilder()
                .setSigningKey(signingKey)
                .build()
                .parseClaimsJws(token)
            !claims.body.expiration.before(Date())
        } catch (e: Exception) {
            false
        }
    }

    fun extractSubject(token: String): String? {
        return try {
            Jwts.parserBuilder().setSigningKey(signingKey).build()
                .parseClaimsJws(token).body.subject
        } catch (e: Exception) {
            null
        }
    }

    fun extractRoles(token: String): List<String> {
        return try {
            val claims = Jwts.parserBuilder().setSigningKey(signingKey).build()
                .parseClaimsJws(token).body
            (claims["roles"] as? List<*>)?.filterIsInstance<String>() ?: emptyList()
        } catch (e: Exception) {
            emptyList()
        }
    }
}
