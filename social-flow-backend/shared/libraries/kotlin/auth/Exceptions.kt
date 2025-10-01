package common.libraries.kotlin.auth

/**
 * Specific exceptions to represent auth failures.
 */
sealed class AuthException(message: String, cause: Throwable? = null): RuntimeException(message, cause)

class UserNotFoundException(userIdOrName: String): AuthException("User not found: $userIdOrName")
class InvalidCredentialsException: AuthException("Invalid credentials")
class AccountDisabledException: AuthException("Account disabled")
class TokenExpiredException: AuthException("Token expired")
class TokenNotFoundException: AuthException("Token not found")
class TokenRevokedException: AuthException("Token revoked")
class UnsupportedHashAlgorithmException(msg: String): AuthException(msg)
