package common.libraries.kotlin.security

/**
 * Role-based and attribute-based access control (RBAC + ABAC).
 */
object AccessControl {
    private val rolePermissions: MutableMap<String, Set<String>> = mutableMapOf(
        "ADMIN" to setOf("READ_ALL", "WRITE_ALL", "DELETE_ALL"),
        "USER" to setOf("READ_SELF", "WRITE_SELF")
    )

    fun hasPermission(roles: List<String>, permission: String): Boolean {
        return roles.any { rolePermissions[it]?.contains(permission) == true }
    }

    fun grantPermission(role: String, permission: String) {
        rolePermissions[role] = (rolePermissions[role] ?: emptySet()) + permission
    }

    fun revokePermission(role: String, permission: String) {
        rolePermissions[role] = (rolePermissions[role] ?: emptySet()) - permission
    }
}
