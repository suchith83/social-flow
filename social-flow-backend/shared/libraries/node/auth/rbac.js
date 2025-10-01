/**
 * Role-Based Access Control (RBAC)
 */

class RBAC {
  constructor() {
    this.roles = new Map(); // role -> permissions
  }

  addRole(role, permissions = []) {
    this.roles.set(role, new Set(permissions));
  }

  grant(role, permission) {
    if (!this.roles.has(role)) this.roles.set(role, new Set());
    this.roles.get(role).add(permission);
  }

  revoke(role, permission) {
    if (this.roles.has(role)) this.roles.get(role).delete(permission);
  }

  can(role, permission) {
    return this.roles.has(role) && this.roles.get(role).has(permission);
  }
}

const rbac = new RBAC();

// Default roles
rbac.addRole('user', ['read:self']);
rbac.addRole('admin', ['read:any', 'write:any', 'delete:any']);

module.exports = rbac;
