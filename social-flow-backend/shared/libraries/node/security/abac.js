/**
 * abac.js
 *
 * Lightweight Attribute-Based Access Control (ABAC) engine.
 *
 * - Policies are simple JS objects exported by teams or built dynamically
 * - Policy structure:
 *    {
 *      id: 'policy1',
 *      description: '',
 *      target: (subject, resource, action) => boolean, // optional quick filter
 *      rules: [
 *        { effect: 'allow'|'deny', condition: (subject, resource, action) => boolean }
 *      ],
 *    }
 *
 * - evaluate(subject, resource, action) => { allowed: true/false, matched: [policyIds] }
 *
 * This is intentionally simple but expressive enough for many use cases.
 */

const logger = require('./logger');

class ABAC {
  constructor() {
    this.policies = new Map();
  }

  addPolicy(policy) {
    if (!policy || !policy.id) throw new Error('Invalid policy');
    this.policies.set(policy.id, policy);
  }

  removePolicy(id) {
    this.policies.delete(id);
  }

  listPolicies() {
    return Array.from(this.policies.values());
  }

  evaluate(subject = {}, resource = {}, action = '') {
    const matched = [];
    let decision = 'deny';
    for (const policy of this.policies.values()) {
      try {
        if (policy.target && typeof policy.target === 'function') {
          if (!policy.target(subject, resource, action)) continue;
        }
        // evaluate rules
        for (const rule of policy.rules || []) {
          try {
            if (typeof rule.condition === 'function' && rule.condition(subject, resource, action)) {
              matched.push(policy.id);
              if (rule.effect === 'deny') {
                // a deny rule takes precedence
                return { allowed: false, matched };
              } else if (rule.effect === 'allow') {
                decision = 'allow';
              }
            }
          } catch (e) {
            logger.warn({ e }, 'policy rule evaluation error');
            // ignore failing rule
          }
        }
      } catch (e) {
        logger.warn({ e }, 'policy target evaluation error');
      }
    }
    return { allowed: decision === 'allow', matched };
  }
}

module.exports = new ABAC();
