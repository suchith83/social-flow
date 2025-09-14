// Provides a custom rule set for Detekt
package rules

import io.gitlab.arturbosch.detekt.api.*

/**
 * CustomRuleSetProvider is the entrypoint for registering
 * all custom Detekt rules created in this package.
 *
 * It will automatically register ForbiddenCommentRule,
 * CyclomaticComplexityRule, and NamingConventionRule.
 */
class CustomRuleSetProvider : RuleSetProvider {
    override val ruleSetId: String = "custom-rules"

    override fun instance(config: Config): RuleSet {
        return RuleSet(
            ruleSetId,
            listOf(
                ForbiddenCommentRule(config),
                CyclomaticComplexityRule(config),
                NamingConventionRule(config)
            )
        )
    }
}
