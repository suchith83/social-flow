// Rule to detect forbidden comments (TODO, FIXME, etc.)
package rules

import io.gitlab.arturbosch.detekt.api.*

/**
 * Rule: ForbiddenCommentRule
 * Detects and flags forbidden comments like TODO, FIXME, HACK, etc.
 * This enforces a clean code policy.
 */
class ForbiddenCommentRule(config: Config) : Rule(config) {

    override val issue = Issue(
        javaClass.simpleName,
        Severity.CodeSmell,
        "Flags forbidden comments such as TODO, FIXME, and HACK.",
        Debt.TWENTY_MINS
    )

    private val forbiddenPatterns = listOf("TODO", "FIXME", "HACK")

    override fun visitComment(node: PsiElement) {
        val text = node.text
        forbiddenPatterns.forEach { pattern ->
            if (text.contains(pattern, ignoreCase = true)) {
                report(
                    CodeSmell(
                        issue,
                        Entity.from(node),
                        "Forbidden comment '$pattern' found. Please remove or resolve it."
                    )
                )
            }
        }
        super.visitComment(node)
    }
}
