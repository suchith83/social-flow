// Rule for measuring cyclomatic complexity
package rules

import io.gitlab.arturbosch.detekt.api.*
import org.jetbrains.kotlin.psi.KtNamedFunction
import org.jetbrains.kotlin.psi.psiUtil.collectDescendantsOfType
import org.jetbrains.kotlin.psi.KtIfExpression
import org.jetbrains.kotlin.psi.KtForExpression
import org.jetbrains.kotlin.psi.KtWhileExpression
import org.jetbrains.kotlin.psi.KtWhenExpression

/**
 * Rule: CyclomaticComplexityRule
 * Enforces cyclomatic complexity thresholds for functions.
 * High complexity leads to harder maintainability and more bugs.
 */
class CyclomaticComplexityRule(config: Config) : Rule(config) {

    override val issue = Issue(
        javaClass.simpleName,
        Severity.Maintainability,
        "Function exceeds allowed cyclomatic complexity threshold.",
        Debt.TWENTY_MINS
    )

    private val threshold: Int = valueOrDefault("threshold", 10)

    override fun visitNamedFunction(function: KtNamedFunction) {
        val complexity = calculateCyclomaticComplexity(function)
        if (complexity > threshold) {
            report(
                CodeSmell(
                    issue,
                    Entity.atName(function),
                    "Function '${function.name}' has cyclomatic complexity of $complexity, " +
                            "which exceeds threshold of $threshold."
                )
            )
        }
        super.visitNamedFunction(function)
    }

    private fun calculateCyclomaticComplexity(function: KtNamedFunction): Int {
        var complexity = 1 // Base complexity
        complexity += function.collectDescendantsOfType<KtIfExpression>().size
        complexity += function.collectDescendantsOfType<KtWhenExpression>().size
        complexity += function.collectDescendantsOfType<KtForExpression>().size
        complexity += function.collectDescendantsOfType<KtWhileExpression>().size
        return complexity
    }
}
