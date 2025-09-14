// Rule to enforce naming conventions
package rules

import io.gitlab.arturbosch.detekt.api.*
import org.jetbrains.kotlin.psi.KtNamedFunction
import org.jetbrains.kotlin.psi.KtProperty
import org.jetbrains.kotlin.psi.KtClass

/**
 * Rule: NamingConventionRule
 * Enforces naming standards for functions, variables, and classes.
 */
class NamingConventionRule(config: Config) : Rule(config) {

    override val issue = Issue(
        javaClass.simpleName,
        Severity.Style,
        "Violates naming conventions for functions, variables, or classes.",
        Debt.FIVE_MINS
    )

    private val functionPattern = Regex("[a-z][A-Za-z0-9]*")
    private val variablePattern = Regex("[a-z][A-Za-z0-9]*")
    private val classPattern = Regex("[A-Z][A-Za-z0-9]*")

    override fun visitNamedFunction(function: KtNamedFunction) {
        val name = function.name ?: return
        if (!functionPattern.matches(name)) {
            report(
                CodeSmell(
                    issue,
                    Entity.atName(function),
                    "Function name '$name' does not follow lowerCamelCase convention."
                )
            )
        }
        super.visitNamedFunction(function)
    }

    override fun visitProperty(property: KtProperty) {
        val name = property.name ?: return
        if (!variablePattern.matches(name)) {
            report(
                CodeSmell(
                    issue,
                    Entity.atName(property),
                    "Variable name '$name' does not follow lowerCamelCase convention."
                )
            )
        }
        super.visitProperty(property)
    }

    override fun visitClass(klass: KtClass) {
        val name = klass.name ?: return
        if (!classPattern.matches(name)) {
            report(
                CodeSmell(
                    issue,
                    Entity.atName(klass),
                    "Class name '$name' does not follow UpperCamelCase convention."
                )
            )
        }
        super.visitClass(klass)
    }
}
