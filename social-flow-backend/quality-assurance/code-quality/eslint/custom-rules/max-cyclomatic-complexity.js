/**
 * Rule: max-cyclomatic-complexity
 * Enforces cyclomatic complexity limits in functions.
 */
module.exports = {
  meta: {
    type: "suggestion",
    docs: {
      description: "Enforce cyclomatic complexity threshold",
      category: "Maintainability",
      recommended: false
    },
    schema: [
      {
        type: "object",
        properties: { max: { type: "number" } },
        additionalProperties: false
      }
    ],
    messages: {
      highComplexity:
        "Function '{{name}}' has cyclomatic complexity {{complexity}}, exceeding max of {{max}}."
    }
  },
  create(context) {
    const max = context.options[0]?.max || 10;

    function calculateComplexity(body) {
      let complexity = 1;
      context.getSourceCode().getTokens(body).forEach(token => {
        if (["if", "for", "while", "case", "catch", "&&", "||"].includes(token.value)) {
          complexity++;
        }
      });
      return complexity;
    }

    return {
      FunctionDeclaration(node) {
        const complexity = calculateComplexity(node.body);
        if (complexity > max) {
          context.report({
            node,
            messageId: "highComplexity",
            data: { name: node.id.name, complexity, max }
          });
        }
      },
      FunctionExpression(node) {
        const complexity = calculateComplexity(node.body);
        if (complexity > max) {
          context.report({
            node,
            messageId: "highComplexity",
            data: { name: "(anonymous)", complexity, max }
          });
        }
      }
    };
  }
};
