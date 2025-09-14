/**
 * Rule: naming-convention
 * Enforces naming standards: variables lowerCamelCase, classes UpperCamelCase.
 */
module.exports = {
  meta: {
    type: "suggestion",
    docs: {
      description: "Enforce naming conventions for variables and classes",
      category: "Stylistic Issues",
      recommended: false
    },
    messages: {
      invalidVariable: "Variable '{{name}}' should be in lowerCamelCase.",
      invalidClass: "Class '{{name}}' should be in UpperCamelCase."
    }
  },
  create(context) {
    const camelCase = /^[a-z][a-zA-Z0-9]*$/;
    const pascalCase = /^[A-Z][a-zA-Z0-9]*$/;

    return {
      VariableDeclarator(node) {
        if (node.id.name && !camelCase.test(node.id.name)) {
          context.report({
            node,
            messageId: "invalidVariable",
            data: { name: node.id.name }
          });
        }
      },
      ClassDeclaration(node) {
        if (node.id.name && !pascalCase.test(node.id.name)) {
          context.report({
            node,
            messageId: "invalidClass",
            data: { name: node.id.name }
          });
        }
      }
    };
  }
};
