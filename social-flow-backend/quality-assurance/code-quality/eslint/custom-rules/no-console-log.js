/**
 * Rule: no-console-log
 * Disallows usage of console.log in production code.
 */
module.exports = {
  meta: {
    type: "problem",
    docs: {
      description: "Disallow console.log statements",
      category: "Best Practices",
      recommended: true
    },
    messages: {
      noConsole: "Unexpected console.log detected. Use a logger instead."
    }
  },
  create(context) {
    return {
      CallExpression(node) {
        if (
          node.callee.object?.name === "console" &&
          node.callee.property?.name === "log"
        ) {
          context.report({ node, messageId: "noConsole" });
        }
      }
    };
  }
};
