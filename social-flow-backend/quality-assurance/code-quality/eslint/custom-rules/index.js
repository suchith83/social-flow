/**
 * Custom ESLint Rule Plugin Index
 * Exports all custom rules so they can be registered in .eslintrc.js
 */
module.exports.rules = {
  "no-console-log": require("./no-console-log"),
  "max-cyclomatic-complexity": require("./max-cyclomatic-complexity"),
  "naming-convention": require("./naming-convention")
};
