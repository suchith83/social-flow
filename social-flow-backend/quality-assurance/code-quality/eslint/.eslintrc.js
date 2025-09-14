/**
 * Advanced ESLint configuration
 * Enforces strict TypeScript/JavaScript code quality
 */
module.exports = {
  root: true,
  parser: "@typescript-eslint/parser",
  parserOptions: {
    ecmaVersion: "latest",
    sourceType: "module",
    ecmaFeatures: {
      jsx: true
    }
  },
  env: {
    es2021: true,
    node: true,
    browser: true
  },
  plugins: [
    "@typescript-eslint",
    "react",
    "react-hooks",
    "jsx-a11y",
    "import",
    "custom-rules" // our custom rules folder
  ],
  extends: [
    "eslint:recommended",
    "plugin:@typescript-eslint/recommended",
    "plugin:react/recommended",
    "plugin:react-hooks/recommended",
    "plugin:jsx-a11y/recommended",
    "plugin:import/recommended"
  ],
  rules: {
    // Core best practices
    "eqeqeq": ["error", "always"],
    "no-unused-vars": "warn",
    "prefer-const": "error",
    "no-var": "error",

    // React-specific
    "react/prop-types": "off",
    "react/react-in-jsx-scope": "off",

    // Import rules
    "import/no-unresolved": "error",
    "import/order": ["error", {
      "groups": [["builtin", "external", "internal"]],
      "alphabetize": { "order": "asc", "caseInsensitive": true }
    }],

    // Custom rules
    "custom-rules/no-console-log": "error",
    "custom-rules/max-cyclomatic-complexity": ["error", { "max": 10 }],
    "custom-rules/naming-convention": "warn"
  },
  settings: {
    react: {
      version: "detect"
    }
  }
};
