/**
 * Quick rule test runner (lightweight without Jest/Mocha)
 * Just validates custom rules with dummy code.
 */
const { ESLint } = require("eslint");

(async function testRules() {
  const eslint = new ESLint({
    useEslintrc: true,
    overrideConfig: {
      plugins: ["custom-rules"],
      rules: {
        "custom-rules/no-console-log": "error",
        "custom-rules/max-cyclomatic-complexity": ["error", { max: 2 }],
        "custom-rules/naming-convention": "warn"
      }
    }
  });

  const code = `
    class bad_name {}
    function test() {
      console.log("debugging");
      if (true) { if (false) {} }
    }
  `;

  const results = await eslint.lintText(code, { filePath: "test.js" });
  console.log(JSON.stringify(results[0].messages, null, 2));
})();
