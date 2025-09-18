const { defineConfig } = require("cypress");

module.exports = defineConfig({
  e2e: {
    baseUrl: process.env.BASE_URL || "http://localhost:3000",
    setupNodeEvents(on, config) {
      require("./plugins/index")(on, config);
      return config;
    },
    specPattern: "cypress/e2e/**/*.cy.{js,jsx,ts,tsx}",
    supportFile: "cypress/support/e2e.js",
  },
  retries: {
    runMode: 2,
    openMode: 1,
  },
  video: true,
  screenshotsFolder: "cypress/screenshots",
  videosFolder: "cypress/videos",
});
