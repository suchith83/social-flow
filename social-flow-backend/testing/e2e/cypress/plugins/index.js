/// <reference types="cypress" />

/**
 * Cypress plugins file
 * Used to hook into Node events and configure reporters, CI/CD, etc.
 */

const { beforeRunHook, afterRunHook } = require("cypress-mochawesome-reporter/lib");

module.exports = (on, config) => {
  on("before:run", async (details) => {
    await beforeRunHook(details);
  });

  on("after:run", async () => {
    await afterRunHook();
  });

  return config;
};
