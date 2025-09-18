// Import custom commands
import "./commands";

// Configure test reporting
import "cypress-mochawesome-reporter/register";

beforeEach(() => {
  cy.log("Starting test...");
});

afterEach(() => {
  cy.log("Test completed.");
});
