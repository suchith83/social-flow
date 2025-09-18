/// <reference types="cypress" />

describe("Regression Tests", () => {
  it("dashboard loads with widgets", () => {
    cy.login("admin", "password123");
    cy.visit("/dashboard");
    cy.get(".widget").should("have.length.greaterThan", 2);
  });

  it("user settings can be updated", () => {
    cy.login("user", "userpass");
    cy.visit("/settings");
    cy.get("input[name='email']").clear().type("newemail@example.com");
    cy.get("button[type='submit']").click();
    cy.contains("Settings updated").should("be.visible");
  });
});
