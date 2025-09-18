/// <reference types="cypress" />

describe("Authentication Tests", () => {
  beforeEach(() => {
    cy.fixture("users").as("users");
  });

  it("should login successfully with valid credentials", function () {
    cy.login(this.users[0].username, this.users[0].password);
    cy.url().should("include", "/dashboard");
  });

  it("should reject invalid login", () => {
    cy.visit("/login");
    cy.get("input[name='username']").type("wronguser");
    cy.get("input[name='password']").type("wrongpass");
    cy.get("button[type='submit']").click();
    cy.contains("Invalid credentials").should("be.visible");
  });
});
