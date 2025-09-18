/// <reference types="cypress" />

describe("Multi-Cloud Manager", () => {
  beforeEach(() => {
    cy.login("admin", "password123");
  });

  it("should switch providers dynamically", () => {
    cy.visit("/multi-cloud");
    cy.contains("Current Provider: AWS S3").should("be.visible");

    cy.get("#switchProvider").select("Azure Blob");
    cy.contains("Current Provider: Azure Blob").should("be.visible");

    cy.get("#switchProvider").select("Google Cloud Storage");
    cy.contains("Current Provider: Google Cloud Storage").should("be.visible");
  });

  it("should failover when provider is down", () => {
    cy.intercept("GET", "/api/storage/status", { forceNetworkError: true }).as("failover");
    cy.visit("/multi-cloud");
    cy.wait("@failover");
    cy.contains("Failover activated").should("be.visible");
  });
});
