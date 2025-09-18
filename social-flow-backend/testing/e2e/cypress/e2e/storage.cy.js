/// <reference types="cypress" />

describe("Cloud Storage Upload/Download", () => {
  beforeEach(() => {
    cy.fixture("testdata").as("data");
  });

  it("uploads file to AWS S3", function () {
    cy.login("admin", "password123");
    cy.visit("/storage/s3");
    cy.uploadFile(this.data.files.sampleTxt, "#uploadInput");
    cy.contains("Upload successful").should("be.visible");
  });

  it("downloads file from Azure Blob", function () {
    cy.login("admin", "password123");
    cy.visit("/storage/azure");
    cy.contains("Download test-azure.txt").click();
    cy.readFile("cypress/downloads/test-azure.txt").should("exist");
  });

  it("uploads file to GCS", function () {
    cy.login("admin", "password123");
    cy.visit("/storage/gcs");
    cy.uploadFile(this.data.files.sampleImage, "#uploadInput");
    cy.contains("Upload successful").should("be.visible");
  });
});
