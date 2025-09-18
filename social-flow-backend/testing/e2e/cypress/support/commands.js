// Custom Cypress commands for login, API requests, and storage actions

Cypress.Commands.add("login", (username, password) => {
  cy.session([username, password], () => {
    cy.visit("/login");
    cy.get("input[name='username']").type(username);
    cy.get("input[name='password']").type(password);
    cy.get("button[type='submit']").click();
    cy.url().should("include", "/dashboard");
  });
});

Cypress.Commands.add("uploadFile", (filePath, selector) => {
  cy.get(selector).selectFile(filePath, { force: true });
});

Cypress.Commands.add("apiRequest", (method, url, body = null) => {
  return cy.request({
    method,
    url,
    body,
    failOnStatusCode: false,
  });
});
