/**
 * API utility functions for Cypress tests.
 */
export const loginApi = (username, password) => {
  return cy.apiRequest("POST", "/api/auth/login", { username, password });
};

export const getStorageStatus = (provider) => {
  return cy.apiRequest("GET", `/api/storage/${provider}/status`);
};
