/**
 * Helper functions for Cypress tests.
 */

export const randomString = (length = 8) => {
  return Math.random().toString(36).substring(2, 2 + length);
};

export const timestamp = () => {
  return new Date().toISOString();
};
