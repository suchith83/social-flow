/**
 * OAuth2 integration for Google & GitHub
 */

const { google } = require('googleapis');
const axios = require('axios');
const config = require('./config');

module.exports = {
  // Google OAuth2
  getGoogleAuthUrl: () => {
    const oauth2Client = new google.auth.OAuth2(
      config.oauth.googleClientId,
      config.oauth.googleClientSecret,
      config.oauth.redirectUri
    );

    return oauth2Client.generateAuthUrl({
      access_type: 'offline',
      scope: ['profile', 'email'],
    });
  },

  getGoogleUser: async (code) => {
    const oauth2Client = new google.auth.OAuth2(
      config.oauth.googleClientId,
      config.oauth.googleClientSecret,
      config.oauth.redirectUri
    );

    const { tokens } = await oauth2Client.getToken(code);
    oauth2Client.setCredentials(tokens);

    const oauth2 = google.oauth2({
      auth: oauth2Client,
      version: 'v2',
    });

    const { data } = await oauth2.userinfo.get();
    return data;
  },

  // GitHub OAuth2
  getGithubAuthUrl: () =>
    `https://github.com/login/oauth/authorize?client_id=${config.oauth.githubClientId}&redirect_uri=${config.oauth.redirectUri}&scope=user:email`,

  getGithubUser: async (code) => {
    const tokenRes = await axios.post(
      'https://github.com/login/oauth/access_token',
      {
        client_id: config.oauth.githubClientId,
        client_secret: config.oauth.githubClientSecret,
        code,
        redirect_uri: config.oauth.redirectUri,
      },
      { headers: { Accept: 'application/json' } }
    );

    const accessToken = tokenRes.data.access_token;
    const userRes = await axios.get('https://api.github.com/user', {
      headers: { Authorization: `Bearer ${accessToken}` },
    });

    return userRes.data;
  },
};
