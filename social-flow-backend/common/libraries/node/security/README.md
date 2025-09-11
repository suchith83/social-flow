# common/libraries/node/security

A production-ready, modular Node.js security library.

## Features
- Crypto helpers: AES-GCM, HKDF, RSA sign/verify, HMAC
- KMS abstraction with file-based adapter for local development
- Envelope encryption for secrets
- Helmet wrapper & CSP builder
- CSRF double-submit middleware
- Rate limiting (in-memory & Redis)
- Input sanitization (DOMPurify if available)
- JSON-schema validation wrapper (AJV)
- Security audit logging (NDJSON)
- Simple ABAC policy engine

## Usage examples

### Helmet + CSP + CSRF
```js
const security = require('common/libraries/node/security');
const express = require('express');
const cookieParser = require('cookie-parser');

const app = express();
app.use(cookieParser());
app.use(security.helmetWrapper.createHelmetMiddleware());
app.use(security.csp.build({ withNonce: true }));
app.use(security.csrf.csrfCookieSetter());

app.post('/submit', security.csrf.csrfValidator(), (req, res) => {
  // safe to process
  res.send('ok');
});
