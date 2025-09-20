# API Overview

This document lists core HTTP APIs and how to obtain OpenAPI specs for client generation.

Core services and primary endpoints
- API Gateway / app
  - GET /health
  - POST /auth/signup
  - POST /auth/login
  - GET /me
- Recommendation Service (FastAPI)
  - GET /recommendations/{user_id}?limit=
  - GET /trending?limit=
  - POST /feedback
  - OpenAPI: api-specs/rest/recommendation-service/openapi.yaml
- Video Service
  - POST /upload (multipart)
  - GET /videos/{video_id}
  - OpenAPI: api-specs/rest/video-service/openapi.yaml
- User Service
  - POST /signup
  - POST /login
  - GET /users/{id}
  - OpenAPI: api-specs/rest/user-service/openapi.yaml

Generating clients
- Use the OpenAPI YAML files in api-specs/rest/ with codegen tools (openapi-generator, Swagger Codegen).
- Example:
  openapi-generator-cli generate -i api-specs/rest/recommendation-service/openapi.yaml -g python -o clients/recommendation

Auth
- Services use bearer JWT tokens (see app/core/auth.py). Include header:
  Authorization: Bearer <access_token>

Notes
- The OpenAPI files are minimal starting points; run-time contract may evolve — prefer stable API versions for mobile clients.
