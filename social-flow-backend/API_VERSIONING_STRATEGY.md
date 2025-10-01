# API Versioning Strategy

## Overview

Social Flow API follows a path-based versioning strategy to ensure backward compatibility while allowing for API evolution and improvements.

## Versioning Approach

### Path-Based Versioning

All API endpoints include the version in the URL path:

```
https://api.socialflow.com/api/v1/users/me
https://api.socialflow.com/api/v2/users/me
```

**Advantages:**
- Clear and explicit version identification
- Easy to route different versions to different implementations
- Simple for clients to understand and use
- Can run multiple versions simultaneously

### Version Format

- Format: `v{major}`
- Examples: `v1`, `v2`, `v3`
- Only major versions are included in the path

## Version Lifecycle

### 1. Alpha (Internal Testing)

- **Purpose:** Internal development and testing
- **Stability:** Unstable, breaking changes expected
- **Access:** Limited to development team
- **URL Pattern:** `/api/alpha/*` or `/api/v2-alpha/*`
- **Duration:** 2-4 weeks

### 2. Beta (Early Access)

- **Purpose:** External testing with select partners
- **Stability:** Feature-complete but may have bugs
- **Access:** Opt-in for early adopters
- **URL Pattern:** `/api/beta/*` or `/api/v2-beta/*`
- **Documentation:** Full documentation provided
- **Duration:** 4-8 weeks
- **Breaking Changes:** Allowed but communicated clearly

### 3. Stable (Production)

- **Purpose:** General availability
- **Stability:** Stable, backward compatible
- **Access:** All users
- **URL Pattern:** `/api/v1/*`, `/api/v2/*`
- **Documentation:** Complete and up-to-date
- **Breaking Changes:** Not allowed within major version

### 4. Deprecated

- **Notice Period:** Minimum 6 months before removal
- **Status:** Still functional but not recommended
- **Documentation:** Marked with deprecation notices
- **Headers:** Response includes `X-API-Deprecation-Date` header
- **Support:** Bug fixes only, no new features

### 5. Sunset

- **Final Notice:** 30 days before shutdown
- **Status:** Will be removed completely
- **Migration:** Required to newer version
- **Support:** No support provided

## Change Types

### Breaking Changes (Require New Major Version)

- Removing or renaming endpoints
- Removing or renaming request/response fields
- Changing field data types
- Changing authentication methods
- Modifying error response formats
- Changing HTTP status codes
- Removing query parameters
- Making optional fields required

**Example:**
```
v1: GET /posts/{id}  → Returns { "likes": 100 }
v2: GET /posts/{id}  → Returns { "likes_count": 100 }  # Breaking: renamed field
```

### Non-Breaking Changes (Same Major Version)

- Adding new endpoints
- Adding new optional request parameters
- Adding new response fields
- Adding new HTTP headers
- Expanding enum values
- Making required fields optional
- Performance improvements
- Bug fixes

**Example:**
```
v1: GET /posts/{id}  → Returns { "likes": 100 }
v1: GET /posts/{id}  → Returns { "likes": 100, "shares": 50 }  # Non-breaking: added field
```

## Version Support Policy

### Active Support

- **Current Version (v2):** Full support, all features, bug fixes, security patches
- **Previous Version (v1):** Bug fixes and security patches only
- **Duration:** Minimum 12 months after new version release

### Deprecated Support

- **Duration:** 6 months deprecation period
- **Support Level:** Critical security fixes only
- **Communication:** 
  - API response headers indicate deprecation
  - Email notifications to users
  - Dashboard warnings
  - Documentation updates

### End of Life (EOL)

- **Final Warning:** 30 days before EOL
- **Communication Channels:**
  - Email to all API users
  - API status page announcements
  - Response headers with sunset date
  - Developer blog post

## Migration Strategy

### Version Migration Checklist

1. **Review Changes**
   - Read migration guide
   - Review breaking changes list
   - Check deprecated features

2. **Update Dependencies**
   - Update SDK to latest version
   - Update API endpoint URLs
   - Update authentication if changed

3. **Test in Sandbox**
   - Use test environment
   - Run integration tests
   - Validate responses

4. **Gradual Rollout**
   - Test with small percentage of traffic
   - Monitor error rates
   - Validate metrics

5. **Complete Migration**
   - Switch all traffic to new version
   - Remove old version code
   - Update documentation

### Migration Tools

**Version Comparison Tool:**
```bash
curl https://api.socialflow.com/api/version-diff?from=v1&to=v2
```

**Migration Validator:**
```bash
curl -X POST https://api.socialflow.com/api/migration-check \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -d '{"current_version": "v1", "target_version": "v2"}'
```

## Deprecation Communication

### Response Headers

```http
X-API-Version: v1
X-API-Deprecation-Date: 2025-12-31
X-API-Sunset-Date: 2026-06-30
X-API-Latest-Version: v2
X-API-Migration-Guide: https://docs.socialflow.com/migration/v1-to-v2
```

### Deprecation Notice in Response

```json
{
  "data": { ... },
  "_deprecation": {
    "deprecated": true,
    "deprecation_date": "2025-12-31",
    "sunset_date": "2026-06-30",
    "current_version": "v1",
    "latest_version": "v2",
    "migration_guide": "https://docs.socialflow.com/migration/v1-to-v2",
    "message": "This API version will be sunset on 2026-06-30. Please migrate to v2."
  }
}
```

## Backward Compatibility

### Guarantees

Within the same major version:
- Existing endpoints will not change behavior
- Existing fields will not change data types
- Existing response codes will remain consistent
- Authentication mechanisms will remain compatible

### Best Practices for Clients

1. **Ignore Unknown Fields**
   - New fields may be added to responses
   - Client should ignore fields it doesn't recognize

2. **Don't Rely on Field Order**
   - JSON field order is not guaranteed
   - Parse fields by name, not position

3. **Handle New Enum Values**
   - Enums may have new values added
   - Implement default handling for unknown values

4. **Validate Status Codes**
   - Check HTTP status codes
   - Don't rely on specific error message text

## Version Discovery

### Current Version Endpoint

```http
GET /api/version
```

**Response:**
```json
{
  "current_version": "v2",
  "supported_versions": ["v1", "v2"],
  "deprecated_versions": ["v1"],
  "latest_features": {
    "v2": ["improved-recommendations", "real-time-analytics", "enhanced-moderation"]
  },
  "deprecation_timeline": {
    "v1": {
      "deprecated_date": "2025-12-31",
      "sunset_date": "2026-06-30"
    }
  }
}
```

## Example: v1 to v2 Migration

### Breaking Changes in v2

1. **Authentication:**
   - v1: API Key in header `X-API-Key`
   - v2: JWT Bearer token only

2. **User Endpoints:**
   - v1: `GET /users/{id}` returns `{ "name": "..." }`
   - v2: `GET /users/{id}` returns `{ "full_name": "..." }`

3. **Pagination:**
   - v1: `?page=1&size=20`
   - v2: `?page=1&limit=20&cursor=abc123`

### Migration Code Example

**Before (v1):**
```javascript
const response = await fetch('https://api.socialflow.com/api/v1/users/me', {
  headers: {
    'X-API-Key': 'your-api-key'
  }
});
const data = await response.json();
console.log(data.name); // "John Doe"
```

**After (v2):**
```javascript
const response = await fetch('https://api.socialflow.com/api/v2/users/me', {
  headers: {
    'Authorization': `Bearer ${accessToken}`
  }
});
const data = await response.json();
console.log(data.full_name); // "John Doe"
```

## SDK Support

### Official SDKs

All official SDKs support multiple API versions:

```javascript
// JavaScript/TypeScript
import { SocialFlowClient } from '@socialflow/sdk';

const client = new SocialFlowClient({
  apiVersion: 'v2', // Specify version
  accessToken: 'your-token'
});
```

```python
# Python
from socialflow import Client

client = Client(
    api_version='v2',  # Specify version
    access_token='your-token'
)
```

### Version Pinning

**Recommended:**
```javascript
const client = new SocialFlowClient({
  apiVersion: 'v2', // Pin to specific version
  accessToken: 'your-token'
});
```

**Not Recommended:**
```javascript
const client = new SocialFlowClient({
  apiVersion: 'latest', // Avoid - may break unexpectedly
  accessToken: 'your-token'
});
```

## Monitoring and Analytics

### Version Usage Metrics

Track API version usage:
- Requests per version
- Error rates per version
- Response times per version
- Unique clients per version

### Alerts

Set up alerts for:
- Deprecated version usage spikes
- High error rates on new versions
- Sunset date approaching
- Migration blockers

## Documentation

### Version-Specific Documentation

Each version has its own documentation:
- `https://docs.socialflow.com/api/v1`
- `https://docs.socialflow.com/api/v2`

### Migration Guides

- `https://docs.socialflow.com/migration/v1-to-v2`
- Includes code examples
- Lists all breaking changes
- Provides migration timeline

## Support

### Version-Specific Support

- **v2 (Current):** Full support via all channels
- **v1 (Deprecated):** Email support only
- **v0 (Sunset):** No support

### Contact

- **API Support:** api-support@socialflow.com
- **Migration Help:** migration-support@socialflow.com
- **Status Page:** https://status.socialflow.com
- **Developer Forum:** https://forum.socialflow.com

## Changelog

All API changes are documented in the changelog:
- `https://docs.socialflow.com/changelog`

Format:
```markdown
## v2.0.0 (2025-01-15)

### Breaking Changes
- Renamed `name` field to `full_name` in User model
- Changed pagination parameters

### New Features
- Added real-time analytics endpoints
- Enhanced recommendation engine

### Deprecated
- `/api/v1/users` endpoints
```

## Conclusion

This versioning strategy ensures:
- **Stability:** Existing integrations won't break unexpectedly
- **Evolution:** API can improve and add features
- **Clear Communication:** Deprecation and migration timelines are transparent
- **Developer Experience:** Smooth migration path with tools and support

For questions or migration assistance, contact our API support team.
