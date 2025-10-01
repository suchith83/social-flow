# Social Flow Backend - Deployment Checklist

## Pre-Deployment Checklist

### 1. Code & Version Control ✓
- [ ] All code is committed to version control
- [ ] No sensitive data in code (passwords, API keys, etc.)
- [ ] Version tag created (e.g., v1.0.0)
- [ ] CHANGELOG.md updated
- [ ] Release notes prepared

### 2. Environment Configuration
- [ ] Production environment variables configured
- [ ] `.env` file created (not committed to git)
- [ ] Database connection string verified
- [ ] Redis connection string verified
- [ ] AWS credentials configured (S3, MediaConvert)
- [ ] Stripe API keys (production mode)
- [ ] JWT secret keys generated (strong, random)
- [ ] API rate limit settings configured
- [ ] CORS origins configured for production domains

**Critical Environment Variables:**
```bash
# Database
DATABASE_URL=postgresql://user:pass@host:5432/socialflow
DATABASE_POOL_SIZE=20
DATABASE_MAX_OVERFLOW=10

# Redis
REDIS_URL=redis://host:6379/0

# AWS
AWS_ACCESS_KEY_ID=your_key
AWS_SECRET_ACCESS_KEY=your_secret
AWS_REGION=us-east-1
AWS_S3_BUCKET=socialflow-media
AWS_CLOUDFRONT_DOMAIN=cdn.socialflow.com

# Stripe
STRIPE_API_KEY=sk_live_...
STRIPE_WEBHOOK_SECRET=whsec_...

# Security
JWT_SECRET_KEY=very_strong_random_secret_here
JWT_ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=60
REFRESH_TOKEN_EXPIRE_DAYS=7

# Application
ENVIRONMENT=production
DEBUG=false
API_V1_PREFIX=/api/v1
ALLOWED_HOSTS=api.socialflow.com,socialflow.com
```

### 3. Database
- [ ] Production database created
- [ ] Database user created with appropriate permissions
- [ ] Connection pooling configured
- [ ] Database backups configured (automated)
- [ ] Run all migrations: `alembic upgrade head`
- [ ] Verify migrations applied: `alembic current`
- [ ] Database indexes created (32 indexes from migration)
- [ ] Full-text search indexes created
- [ ] Test database connectivity
- [ ] Read replicas configured (optional, for scaling)

**Migration Commands:**
```bash
# Check current version
alembic current

# Run migrations
alembic upgrade head

# Verify all tables created
psql -d socialflow -c "\dt"

# Verify indexes
psql -d socialflow -c "\di"
```

### 4. Redis
- [ ] Redis instance deployed
- [ ] Redis authentication configured
- [ ] Redis persistence enabled (AOF or RDB)
- [ ] Redis max memory policy set
- [ ] Test Redis connectivity
- [ ] Redis Sentinel configured (for HA - optional)

**Redis Configuration Check:**
```bash
redis-cli ping
# Expected: PONG

redis-cli INFO server
redis-cli INFO memory
```

### 5. AWS Services
- [ ] S3 bucket created
- [ ] S3 bucket policy configured (public read for media)
- [ ] S3 lifecycle policies set (auto-delete old files)
- [ ] CloudFront distribution created
- [ ] CloudFront origin configured (S3 bucket)
- [ ] SSL certificate for CloudFront (ACM)
- [ ] MediaConvert role created
- [ ] MediaConvert job templates created

**S3 Bucket Policy:**
```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "PublicReadGetObject",
      "Effect": "Allow",
      "Principal": "*",
      "Action": "s3:GetObject",
      "Resource": "arn:aws:s3:::socialflow-media/public/*"
    }
  ]
}
```

### 6. Docker Images
- [ ] Docker images built for production
- [ ] Images tagged with version number
- [ ] Images pushed to container registry (Docker Hub, ECR, GCR)
- [ ] Multi-stage builds optimized
- [ ] Image size optimized
- [ ] Security scan passed (no critical vulnerabilities)

**Build Commands:**
```bash
# Build API image
docker build -t socialflow-api:v1.0.0 -f Dockerfile .

# Build ML service image
docker build -t socialflow-ml:v1.0.0 -f Dockerfile.ml .

# Tag for registry
docker tag socialflow-api:v1.0.0 your-registry/socialflow-api:v1.0.0

# Push to registry
docker push your-registry/socialflow-api:v1.0.0
```

### 7. Kubernetes (if applicable)
- [ ] Cluster created and configured
- [ ] kubectl configured to access cluster
- [ ] Namespace created: `kubectl create namespace socialflow`
- [ ] Secrets created (database, Redis, AWS, Stripe)
- [ ] ConfigMaps created
- [ ] Persistent volumes created (if needed)
- [ ] Ingress controller installed (nginx, traefik)
- [ ] SSL certificates configured (cert-manager)
- [ ] Resource limits set (CPU, memory)
- [ ] Horizontal Pod Autoscaler configured
- [ ] Network policies configured (optional)

**Kubernetes Secret Creation:**
```bash
kubectl create secret generic socialflow-secrets \
  --from-literal=database-url=postgresql://... \
  --from-literal=redis-url=redis://... \
  --from-literal=jwt-secret=... \
  --from-literal=stripe-api-key=... \
  -n socialflow
```

### 8. Load Balancer & SSL
- [ ] Load balancer created (ALB, nginx, etc.)
- [ ] Health check endpoint configured: `/api/v1/health`
- [ ] SSL certificate installed
- [ ] HTTPS enforced (redirect HTTP to HTTPS)
- [ ] Domain name configured (api.socialflow.com)
- [ ] DNS records updated (A record, CNAME)
- [ ] SSL certificate auto-renewal configured (Let's Encrypt)

### 9. Monitoring & Logging
- [ ] Prometheus installed and configured
- [ ] Grafana dashboards created
- [ ] Alerting rules configured
- [ ] Log aggregation configured (ELK, CloudWatch, etc.)
- [ ] Error tracking configured (Sentry)
- [ ] Application insights configured
- [ ] Uptime monitoring (Pingdom, UptimeRobot)
- [ ] Status page created (status.socialflow.com)

**Key Metrics to Monitor:**
- API response time (p50, p95, p99)
- Request rate (requests per second)
- Error rate (5xx errors)
- Database connection pool usage
- Redis memory usage
- Celery queue length
- CPU and memory usage
- Disk usage

### 10. Celery Workers
- [ ] Celery workers deployed
- [ ] Celery beat scheduler deployed (for periodic tasks)
- [ ] Flower monitoring UI deployed (optional)
- [ ] Worker scaling configured
- [ ] Dead letter queue configured
- [ ] Task timeout configured
- [ ] Task retry policy configured

**Celery Deployment:**
```bash
# Start Celery worker
celery -A app.workers.celery_app worker \
  --loglevel=info \
  --concurrency=4 \
  --queue=default,video_processing,notifications,ml_tasks,analytics

# Start Celery beat
celery -A app.workers.celery_app beat \
  --loglevel=info \
  --scheduler=celery.beat:PersistentScheduler

# Start Flower (monitoring)
celery -A app.workers.celery_app flower \
  --port=5555
```

### 11. Security Hardening
- [ ] Firewall rules configured (only allow necessary ports)
- [ ] SSH access restricted (key-based only)
- [ ] Database access restricted (whitelist IPs)
- [ ] Redis protected mode enabled
- [ ] API rate limiting enabled
- [ ] CORS properly configured
- [ ] Security headers enabled (HSTS, CSP, X-Frame-Options)
- [ ] Secrets not in environment variables (use secret manager)
- [ ] DDoS protection enabled (CloudFlare, AWS Shield)
- [ ] Web Application Firewall (WAF) configured

**Security Headers:**
```python
# FastAPI middleware
@app.middleware("http")
async def add_security_headers(request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    return response
```

### 12. Testing
- [ ] All unit tests passing: `pytest tests/unit/`
- [ ] All integration tests passing: `pytest tests/integration/`
- [ ] End-to-end tests passing: `pytest tests/e2e/`
- [ ] Load tests performed: `locust -f tests/performance/locustfile.py`
- [ ] Security tests passed
- [ ] Smoke tests passed in staging environment
- [ ] Performance benchmarks met

**Test Commands:**
```bash
# Run all tests
pytest tests/ -v --cov=app

# Run smoke tests against staging
API_BASE_URL=https://staging-api.socialflow.com pytest tests/e2e/test_smoke.py -v

# Run load tests
locust -f tests/performance/locustfile.py --host=https://staging-api.socialflow.com
```

### 13. Backups
- [ ] Database backup strategy defined
- [ ] Automated daily backups configured
- [ ] Backup retention policy defined (30 days recommended)
- [ ] Backup restore tested (verify backups are valid)
- [ ] S3 bucket versioning enabled
- [ ] S3 cross-region replication (optional, for DR)

**Backup Commands:**
```bash
# Backup PostgreSQL
pg_dump -h localhost -U socialflow -d socialflow > backup_$(date +%Y%m%d).sql

# Backup to S3
aws s3 cp backup_$(date +%Y%m%d).sql s3://socialflow-backups/

# Test restore
psql -h localhost -U socialflow -d socialflow_test < backup_20250115.sql
```

### 14. Documentation
- [ ] API documentation published (https://docs.socialflow.com)
- [ ] Postman collection shared
- [ ] SDK documentation available
- [ ] Webhook documentation published
- [ ] Rate limit documentation clear
- [ ] Status page live (https://status.socialflow.com)
- [ ] Support email configured (support@socialflow.com)
- [ ] Changelog published

### 15. CI/CD Pipeline
- [ ] GitHub Actions workflow configured
- [ ] Automated tests run on every commit
- [ ] Automated deployment to staging on merge to `develop`
- [ ] Manual approval required for production deployment
- [ ] Rollback procedure documented
- [ ] Deployment notifications configured (Slack, email)

### 16. Stripe Integration
- [ ] Stripe account verified (production mode)
- [ ] Payment methods configured
- [ ] Subscription plans created
- [ ] Webhook endpoint configured: `https://api.socialflow.com/api/v1/payments/stripe/webhooks`
- [ ] Webhook signing secret saved
- [ ] Test payment in production environment
- [ ] Refund policy configured
- [ ] Tax calculation configured (Stripe Tax)

### 17. Third-party Services
- [ ] SendGrid configured (email notifications)
- [ ] Twilio configured (SMS, 2FA)
- [ ] Firebase Cloud Messaging configured (push notifications)
- [ ] Sentry configured (error tracking)
- [ ] Google OAuth credentials (production)
- [ ] Facebook OAuth credentials (production)
- [ ] GitHub OAuth credentials (production)

## Deployment Day Checklist

### 1. Pre-Deployment (T-2 hours)
- [ ] All team members notified
- [ ] Maintenance window announced (if applicable)
- [ ] Status page updated: "Deployment in progress"
- [ ] Deployment runbook reviewed
- [ ] Rollback plan reviewed
- [ ] On-call engineer assigned

### 2. Deployment (T-0)
- [ ] Database migrations applied
- [ ] Docker containers deployed
- [ ] Kubernetes pods rolled out
- [ ] Health checks passing
- [ ] Load balancer connected
- [ ] DNS updated (if needed)
- [ ] SSL certificate verified

### 3. Smoke Tests (T+15 minutes)
- [ ] API health endpoint responding: `curl https://api.socialflow.com/api/v1/health`
- [ ] User registration working
- [ ] User login working
- [ ] Post creation working
- [ ] Video upload working (small file)
- [ ] Payment flow working (test transaction)
- [ ] Notifications sending
- [ ] Real-time features working (WebSocket)

### 4. Monitoring (T+30 minutes)
- [ ] API response times normal
- [ ] Error rate acceptable (< 1%)
- [ ] Database queries performing well
- [ ] Redis cache hit rate healthy
- [ ] Celery workers processing tasks
- [ ] No critical errors in logs
- [ ] No alerts triggered

### 5. Post-Deployment (T+1 hour)
- [ ] Status page updated: "All systems operational"
- [ ] Team notified of successful deployment
- [ ] Deployment documentation updated
- [ ] Post-mortem scheduled (if issues occurred)
- [ ] Monitor for next 24 hours

## Rollback Procedure

If deployment fails or critical issues arise:

### 1. Immediate Actions
1. Update status page: "Service degradation"
2. Notify team
3. Stop deployment if in progress

### 2. Rollback Steps
```bash
# Kubernetes rollback
kubectl rollout undo deployment/socialflow-api -n socialflow

# Docker Compose rollback
docker-compose down
git checkout v1.0.0
docker-compose up -d

# Database rollback (if needed - be cautious!)
alembic downgrade -1  # Downgrade one version
```

### 3. Verification
- [ ] Health checks passing
- [ ] Smoke tests passing
- [ ] Error rate back to normal
- [ ] Users can access the system

### 4. Communication
- [ ] Status page updated
- [ ] Users notified (if affected)
- [ ] Post-mortem scheduled

## Post-Deployment Monitoring (24 hours)

### Hour 1-4 (Critical Monitoring)
- [ ] Check health endpoint every 5 minutes
- [ ] Monitor error rate (should be < 1%)
- [ ] Watch for memory leaks
- [ ] Check database connection pool
- [ ] Verify Celery tasks processing

### Hour 4-12 (Active Monitoring)
- [ ] Check metrics every 15 minutes
- [ ] Review error logs
- [ ] Monitor user feedback
- [ ] Check payment transactions
- [ ] Verify video processing

### Hour 12-24 (Passive Monitoring)
- [ ] Set up automated alerts
- [ ] Review daily metrics
- [ ] Check system stability
- [ ] Prepare performance report

## Success Criteria

Deployment is considered successful when:

1. **Health Checks**
   - All health endpoints responding with 200 OK
   - All dependencies (DB, Redis, S3) healthy

2. **Performance**
   - API response time < 200ms (p95)
   - Error rate < 1%
   - No 5xx errors

3. **Functionality**
   - All critical paths working:
     - User registration and login
     - Post creation and retrieval
     - Video upload and streaming
     - Payment processing
     - Notifications sending

4. **Stability**
   - No crashes or restarts
   - Memory usage stable
   - CPU usage within limits
   - No resource leaks

5. **User Experience**
   - No user-reported issues
   - Response times acceptable
   - All features accessible

## Emergency Contacts

- **DevOps Lead:** devops-lead@socialflow.com
- **Backend Lead:** backend-lead@socialflow.com
- **On-Call Engineer:** +1-xxx-xxx-xxxx
- **Platform Status:** https://status.socialflow.com
- **Incident Slack:** #incidents

## Deployment Approval

Before deploying to production, obtain approvals from:

- [ ] Engineering Lead
- [ ] DevOps Lead
- [ ] Product Manager
- [ ] CTO (for major releases)

**Approval Date:** _______________

**Deployment Date:** _______________

**Deployed By:** _______________

---

## Notes

Use this space to document any deployment-specific notes, issues encountered, or deviations from the standard procedure:

```
Deployment Notes:
-----------------









```

---

**Deployment Status:** ⬜ Not Started | 🔄 In Progress | ✅ Complete | ❌ Failed

**Last Updated:** January 2025
