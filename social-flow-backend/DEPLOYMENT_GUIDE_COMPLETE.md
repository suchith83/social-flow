# Deployment Guide - Social Flow Backend

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Initial Setup](#initial-setup)
3. [Infrastructure Deployment](#infrastructure-deployment)
4. [Application Deployment](#application-deployment)
5. [Database Migrations](#database-migrations)
6. [Post-Deployment Verification](#post-deployment-verification)
7. [Rollback Procedures](#rollback-procedures)
8. [Troubleshooting](#troubleshooting)

## Prerequisites

### Required Tools

Install the following tools before deployment:

```powershell
# AWS CLI
winget install Amazon.AWSCLI

# Terraform
winget install HashiCorp.Terraform

# Docker Desktop
winget install Docker.DockerDesktop

# Git
winget install Git.Git
```

### AWS Account Setup

1. **Create AWS Account**
   - Sign up at https://aws.amazon.com
   - Enable billing alerts
   - Set up MFA for root account

2. **Create IAM User**
   ```bash
   # Required permissions:
   - AmazonEC2FullAccess
   - AmazonRDSFullAccess
   - AmazonS3FullAccess
   - AmazonECSFullAccess
   - AmazonElastiCacheFullAccess
   - CloudWatchFullAccess
   - IAMFullAccess (for role creation)
   ```

3. **Configure AWS CLI**
   ```powershell
   aws configure
   # Enter:
   # - AWS Access Key ID
   # - AWS Secret Access Key
   # - Default region: us-east-1
   # - Default output: json
   ```

4. **Verify Configuration**
   ```powershell
   aws sts get-caller-identity
   # Should return your account details
   ```

### S3 Backend Setup

Create S3 bucket and DynamoDB table for Terraform state:

```powershell
# Create S3 bucket for state
aws s3api create-bucket `
  --bucket social-flow-terraform-state `
  --region us-east-1

# Enable versioning
aws s3api put-bucket-versioning `
  --bucket social-flow-terraform-state `
  --versioning-configuration Status=Enabled

# Enable encryption
aws s3api put-bucket-encryption `
  --bucket social-flow-terraform-state `
  --server-side-encryption-configuration '{
    "Rules": [{
      "ApplyServerSideEncryptionByDefault": {
        "SSEAlgorithm": "AES256"
      }
    }]
  }'

# Create DynamoDB table for state locking
aws dynamodb create-table `
  --table-name social-flow-terraform-locks `
  --attribute-definitions AttributeName=LockID,AttributeType=S `
  --key-schema AttributeName=LockID,KeyType=HASH `
  --billing-mode PAY_PER_REQUEST `
  --region us-east-1
```

### Secrets Setup

Create secrets in AWS Secrets Manager:

```powershell
# Database password
aws secretsmanager create-secret `
  --name /social-flow/dev/database/password `
  --secret-string "YOUR_SECURE_PASSWORD_HERE"

# Application secrets
aws secretsmanager create-secret `
  --name /social-flow/dev/app/secrets `
  --secret-string '{
    "SECRET_KEY": "your-secret-key-here",
    "JWT_SECRET_KEY": "your-jwt-secret-here",
    "STRIPE_API_KEY": "your-stripe-key-here",
    "SMTP_PASSWORD": "your-smtp-password-here"
  }'
```

## Initial Setup

### 1. Clone Repository

```powershell
git clone https://github.com/your-org/social-flow-backend.git
cd social-flow-backend
```

### 2. Environment Configuration

Create a `.env.production` file:

```bash
# Application
ENVIRONMENT=production
DEBUG=false
SECRET_KEY=<from secrets manager>
JWT_SECRET_KEY=<from secrets manager>

# Database (will be populated after infrastructure deployment)
DATABASE_URL=postgresql+asyncpg://user:pass@host:5432/social_flow

# Redis (will be populated after infrastructure deployment)
REDIS_URL=redis://host:6379/0

# AWS
AWS_REGION=us-east-1
AWS_ACCESS_KEY_ID=<your-key>
AWS_SECRET_ACCESS_KEY=<your-secret>
S3_BUCKET_VIDEOS=<will-be-created>
S3_BUCKET_UPLOADS=<will-be-created>

# Stripe
STRIPE_API_KEY=<from secrets manager>
STRIPE_WEBHOOK_SECRET=<from stripe dashboard>

# Email
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=noreply@socialflow.com
SMTP_PASSWORD=<from secrets manager>
```

### 3. Build Docker Image

Build the optimized production image:

```powershell
# Build the image
docker build -f Dockerfile.optimized --target runtime -t social-flow-backend:latest .

# Test locally
docker run -p 8000:8000 --env-file .env.production social-flow-backend:latest
```

### 4. Create ECR Repository

```powershell
# Create repository
aws ecr create-repository `
  --repository-name social-flow-backend `
  --region us-east-1

# Get login credentials
aws ecr get-login-password --region us-east-1 | `
  docker login --username AWS --password-stdin `
  123456789012.dkr.ecr.us-east-1.amazonaws.com

# Tag and push image
$ECR_URI = "123456789012.dkr.ecr.us-east-1.amazonaws.com/social-flow-backend"
docker tag social-flow-backend:latest "${ECR_URI}:latest"
docker push "${ECR_URI}:latest"
```

## Infrastructure Deployment

### Development Environment

```powershell
cd deployment/terraform

# Initialize Terraform
terraform init

# Validate configuration
terraform validate

# Plan deployment
terraform plan -var-file=environments/dev.tfvars -out=dev.tfplan

# Review the plan carefully
# Verify:
# - Resource counts
# - Instance types
# - Cost estimates

# Apply the plan
terraform apply dev.tfplan

# Save outputs
terraform output -json > outputs/dev-outputs.json
```

### Staging Environment

```powershell
# Switch workspace (if using workspaces)
terraform workspace new staging
# OR use separate state files with -var-file

# Plan deployment
terraform plan -var-file=environments/staging.tfvars -out=staging.tfplan

# Apply
terraform apply staging.tfplan

# Save outputs
terraform output -json > outputs/staging-outputs.json
```

### Production Environment

```powershell
# Create production workspace
terraform workspace new production

# Plan with production variables
terraform plan -var-file=environments/prod.tfvars -out=prod.tfplan

# Review plan thoroughly
# Check:
# - Multi-AZ enabled
# - Backup retention
# - Auto-scaling limits
# - Monitoring enabled

# Apply (requires confirmation)
terraform apply prod.tfplan

# Save outputs
terraform output -json > outputs/prod-outputs.json
```

### Retrieve Infrastructure Outputs

```powershell
# Get ALB DNS name
$ALB_DNS = terraform output -raw alb_dns_name

# Get database endpoint
$DB_ENDPOINT = terraform output -raw database_endpoint

# Get Redis endpoint
$REDIS_ENDPOINT = terraform output -raw redis_endpoint

# Update .env.production with these values
```

## Application Deployment

### Update Environment Variables

Update `.env.production` with infrastructure outputs:

```bash
DATABASE_URL=postgresql+asyncpg://postgres:${DB_PASSWORD}@${DB_ENDPOINT}/social_flow
REDIS_URL=redis://${REDIS_ENDPOINT}:6379/0
```

### Deploy to ECS

#### Method 1: Using AWS CLI

```powershell
# Register new task definition
aws ecs register-task-definition `
  --cli-input-json file://deployment/ecs/task-definition.json

# Update ECS service
aws ecs update-service `
  --cluster social-flow-prod `
  --service social-flow-app `
  --task-definition social-flow-app:LATEST `
  --force-new-deployment

# Wait for deployment to complete
aws ecs wait services-stable `
  --cluster social-flow-prod `
  --services social-flow-app
```

#### Method 2: Using GitHub Actions

Push to `main` branch to trigger automatic deployment:

```powershell
git add .
git commit -m "Deploy to production"
git push origin main

# Monitor deployment in GitHub Actions
# https://github.com/your-org/social-flow-backend/actions
```

### Verify Deployment

```powershell
# Check service status
aws ecs describe-services `
  --cluster social-flow-prod `
  --services social-flow-app

# Check running tasks
aws ecs list-tasks `
  --cluster social-flow-prod `
  --service-name social-flow-app

# View task logs
$TASK_ID = "your-task-id"
aws logs tail /ecs/social-flow-app `
  --follow `
  --format short
```

## Database Migrations

### Run Migrations

#### Method 1: ECS Task

```powershell
# Run migration task
aws ecs run-task `
  --cluster social-flow-prod `
  --task-definition migration-task `
  --launch-type FARGATE `
  --network-configuration "awsvpcConfiguration={
    subnets=[subnet-xxx,subnet-yyy],
    securityGroups=[sg-xxx],
    assignPublicIp=DISABLED
  }"

# Monitor task
$TASK_ARN = "arn:aws:ecs:region:account:task/cluster/task-id"
aws ecs describe-tasks --cluster social-flow-prod --tasks $TASK_ARN
```

#### Method 2: Local Connection (via Bastion)

```powershell
# Connect to bastion host (if configured)
ssh -i key.pem ec2-user@bastion-host

# From bastion, connect to database
psql -h $DB_ENDPOINT -U postgres -d social_flow

# Run migrations
alembic upgrade head
```

### Verify Migrations

```sql
-- Connect to database
psql -h $DB_ENDPOINT -U postgres -d social_flow

-- Check migration version
SELECT * FROM alembic_version;

-- Verify tables exist
\dt

-- Check table counts
SELECT 
  schemaname,
  tablename,
  (SELECT count(*) FROM pg_catalog.pg_class WHERE relname = tablename) as row_count
FROM pg_tables
WHERE schemaname = 'public';
```

## Post-Deployment Verification

### Health Checks

```powershell
# Check application health
curl https://${ALB_DNS}/health

# Expected response:
# {
#   "status": "healthy",
#   "database": "connected",
#   "redis": "connected",
#   "version": "1.0.0"
# }

# Check API docs
curl https://${ALB_DNS}/docs

# Check metrics endpoint
curl https://${ALB_DNS}/metrics
```

### Smoke Tests

Run critical API endpoints:

```powershell
# Test authentication
curl -X POST https://${ALB_DNS}/api/v1/auth/register `
  -H "Content-Type: application/json" `
  -d '{
    "email": "test@example.com",
    "username": "testuser",
    "password": "SecureP@ss123"
  }'

# Test login
curl -X POST https://${ALB_DNS}/api/v1/auth/login `
  -H "Content-Type: application/json" `
  -d '{
    "username": "testuser",
    "password": "SecureP@ss123"
  }'

# Test authenticated endpoint
$TOKEN = "your-jwt-token"
curl https://${ALB_DNS}/api/v1/users/me `
  -H "Authorization: Bearer ${TOKEN}"
```

### Monitoring Verification

```powershell
# Check CloudWatch logs
aws logs tail /ecs/social-flow-app --follow

# Check CloudWatch alarms
aws cloudwatch describe-alarms `
  --alarm-names "social-flow-prod-cpu-high"

# Check ECS metrics
aws cloudwatch get-metric-statistics `
  --namespace AWS/ECS `
  --metric-name CPUUtilization `
  --dimensions Name=ClusterName,Value=social-flow-prod `
  --start-time (Get-Date).AddHours(-1) `
  --end-time (Get-Date) `
  --period 300 `
  --statistics Average
```

### Load Testing

Run basic load test:

```powershell
# Install k6 (if not already installed)
winget install k6

# Run load test
k6 run testing/performance/load_test.js `
  --vus 10 `
  --duration 30s `
  --env BASE_URL=https://${ALB_DNS}

# Expected results:
# - Response time p95 < 500ms
# - Error rate < 1%
# - Throughput > 100 req/s
```

## Rollback Procedures

### ECS Rollback

```powershell
# List task definition revisions
aws ecs list-task-definitions `
  --family-prefix social-flow-app `
  --sort DESC

# Rollback to previous version
$PREVIOUS_VERSION = "social-flow-app:42"
aws ecs update-service `
  --cluster social-flow-prod `
  --service social-flow-app `
  --task-definition $PREVIOUS_VERSION `
  --force-new-deployment

# Wait for rollback to complete
aws ecs wait services-stable `
  --cluster social-flow-prod `
  --services social-flow-app
```

### Database Rollback

```powershell
# Downgrade one migration
alembic downgrade -1

# Downgrade to specific version
alembic downgrade abc123def456

# Restore from snapshot (if needed)
aws rds restore-db-instance-from-db-snapshot `
  --db-instance-identifier social-flow-prod-restored `
  --db-snapshot-identifier social-flow-prod-snapshot-2024-01-15
```

### Infrastructure Rollback

```powershell
# Show Terraform state
terraform show

# Revert to previous state
# Option 1: Restore from S3 version
aws s3api list-object-versions `
  --bucket social-flow-terraform-state `
  --prefix terraform.tfstate

# Option 2: Use Terraform to update resources
terraform plan -var-file=environments/prod.tfvars -out=rollback.tfplan
terraform apply rollback.tfplan
```

## Troubleshooting

### Common Issues

#### Issue: ECS Tasks Not Starting

**Symptoms:**
- Tasks transition to STOPPED immediately
- Health checks failing

**Diagnosis:**
```powershell
# Check stopped tasks
aws ecs list-tasks `
  --cluster social-flow-prod `
  --desired-status STOPPED `
  --max-results 10

# Get task details
aws ecs describe-tasks `
  --cluster social-flow-prod `
  --tasks $TASK_ARN

# Check logs
aws logs tail /ecs/social-flow-app --since 10m
```

**Solutions:**
1. Check IAM role permissions (task execution role)
2. Verify ECR image exists and is accessible
3. Check environment variables in task definition
4. Verify security group allows outbound traffic
5. Check container memory limits

#### Issue: Database Connection Errors

**Symptoms:**
- Application logs show "could not connect to database"
- Health check fails

**Diagnosis:**
```powershell
# Test database connectivity from ECS task
aws ecs execute-command `
  --cluster social-flow-prod `
  --task $TASK_ARN `
  --container app `
  --interactive `
  --command "/bin/bash"

# From container
nc -zv $DB_ENDPOINT 5432
psql -h $DB_ENDPOINT -U postgres -d social_flow
```

**Solutions:**
1. Check security group allows port 5432 from app SG
2. Verify DATABASE_URL environment variable
3. Check RDS instance is available
4. Verify credentials in Secrets Manager
5. Check connection pool settings

#### Issue: High CPU/Memory Usage

**Symptoms:**
- CloudWatch alarms triggered
- Slow response times
- Tasks being killed (OOM)

**Diagnosis:**
```powershell
# Check ECS metrics
aws cloudwatch get-metric-statistics `
  --namespace AWS/ECS `
  --metric-name CPUUtilization `
  --dimensions Name=ServiceName,Value=social-flow-app

# Check container metrics
aws ecs describe-services `
  --cluster social-flow-prod `
  --services social-flow-app | jq '.services[0].deployments[0]'
```

**Solutions:**
1. Scale horizontally (increase task count)
2. Increase task CPU/memory limits
3. Optimize application code
4. Check for memory leaks
5. Review database query performance

#### Issue: Redis Connection Errors

**Symptoms:**
- Application logs show Redis timeouts
- Cache misses increasing

**Diagnosis:**
```powershell
# Check Redis metrics
aws cloudwatch get-metric-statistics `
  --namespace AWS/ElastiCache `
  --metric-name CPUUtilization `
  --dimensions Name=ReplicationGroupId,Value=social-flow-prod

# Test connectivity
# From ECS task:
redis-cli -h $REDIS_ENDPOINT -p 6379 ping
```

**Solutions:**
1. Check security group allows port 6379
2. Verify REDIS_URL environment variable
3. Check Redis cluster status
4. Increase maxclients parameter
5. Review connection pool settings

#### Issue: ALB Health Checks Failing

**Symptoms:**
- Targets marked unhealthy
- 503 errors from ALB

**Diagnosis:**
```powershell
# Check target health
aws elbv2 describe-target-health `
  --target-group-arn $TG_ARN

# Check ALB access logs
aws s3 sync s3://alb-logs-bucket/AWSLogs/ ./logs/
```

**Solutions:**
1. Verify /health endpoint responds correctly
2. Check security group allows traffic from ALB
3. Increase health check thresholds
4. Check application startup time
5. Review container logs for errors

### Getting Help

**Support Channels:**
- DevOps Team: devops@socialflow.com
- Slack: #infrastructure-support
- On-call: PagerDuty alert for critical issues
- Documentation: https://docs.socialflow.com/infrastructure

**Before Contacting Support:**
1. Check CloudWatch logs
2. Review recent deployments
3. Check CloudWatch alarms
4. Gather error messages
5. Document reproduction steps

### Monitoring & Logging

**CloudWatch Dashboards:**
- Infrastructure: https://console.aws.amazon.com/cloudwatch/dashboards/social-flow-infra
- Application: https://console.aws.amazon.com/cloudwatch/dashboards/social-flow-app
- Database: https://console.aws.amazon.com/cloudwatch/dashboards/social-flow-db

**Log Groups:**
- Application: `/ecs/social-flow-app`
- Database: `/aws/rds/instance/social-flow-prod/postgresql`
- Redis: `/aws/elasticache/social-flow-prod`
- VPC Flow Logs: `/aws/vpc/social-flow-prod`

## Best Practices

### Deployment Checklist

- [ ] Run tests locally
- [ ] Review infrastructure changes (terraform plan)
- [ ] Backup database before major changes
- [ ] Deploy during low-traffic window
- [ ] Monitor metrics during deployment
- [ ] Verify health checks passing
- [ ] Run smoke tests
- [ ] Document any issues encountered
- [ ] Notify team of deployment completion

### Security Checklist

- [ ] Secrets stored in Secrets Manager (not in code)
- [ ] Security groups follow least privilege
- [ ] Encryption enabled (at rest and in transit)
- [ ] IAM roles use minimal permissions
- [ ] CloudTrail logging enabled
- [ ] VPC Flow Logs enabled
- [ ] SSL/TLS certificates valid
- [ ] Regular security scanning (Trivy, Safety)

### Maintenance Schedule

**Daily:**
- Review CloudWatch alarms
- Check application error logs
- Monitor cost usage

**Weekly:**
- Review performance metrics
- Check backup status
- Update dependencies

**Monthly:**
- Security audit
- Cost optimization review
- Database performance tuning
- Test disaster recovery

**Quarterly:**
- Full disaster recovery drill
- Infrastructure review
- Security compliance audit
- Capacity planning

## Next Steps

After successful deployment:

1. **Configure Monitoring**: Set up CloudWatch dashboards and alarms
2. **Setup CI/CD**: Configure GitHub Actions for automated deployments
3. **Load Testing**: Run comprehensive performance tests
4. **Security Audit**: Complete security scanning and hardening
5. **Documentation**: Update runbooks and procedures
6. **Training**: Train team on operations and troubleshooting

## Additional Resources

- [Infrastructure Documentation](./INFRASTRUCTURE.md)
- [API Documentation](./API_DOCUMENTATION.md)
- [Architecture Overview](./ARCHITECTURE.md)
- [Security Guide](./SECURITY.md)
- [Monitoring Guide](./MONITORING.md)
- [AWS Well-Architected Framework](https://aws.amazon.com/architecture/well-architected/)
