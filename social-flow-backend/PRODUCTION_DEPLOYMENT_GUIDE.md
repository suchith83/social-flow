# 🚀 Production Deployment Guide - Complete

## Overview

This guide provides complete step-by-step instructions for deploying the SocialFlow backend to production. Follow these instructions carefully to ensure a smooth deployment.

**Deployment Time**: ~4-6 hours (first deployment)  
**Prerequisites Time**: ~2-3 hours  
**Total Time**: ~6-9 hours

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [AWS Infrastructure Setup](#aws-infrastructure-setup)
3. [Database Setup](#database-setup)
4. [Environment Configuration](#environment-configuration)
5. [Docker Deployment](#docker-deployment)
6. [Monitoring Setup](#monitoring-setup)
7. [SSL/TLS Configuration](#ssltls-configuration)
8. [CI/CD Pipeline](#cicd-pipeline)
9. [Post-Deployment](#post-deployment)
10. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### Required Tools

```bash
# Docker & Docker Compose
docker --version  # Should be 24.0+
docker-compose --version  # Should be 2.20+

# AWS CLI
aws --version  # Should be 2.13+
aws configure  # Set up AWS credentials

# Python
python --version  # Should be 3.11+

# Git
git --version  # Should be 2.40+
```

### Required Accounts

- ✅ AWS Account with admin access
- ✅ Stripe Account (for payments)
- ✅ SendGrid Account (for emails)
- ✅ Firebase Project (for push notifications)
- ✅ Domain name registered
- ✅ GitHub repository access

---

## AWS Infrastructure Setup

### Step 1: Create VPC and Security Groups

```bash
# Create VPC
aws ec2 create-vpc \
  --cidr-block 10.0.0.0/16 \
  --tag-specifications 'ResourceType=vpc,Tags=[{Key=Name,Value=socialflow-vpc}]'

# Create public subnet
aws ec2 create-subnet \
  --vpc-id vpc-xxxxx \
  --cidr-block 10.0.1.0/24 \
  --availability-zone us-east-1a

# Create private subnet
aws ec2 create-subnet \
  --vpc-id vpc-xxxxx \
  --cidr-block 10.0.2.0/24 \
  --availability-zone us-east-1b

# Create security group for web servers
aws ec2 create-security-group \
  --group-name socialflow-web \
  --description "Security group for web servers" \
  --vpc-id vpc-xxxxx

# Allow HTTP/HTTPS
aws ec2 authorize-security-group-ingress \
  --group-id sg-xxxxx \
  --protocol tcp \
  --port 80 \
  --cidr 0.0.0.0/0

aws ec2 authorize-security-group-ingress \
  --group-id sg-xxxxx \
  --protocol tcp \
  --port 443 \
  --cidr 0.0.0.0/0
```

### Step 2: Create RDS PostgreSQL Database

```bash
# Create DB subnet group
aws rds create-db-subnet-group \
  --db-subnet-group-name socialflow-db-subnet \
  --db-subnet-group-description "Subnet group for SocialFlow DB" \
  --subnet-ids subnet-xxxxx subnet-yyyyy

# Create RDS instance
aws rds create-db-instance \
  --db-instance-identifier socialflow-db-prod \
  --db-instance-class db.t3.medium \
  --engine postgres \
  --engine-version 15.3 \
  --master-username postgres \
  --master-user-password <SECURE_PASSWORD> \
  --allocated-storage 100 \
  --storage-type gp3 \
  --storage-encrypted \
  --vpc-security-group-ids sg-xxxxx \
  --db-subnet-group-name socialflow-db-subnet \
  --backup-retention-period 7 \
  --preferred-backup-window "03:00-04:00" \
  --preferred-maintenance-window "sun:04:00-sun:05:00" \
  --multi-az \
  --publicly-accessible false
```

### Step 3: Create ElastiCache Redis Cluster

```bash
# Create cache subnet group
aws elasticache create-cache-subnet-group \
  --cache-subnet-group-name socialflow-cache-subnet \
  --cache-subnet-group-description "Subnet group for SocialFlow cache" \
  --subnet-ids subnet-xxxxx subnet-yyyyy

# Create Redis cluster
aws elasticache create-cache-cluster \
  --cache-cluster-id socialflow-redis-prod \
  --cache-node-type cache.t3.medium \
  --engine redis \
  --engine-version 7.0 \
  --num-cache-nodes 1 \
  --cache-subnet-group-name socialflow-cache-subnet \
  --security-group-ids sg-xxxxx \
  --snapshot-retention-limit 5 \
  --preferred-maintenance-window "sun:05:00-sun:06:00"
```

### Step 4: Create S3 Buckets

```bash
# Media storage bucket
aws s3 mb s3://socialflow-media-prod \
  --region us-east-1

# Enable versioning
aws s3api put-bucket-versioning \
  --bucket socialflow-media-prod \
  --versioning-configuration Status=Enabled

# Enable encryption
aws s3api put-bucket-encryption \
  --bucket socialflow-media-prod \
  --server-side-encryption-configuration '{
    "Rules": [{
      "ApplyServerSideEncryptionByDefault": {
        "SSEAlgorithm": "AES256"
      }
    }]
  }'

# Configure CORS
aws s3api put-bucket-cors \
  --bucket socialflow-media-prod \
  --cors-configuration file://s3-cors.json

# Configure lifecycle rules
aws s3api put-bucket-lifecycle-configuration \
  --bucket socialflow-media-prod \
  --lifecycle-configuration file://s3-lifecycle.json
```

### Step 5: Setup AWS MediaConvert

```bash
# Get MediaConvert endpoint
aws mediaconvert describe-endpoints \
  --region us-east-1

# Create MediaConvert role
aws iam create-role \
  --role-name MediaConvertRole \
  --assume-role-policy-document file://mediaconvert-trust-policy.json

# Attach S3 access policy
aws iam put-role-policy \
  --role-name MediaConvertRole \
  --policy-name S3Access \
  --policy-document file://mediaconvert-s3-policy.json
```

### Step 6: Setup AWS IVS for Live Streaming

```bash
# Create IVS channel
aws ivs create-channel \
  --name socialflow-live-prod \
  --latency-mode LOW \
  --type STANDARD

# Note the Channel ARN, Stream Key, and Ingest/Playback URLs
```

---

## Database Setup

### Step 1: Initial Database Configuration

```bash
# Connect to database
psql postgresql://postgres:<PASSWORD>@<RDS_ENDPOINT>:5432/postgres

# Create database
CREATE DATABASE socialflow;

# Create extensions
\c socialflow
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";
CREATE EXTENSION IF NOT EXISTS "btree_gin";
```

### Step 2: Run Migrations

```bash
# Copy .env.production.example to .env.production
cp .env.production.example .env.production

# Edit .env.production with actual values
nano .env.production

# Run Alembic migrations
export DATABASE_URL=postgresql+asyncpg://postgres:<PASSWORD>@<RDS_ENDPOINT>:5432/socialflow
alembic upgrade head
```

### Step 3: Create Database Backups

```bash
# Enable automated backups (already done in RDS creation)
# Create manual snapshot
aws rds create-db-snapshot \
  --db-instance-identifier socialflow-db-prod \
  --db-snapshot-identifier socialflow-initial-snapshot
```

---

## Environment Configuration

### Step 1: Configure Production Environment

```bash
# Copy example file
cp .env.production.example .env.production

# Edit with actual production values
# IMPORTANT: Never commit this file to Git!
nano .env.production
```

### Required Environment Variables

Update these critical variables in `.env.production`:

```env
# Application
SECRET_KEY=<generate-with: openssl rand -hex 32>
JWT_SECRET_KEY=<generate-with: openssl rand -hex 32>

# Database
DATABASE_URL=postgresql+asyncpg://postgres:<PASSWORD>@<RDS_ENDPOINT>:5432/socialflow

# Redis
REDIS_URL=redis://<REDIS_ENDPOINT>:6379/0

# AWS
AWS_ACCESS_KEY_ID=<your-key>
AWS_SECRET_ACCESS_KEY=<your-secret>
S3_BUCKET=socialflow-media-prod
MEDIACONVERT_ENDPOINT=<your-endpoint>
IVS_CHANNEL_ARN=<your-channel-arn>

# Stripe
STRIPE_SECRET_KEY=<your-stripe-key>
STRIPE_WEBHOOK_SECRET=<your-webhook-secret>

# SendGrid
SENDGRID_API_KEY=<your-sendgrid-key>

# Firebase
FIREBASE_CREDENTIALS=<your-firebase-json>

# Sentry
SENTRY_DSN=<your-sentry-dsn>
```

---

## Docker Deployment

### Step 1: Build Docker Images

```bash
# Set build variables
export VERSION=$(git describe --tags --always)
export BUILD_DATE=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
export VCS_REF=$(git rev-parse --short HEAD)

# Build production image
docker build \
  -f Dockerfile.production \
  -t socialflow-backend:$VERSION \
  --build-arg BUILD_DATE=$BUILD_DATE \
  --build-arg VCS_REF=$VCS_REF \
  --build-arg VERSION=$VERSION \
  .

# Tag as latest
docker tag socialflow-backend:$VERSION socialflow-backend:latest
```

### Step 2: Deploy with Docker Compose

```bash
# Pull latest code
git pull origin main

# Load environment variables
export $(cat .env.production | grep -v '^#' | xargs)

# Deploy services
docker-compose -f docker-compose.production.yml up -d

# View logs
docker-compose -f docker-compose.production.yml logs -f
```

### Step 3: Verify Deployment

```bash
# Check service health
docker-compose -f docker-compose.production.yml ps

# Test health endpoint
curl http://localhost:8000/health

# Test API
curl http://localhost:8000/api/v1/health

# Check logs for errors
docker-compose -f docker-compose.production.yml logs web
```

---

## Monitoring Setup

### Step 1: Setup CloudWatch

```bash
# Install boto3
pip install boto3

# Run monitoring setup script
python scripts/setup_monitoring.py

# Verify in AWS Console
# Go to CloudWatch > Dashboards > SocialFlow-Production
```

### Step 2: Setup Sentry

```bash
# Install Sentry SDK (already in requirements.txt)
pip install sentry-sdk

# Configure in app
# Already configured in app/core/config.py

# Test Sentry integration
python -c "import sentry_sdk; sentry_sdk.init('<YOUR_DSN>'); sentry_sdk.capture_message('Test from production')"
```

### Step 3: Setup Log Aggregation

```bash
# Configure CloudWatch Logs
aws logs create-log-group \
  --log-group-name /aws/socialflow/backend

# Create log stream
aws logs create-log-stream \
  --log-group-name /aws/socialflow/backend \
  --log-stream-name production
```

---

## SSL/TLS Configuration

### Step 1: Generate SSL Certificates (Let's Encrypt)

```bash
# Install Certbot
sudo apt-get install certbot

# Generate certificates
sudo certbot certonly --standalone \
  -d socialflow.com \
  -d www.socialflow.com \
  --email admin@socialflow.com \
  --agree-tos

# Copy certificates to nginx/ssl/
sudo cp /etc/letsencrypt/live/socialflow.com/fullchain.pem nginx/ssl/
sudo cp /etc/letsencrypt/live/socialflow.com/privkey.pem nginx/ssl/
sudo cp /etc/letsencrypt/live/socialflow.com/chain.pem nginx/ssl/
```

### Step 2: Configure Auto-Renewal

```bash
# Test renewal
sudo certbot renew --dry-run

# Add cron job for auto-renewal
echo "0 0 1 * * certbot renew --quiet && docker-compose -f docker-compose.production.yml restart nginx" | sudo crontab -
```

---

## CI/CD Pipeline

### Step 1: Configure GitHub Secrets

Go to GitHub Repository > Settings > Secrets and add:

```
AWS_ACCESS_KEY_ID
AWS_SECRET_ACCESS_KEY
DATABASE_URL
REDIS_PASSWORD
SECRET_KEY
STRIPE_SECRET_KEY
SENDGRID_API_KEY
FIREBASE_CREDENTIALS
SENTRY_DSN
```

### Step 2: Enable GitHub Actions

```bash
# Already configured in .github/workflows/ci-cd.yml
# Push to main branch will trigger deployment

git add .
git commit -m "Production deployment setup"
git push origin main
```

---

## Post-Deployment

### Step 1: Run Smoke Tests

```bash
# Test health endpoint
curl https://socialflow.com/health

# Test API documentation
curl https://socialflow.com/docs

# Test authentication
curl -X POST https://socialflow.com/api/v1/auth/register \
  -H "Content-Type: application/json" \
  -d '{"email":"test@example.com","password":"Test123!","username":"testuser"}'
```

### Step 2: Configure DNS

```bash
# Point domain to server IP
# A Record: socialflow.com -> <SERVER_IP>
# A Record: www.socialflow.com -> <SERVER_IP>
# CNAME: api.socialflow.com -> socialflow.com
```

### Step 3: Setup Backups

```bash
# Database backups (already configured in RDS)
# Redis backups (already configured in ElastiCache)

# Create backup script
cat > scripts/backup.sh << 'EOF'
#!/bin/bash
DATE=$(date +%Y%m%d_%H%M%S)
pg_dump $DATABASE_URL > backups/db_$DATE.sql
aws s3 cp backups/db_$DATE.sql s3://socialflow-backups/
EOF

# Add to crontab (daily at 2 AM)
echo "0 2 * * * /app/scripts/backup.sh" | crontab -
```

### Step 4: Performance Tuning

```bash
# Optimize PostgreSQL
# Edit postgresql.conf
shared_buffers = 256MB
effective_cache_size = 1GB
maintenance_work_mem = 64MB
max_connections = 200

# Restart PostgreSQL
sudo systemctl restart postgresql
```

---

## Troubleshooting

### Common Issues

**Issue 1: Database Connection Errors**
```bash
# Check RDS status
aws rds describe-db-instances \
  --db-instance-identifier socialflow-db-prod

# Verify security group rules
# Ensure port 5432 is open to web servers

# Test connection
psql postgresql://postgres:<PASSWORD>@<RDS_ENDPOINT>:5432/socialflow
```

**Issue 2: High Response Times**
```bash
# Check container resources
docker stats

# Increase worker processes
# Edit docker-compose.production.yml
# web: command: gunicorn ... --workers 8

# Scale web service
docker-compose -f docker-compose.production.yml up -d --scale web-replica=4
```

**Issue 3: Celery Tasks Not Running**
```bash
# Check Celery worker status
docker-compose -f docker-compose.production.yml logs celery-worker

# Restart Celery
docker-compose -f docker-compose.production.yml restart celery-worker celery-beat
```

**Issue 4: Redis Connection Errors**
```bash
# Check ElastiCache status
aws elasticache describe-cache-clusters \
  --cache-cluster-id socialflow-redis-prod

# Test Redis connection
redis-cli -h <REDIS_ENDPOINT> ping
```

---

## Production Checklist

Before going live, verify:

- ☐ All environment variables configured
- ☐ Database migrations completed
- ☐ SSL certificates installed
- ☐ Health checks passing
- ☐ Monitoring & alerting configured
- ☐ Backups automated
- ☐ CI/CD pipeline working
- ☐ Load testing completed
- ☐ Security scan passed
- ☐ Documentation updated
- ☐ Team trained on deployment process
- ☐ Rollback plan documented
- ☐ On-call rotation scheduled

---

## Rollback Procedure

If issues occur after deployment:

```bash
# 1. Restore database from snapshot
aws rds restore-db-instance-from-db-snapshot \
  --db-instance-identifier socialflow-db-prod-rollback \
  --db-snapshot-identifier socialflow-pre-deploy-snapshot

# 2. Deploy previous Docker image version
docker-compose -f docker-compose.production.yml down
docker pull socialflow-backend:<previous-version>
docker-compose -f docker-compose.production.yml up -d

# 3. Verify rollback
curl https://socialflow.com/health

# 4. Notify team
echo "Rolled back to version <previous-version>" | mail -s "Deployment Rollback" team@socialflow.com
```

---

## Support & Resources

- **Documentation**: https://docs.socialflow.com
- **Status Page**: https://status.socialflow.com
- **GitHub Issues**: https://github.com/socialflow/backend/issues
- **Slack**: #socialflow-ops
- **On-Call**: PagerDuty rotation

---

**Last Updated**: October 2, 2025  
**Version**: 1.0.0  
**Maintained By**: SocialFlow DevOps Team
