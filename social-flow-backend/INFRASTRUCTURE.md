# Infrastructure as Code Documentation

## Overview

This document describes the AWS infrastructure for Social Flow backend, managed through Terraform. The infrastructure is designed for high availability, scalability, and security.

## Architecture Components

### 1. Network Layer (VPC)

**Resources:**
- VPC with configurable CIDR (default: 10.0.0.0/16)
- 3 Public Subnets across availability zones (for ALB)
- 3 Private Subnets across availability zones (for ECS, RDS, Redis)
- Internet Gateway for public subnet internet access
- NAT Gateways for private subnet outbound traffic
  - Single NAT for dev (cost optimization)
  - One NAT per AZ for prod (high availability)
- VPC Flow Logs for network monitoring

**Key Features:**
- Multi-AZ deployment for high availability
- Isolated private subnets for secure workload placement
- Proper routing with public/private route tables
- Network monitoring with CloudWatch logs

### 2. Security Layer (Security Groups)

**Security Groups:**

1. **ALB Security Group**
   - Inbound: HTTP (80), HTTPS (443) from internet
   - Outbound: All traffic
   - Purpose: Allow public access to load balancer

2. **Application Security Group**
   - Inbound: Port 8000 from ALB only
   - Outbound: All traffic
   - Purpose: Restrict app access to ALB traffic only

3. **Database Security Group**
   - Inbound: PostgreSQL (5432) from app security group
   - Outbound: All traffic
   - Purpose: Restrict database access to application only

4. **Redis Security Group**
   - Inbound: Redis (6379) from app security group
   - Outbound: All traffic
   - Purpose: Restrict Redis access to application only

**Security Principles:**
- Principle of least privilege
- Layer-based isolation
- No direct internet access to workloads
- Security group chaining for defense in depth

### 3. Database Layer (RDS PostgreSQL)

**Configuration:**
- Engine: PostgreSQL 15.4
- Instance Types:
  - Dev: db.t3.small
  - Staging: db.t3.medium
  - Production: db.r6g.xlarge (memory-optimized)
- Storage: gp3 SSD with auto-scaling
- Multi-AZ deployment (staging, prod)
- Automated backups (3-30 days retention)
- Performance Insights enabled
- Enhanced Monitoring (60s interval)

**Performance Optimizations:**
- Custom parameter group with tuned settings
- pg_stat_statements for query analysis
- Optimized memory and connection settings
- Query logging for slow queries (>1s)

**Security Features:**
- Encryption at rest (KMS)
- Private subnet placement
- Security group restrictions
- Automated patching
- Deletion protection (prod)

**Monitoring:**
- CloudWatch logs (postgresql, upgrade)
- Performance Insights
- Enhanced Monitoring metrics
- Custom CloudWatch alarms

### 4. Cache Layer (ElastiCache Redis)

**Configuration:**
- Engine: Redis 7.0
- Node Types:
  - Dev: cache.t3.micro
  - Staging: cache.t3.small
  - Production: cache.r6g.large
- Replication:
  - Dev: Single node
  - Staging: 2 nodes (primary + replica)
  - Production: 3 nodes (primary + 2 replicas)
- Automatic failover enabled (staging, prod)
- Multi-AZ deployment (staging, prod)

**Configuration:**
- Custom parameter group
- maxmemory-policy: allkeys-lru
- Connection timeout: 300s
- TCP keepalive: 300s

**Security Features:**
- Encryption at rest
- Encryption in transit
- Private subnet placement
- Security group restrictions

**Monitoring:**
- Slow query logs → CloudWatch
- Engine logs → CloudWatch
- CloudWatch alarms for CPU and memory
- SNS notifications for alerts

### 5. Compute Layer (ECS Fargate)

**Cluster Components:**
- ECS Fargate cluster (serverless containers)
- Application Load Balancer (ALB)
- Auto Scaling policies
- CloudWatch Container Insights

**Task Definitions:**

1. **Application Tasks**
   - Image: Multi-stage optimized Docker image
   - CPU/Memory:
     - Dev: 512 CPU / 1024 MB
     - Staging: 1024 CPU / 2048 MB
     - Production: 2048 CPU / 4096 MB
   - Health checks: /health endpoint
   - Task count:
     - Dev: 1 task
     - Staging: 2 tasks
     - Production: 3-10 tasks (auto-scaling)

2. **Celery Worker Tasks**
   - CPU/Memory:
     - Dev: 256 CPU / 512 MB
     - Staging: 512 CPU / 1024 MB
     - Production: 1024 CPU / 2048 MB
   - Task count:
     - Dev: 1 worker
     - Staging: 2 workers
     - Production: 4 workers

3. **Celery Beat Task**
   - Single task for scheduled jobs
   - CPU: 256-512, Memory: 512-1024 MB

**Auto-Scaling:**
- Target tracking based on:
  - CPU utilization (70% target)
  - Memory utilization (75% target)
  - ALB request count per target
- Scale-out: Fast (60s evaluation)
- Scale-in: Gradual (300s cooldown)

**Load Balancer:**
- Application Load Balancer (ALB)
- HTTPS/HTTP listeners
- Health checks on /health endpoint
- Sticky sessions enabled
- Connection draining (300s)

### 6. Storage Layer (S3)

**Buckets:**

1. **Videos Bucket**
   - Purpose: Processed video storage
   - Versioning: Enabled (staging, prod)
   - Lifecycle: Glacier after 90 days, delete after 730 days
   - Encryption: AES-256
   - CORS: Enabled for uploads

2. **Uploads Bucket**
   - Purpose: Temporary upload storage
   - Lifecycle: Delete after 7 days
   - Encryption: AES-256
   - Public access: Blocked

3. **Static Assets Bucket**
   - Purpose: CDN origin for static files
   - Versioning: Enabled
   - Encryption: AES-256
   - CloudFront OAI access only

**Security:**
- All buckets: Block public access
- Bucket policies for service access
- Encryption at rest (AES-256)
- Versioning for data protection
- Lifecycle policies for cost optimization

### 7. CDN Layer (CloudFront)

**Configuration:**
- Global edge locations
- Price classes:
  - Dev: US, Canada, Europe (PriceClass_100)
  - Staging: US, Canada, Europe, Asia (PriceClass_200)
  - Production: Global (PriceClass_All)

**Origins:**
1. S3 Static Assets (OAI access)
2. S3 Videos (OAI access)
3. ALB (API traffic)

**Cache Behaviors:**
- Static assets: Long TTL (24 hours)
- Videos: Medium TTL (1 hour)
- API: No caching (dynamic content)

**Features:**
- HTTPS only (SSL/TLS)
- Custom domain support
- Gzip compression
- Query string forwarding
- Origin failover

### 8. Secrets Management (AWS Secrets Manager)

**Secrets Stored:**
- Database credentials (master password)
- Redis password (if auth enabled)
- Application secrets (SECRET_KEY, JWT_SECRET)
- Third-party API keys (Stripe, AWS, SMTP)
- Service account credentials

**Features:**
- Automatic rotation (database passwords)
- Encryption at rest (KMS)
- IAM-based access control
- Audit logging
- Version management

### 9. Monitoring & Observability (CloudWatch)

**Log Groups:**
- Application logs (ECS container logs)
- Database logs (PostgreSQL, slow queries)
- Redis logs (slow log, engine log)
- VPC Flow Logs (network traffic)
- ALB access logs

**Metrics:**
- ECS: CPU, Memory, Task count
- RDS: CPU, Connections, IOPS, Latency
- Redis: CPU, Memory, Connections, Hit rate
- ALB: Request count, Target health, Latency

**Alarms:**
- High CPU utilization (>75%)
- High memory utilization (>80%)
- Database connection exhaustion
- Redis memory pressure
- ALB unhealthy targets
- ECS task failures
- Application errors (>100/min)

**Dashboards:**
- Infrastructure overview
- Application performance
- Database metrics
- Cache metrics
- Cost analysis

## Infrastructure as Code Structure

```
deployment/terraform/
├── main.tf                    # Main orchestration
├── variables.tf               # Variable definitions
├── outputs.tf                 # Output definitions
├── environments/
│   ├── dev.tfvars            # Development variables
│   ├── staging.tfvars        # Staging variables
│   └── prod.tfvars           # Production variables
└── modules/
    ├── vpc/                   # Network infrastructure
    │   ├── main.tf
    │   ├── variables.tf
    │   └── outputs.tf
    ├── security_groups/       # Security groups
    │   ├── main.tf
    │   ├── variables.tf
    │   └── outputs.tf
    ├── rds/                   # PostgreSQL database
    │   ├── main.tf
    │   ├── variables.tf
    │   └── outputs.tf
    ├── elasticache/           # Redis cache
    │   ├── main.tf
    │   ├── variables.tf
    │   └── outputs.tf
    ├── ecs/                   # Container orchestration
    │   ├── main.tf
    │   ├── variables.tf
    │   └── outputs.tf
    ├── s3/                    # Object storage
    │   ├── main.tf
    │   ├── variables.tf
    │   └── outputs.tf
    ├── cloudfront/            # CDN
    │   ├── main.tf
    │   ├── variables.tf
    │   └── outputs.tf
    ├── secrets/               # Secrets Manager
    │   ├── main.tf
    │   ├── variables.tf
    │   └── outputs.tf
    └── monitoring/            # CloudWatch
        ├── main.tf
        ├── variables.tf
        └── outputs.tf
```

## Cost Optimization Strategies

### Development Environment
- Single NAT Gateway ($32/month)
- t3.small instances
- Single AZ deployment
- No detailed monitoring
- Short backup retention
- No versioning

**Estimated Monthly Cost: $200-300**

### Staging Environment
- Multi-AZ with NAT per AZ ($96/month)
- t3.medium instances
- Multi-AZ RDS and Redis
- 7-day backups
- Versioning enabled

**Estimated Monthly Cost: $600-800**

### Production Environment
- Multi-AZ with NAT per AZ ($96/month)
- r6g (ARM-based) instances for better price/performance
- Multi-AZ RDS and Redis with replicas
- 30-day backups
- Full monitoring and logging
- Global CloudFront distribution

**Estimated Monthly Cost: $1,500-2,500**

### Cost Reduction Tips
1. Use ARM-based instances (r6g, t4g) - 20% cheaper
2. Reserved Instances for RDS (40% savings)
3. Savings Plans for ECS (20% savings)
4. S3 Intelligent-Tiering for automatic cost optimization
5. CloudWatch log retention policies (7-30 days)
6. Single NAT Gateway for non-prod (save $64/month per env)

## High Availability Design

### Availability Targets
- **Dev**: 95% (single AZ, minimal redundancy)
- **Staging**: 99% (multi-AZ, replication)
- **Production**: 99.9% (multi-AZ, auto-scaling, failover)

### HA Components

1. **Multi-AZ Deployment**
   - Resources distributed across 3 AZs
   - RDS automatic failover (<2min)
   - Redis automatic failover (<1min)
   - ECS tasks distributed across AZs

2. **Auto-Scaling**
   - ECS tasks scale 3-10 based on demand
   - Celery workers scale based on queue depth
   - Database storage auto-scaling

3. **Load Balancing**
   - ALB health checks every 30s
   - Automatic unhealthy target removal
   - Connection draining for graceful shutdown

4. **Backup & Recovery**
   - Automated daily RDS snapshots
   - Point-in-time recovery (30 days)
   - Redis snapshots (5-7 days)
   - S3 versioning for data protection

5. **Disaster Recovery**
   - RDS snapshots to different region
   - S3 cross-region replication
   - Infrastructure as Code for rapid rebuilding
   - Documented runbooks

## Security Best Practices

### Network Security
- Private subnets for all workloads
- No direct internet access
- NAT Gateway for outbound only
- Security group whitelisting
- VPC Flow Logs enabled

### Data Security
- Encryption at rest (KMS)
- Encryption in transit (TLS)
- Secrets Manager for credentials
- No hardcoded secrets
- Database parameter group security settings

### Access Control
- IAM roles for service authentication
- Principle of least privilege
- No long-term credentials
- MFA for console access
- CloudTrail audit logging

### Compliance
- GDPR data residency
- PCI-DSS for payment data (Stripe)
- HIPAA considerations (if applicable)
- SOC 2 controls
- Regular security scanning

## Maintenance & Operations

### Regular Maintenance
- Weekly: Review CloudWatch alarms
- Weekly: Check auto-scaling metrics
- Monthly: Database performance review
- Monthly: Cost optimization review
- Quarterly: Security audit
- Quarterly: Disaster recovery testing

### Patching Strategy
- RDS: Automatic minor version upgrades
- Redis: Automatic minor version upgrades
- ECS: Rolling updates with new task definitions
- OS: Managed by AWS (Fargate)

### Backup Verification
- Weekly: Test RDS snapshot restoration
- Monthly: Test cross-region failover
- Quarterly: Full disaster recovery drill

## Scaling Guidelines

### Horizontal Scaling
- ECS tasks: Auto-scales based on CPU/memory
- Celery workers: Scale based on queue depth
- RDS: Read replicas for read-heavy workloads
- Redis: Add more replicas for read scaling

### Vertical Scaling
- ECS tasks: Increase CPU/memory allocation
- RDS: Upgrade instance class (minimal downtime)
- Redis: Upgrade node type (minimal downtime)

### Database Scaling
- Connection pooling (SQLAlchemy)
- Query optimization (indexes, EXPLAIN)
- Read replicas for reporting queries
- Partitioning for large tables
- Caching frequently accessed data

## Monitoring & Alerting

### Critical Alerts (PagerDuty)
- All ECS tasks stopped
- Database CPU >90%
- Redis memory >95%
- ALB all targets unhealthy
- Application error rate >500/min

### Warning Alerts (Slack)
- ECS CPU >75%
- Database connections >80%
- Redis memory >80%
- ALB latency >2s
- Error rate >100/min

### Informational (Email)
- Deployment completed
- Auto-scaling events
- Backup completed
- Cost anomalies

## Troubleshooting Guide

### Common Issues

**Issue: High database CPU**
- Check slow query log
- Review query execution plans
- Add missing indexes
- Optimize N+1 queries
- Consider read replicas

**Issue: High memory usage (ECS)**
- Review application memory leaks
- Check Celery worker memory
- Adjust task memory limits
- Scale horizontally

**Issue: Redis connection errors**
- Check security group rules
- Verify connection string
- Check maxclients parameter
- Monitor connection pool

**Issue: Deployment failures**
- Check task definition syntax
- Verify IAM role permissions
- Check ECR image availability
- Review CloudWatch logs

## Next Steps

1. **Initial Deployment**: Follow DEPLOYMENT.md
2. **CI/CD Setup**: Configure GitHub Actions
3. **Monitoring Setup**: Configure alarms and dashboards
4. **Security Review**: Complete security audit
5. **Load Testing**: Validate auto-scaling
6. **Documentation**: Update runbooks

## Support & Resources

- **Terraform Docs**: https://registry.terraform.io/providers/hashicorp/aws
- **AWS Well-Architected**: https://aws.amazon.com/architecture/well-architected/
- **Infrastructure Code**: `deployment/terraform/`
- **Runbooks**: `docs/operations/`
- **Team**: devops-team@socialflow.com
