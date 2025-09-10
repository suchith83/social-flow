# Deployment Guide
Social Flow Deployment Guide
Author: Backend TeamDate: September 11, 2025
Overview
This guide provides detailed instructions for deploying the Social Flow backend to AWS using a combination of Docker, AWS ECS, Kubernetes, and Terraform. The deployment process is automated via CI/CD pipelines, supporting blue-green deployments, canary releases, and rolling updates. The guide covers prerequisites, infrastructure provisioning, service deployment, and post-deployment validation.
Prerequisites

AWS Account: Configured with sufficient permissions for ECS, S3, RDS, and other services.
Terraform: Version 1.5+ for infrastructure provisioning.
Docker: For building and testing container images.
kubectl: For Kubernetes-based deployments (optional).
AWS CLI: Configured with credentials.
Git: For cloning the repository and managing CI/CD.
CI/CD Platform: GitLab CI, GitHub Actions, or Jenkins for pipeline automation.
Monitoring Tools: Prometheus, Grafana, and AWS CloudWatch for post-deployment monitoring.

Infrastructure Setup

Clone the Repository:
git clone https://github.com/social-flow/social-flow-backend.git
cd social-flow-backend


Configure AWS Credentials:

Set up AWS credentials in ~/.aws/credentials or environment variables.
Example:export AWS_ACCESS_KEY_ID=<your-key>
export AWS_SECRET_ACCESS_KEY=<your-secret>
export AWS_DEFAULT_REGION=us-west-2




Provision Infrastructure with Terraform:

Navigate to automation/infrastructure/provisioning.
Initialize Terraform:terraform init


Apply the configuration:terraform apply -var-file=production.tfvars


This sets up VPC, ECS clusters, RDS (CockroachDB), S3 buckets, CloudFront, and other resources.


Configure Databases:

Run migrations for CockroachDB and MongoDB:./scripts/setup/db_migrate.sh


Seed development data (optional):./scripts/setup/db_seed.sh





Service Deployment

Build Docker Images:

Build images for all services:./scripts/deployment/build.sh


This builds images for user-service, video-service, recommendation-service, etc.


Push Images to AWS ECR:

Configure ECR repository:aws ecr get-login-password --region us-west-2 | docker login --username AWS --password-stdin <account-id>.dkr.ecr.us-west-2.amazonaws.com


Push images:./scripts/deployment/push.sh




Deploy to AWS ECS:

Update ECS service definitions in deployment/environments/production.
Deploy services:./scripts/deployment/deploy_ecs.sh


Alternatively, deploy to Kubernetes:./scripts/deployment/deploy_k8s.sh




Configure API Gateway:

Update Kong configuration in services/api-gateway/config/kong.yml.
Deploy the gateway:./scripts/deployment/deploy_gateway.sh





CI/CD Pipelines
The repository includes pre-configured CI/CD pipelines in cicd/gitlab-ci, cicd/github-actions, and cicd/argocd. To set up:

GitLab CI:

Configure .gitlab-ci.yml in cicd/gitlab-ci/pipelines.
Set up GitLab runners with Docker and AWS CLI.
Trigger pipelines on code push.


GitHub Actions:

Configure workflows in cicd/github-actions/.github/workflows.
Add AWS credentials as GitHub secrets.
Enable Actions in the repository settings.


ArgoCD:

Deploy ArgoCD to the Kubernetes cluster:./scripts/deployment/setup_argocd.sh


Configure applications in cicd/argocd/applications.



Post-Deployment Validation

Verify Services:

Check ECS service status:aws ecs describe-services --cluster social-flow --services user-service video-service


Test API endpoints:curl http://<api-gateway-url>/api/v1/health




Monitor Metrics:

Access Grafana dashboards at monitoring/dashboards.
Check CloudWatch logs for errors:aws logs tail /social-flow/user-service




Run Tests:

Execute integration tests:./scripts/testing/run_integration.sh


Perform load testing with k6:k6 run tools/load-testing/k6/script.js





Rollback Procedures

Blue-Green Deployment:
Switch traffic to the previous deployment:./scripts/deployment/rollback_blue_green.sh




Canary Release:
Revert canary changes:./scripts/deployment/rollback_canary.sh





Maintenance

Database Backups:
Configure automated backups in data/backups/database.
Run manual backups:./scripts/maintenance/backup_db.sh




Log Rotation:
Configure log retention in monitoring/logging/retention.


Security Scans:
Run vulnerability scans:./tools/security/vulnerability-scanning/scan.sh





Troubleshooting

Service Failures: Check ECS task logs in CloudWatch.
API Errors: Verify Kong logs in services/api-gateway/logs.
Performance Issues: Monitor metrics in Prometheus and optimize caching in performance/caching.

Contact
For deployment issues, contact the backend team at backend@socialflow.com or open an issue in the repository.