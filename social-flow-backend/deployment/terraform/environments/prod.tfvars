# Environment: Production
# High availability and performance configuration

environment    = "production"
project_name   = "social-flow"
aws_region     = "us-east-1"

# VPC Configuration
vpc_cidr              = "10.2.0.0/16"
availability_zones    = ["us-east-1a", "us-east-1b", "us-east-1c"]
enable_nat_gateway    = true
single_nat_gateway    = false  # One NAT per AZ for HA
enable_vpn_gateway    = false
enable_flow_logs      = true

# RDS Configuration
db_instance_class          = "db.r6g.xlarge"  # Memory-optimized
db_allocated_storage       = 100
db_max_allocated_storage   = 500
db_multi_az                = true
db_backup_retention_period = 30  # Extended retention
db_enable_performance_insights = true

# ElastiCache Configuration
redis_node_type                = "cache.r6g.large"  # Memory-optimized
redis_num_cache_nodes          = 3  # Primary + 2 replicas
redis_automatic_failover       = true
redis_multi_az                 = true
redis_snapshot_retention_limit = 7

# ECS Configuration
ecs_app_count               = 3  # Minimum for HA
ecs_app_cpu                 = 2048
ecs_app_memory              = 4096
ecs_app_min_capacity        = 3
ecs_app_max_capacity        = 10  # Scale up to 10 tasks
ecs_celery_worker_count     = 4  # Multiple workers
ecs_celery_worker_cpu       = 1024
ecs_celery_worker_memory    = 2048
ecs_celery_beat_cpu         = 512
ecs_celery_beat_memory      = 1024

# S3 Configuration
s3_enable_versioning        = true
s3_lifecycle_glacier_days   = 90
s3_lifecycle_expiration_days = 730  # 2 years retention

# CloudFront Configuration
cloudfront_price_class      = "PriceClass_All"  # Global distribution
cloudfront_min_ttl          = 0
cloudfront_default_ttl      = 86400  # 24 hours
cloudfront_max_ttl          = 31536000  # 1 year

# Monitoring
enable_detailed_monitoring  = true
alarm_email                 = "prod-alerts@socialflow.com"

# Tags
tags = {
  Environment = "production"
  ManagedBy   = "terraform"
  Project     = "social-flow"
  Owner       = "devops-team"
  CostCenter  = "engineering"
  Criticality = "high"
  Compliance  = "required"
}
