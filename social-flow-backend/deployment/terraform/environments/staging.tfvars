# Environment: Staging
# Production-like configuration for testing

environment    = "staging"
project_name   = "social-flow"
aws_region     = "us-east-1"

# VPC Configuration
vpc_cidr              = "10.1.0.0/16"
availability_zones    = ["us-east-1a", "us-east-1b", "us-east-1c"]
enable_nat_gateway    = true
single_nat_gateway    = false  # One NAT per AZ
enable_vpn_gateway    = false
enable_flow_logs      = true

# RDS Configuration
db_instance_class          = "db.t3.medium"
db_allocated_storage       = 50
db_max_allocated_storage   = 200
db_multi_az                = true  # Multi-AZ for staging
db_backup_retention_period = 7
db_enable_performance_insights = true

# ElastiCache Configuration
redis_node_type                = "cache.t3.small"
redis_num_cache_nodes          = 2  # Primary + replica
redis_automatic_failover       = true
redis_multi_az                 = true
redis_snapshot_retention_limit = 3

# ECS Configuration
ecs_app_count               = 2
ecs_app_cpu                 = 1024
ecs_app_memory              = 2048
ecs_app_min_capacity        = 2
ecs_app_max_capacity        = 5
ecs_celery_worker_count     = 2
ecs_celery_worker_cpu       = 512
ecs_celery_worker_memory    = 1024
ecs_celery_beat_cpu         = 256
ecs_celery_beat_memory      = 512

# S3 Configuration
s3_enable_versioning        = true
s3_lifecycle_glacier_days   = 60
s3_lifecycle_expiration_days = 365

# CloudFront Configuration
cloudfront_price_class      = "PriceClass_200"  # US, Canada, Europe, Asia
cloudfront_min_ttl          = 0
cloudfront_default_ttl      = 3600
cloudfront_max_ttl          = 86400

# Monitoring
enable_detailed_monitoring  = true
alarm_email                 = "staging-alerts@socialflow.com"

# Tags
tags = {
  Environment = "staging"
  ManagedBy   = "terraform"
  Project     = "social-flow"
  Owner       = "devops-team"
  CostCenter  = "engineering"
}
