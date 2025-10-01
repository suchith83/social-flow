# Environment: Development
# Small instances for cost optimization

environment    = "dev"
project_name   = "social-flow"
aws_region     = "us-east-1"

# VPC Configuration
vpc_cidr              = "10.0.0.0/16"
availability_zones    = ["us-east-1a", "us-east-1b", "us-east-1c"]
enable_nat_gateway    = true
single_nat_gateway    = true  # Cost optimization for dev
enable_vpn_gateway    = false
enable_flow_logs      = false  # Cost optimization

# RDS Configuration
db_instance_class          = "db.t3.small"
db_allocated_storage       = 20
db_max_allocated_storage   = 50
db_multi_az                = false  # Single AZ for dev
db_backup_retention_period = 3
db_enable_performance_insights = false

# ElastiCache Configuration
redis_node_type                = "cache.t3.micro"
redis_num_cache_nodes          = 1  # No replication for dev
redis_automatic_failover       = false
redis_multi_az                 = false
redis_snapshot_retention_limit = 1

# ECS Configuration
ecs_app_count               = 1  # Minimum tasks
ecs_app_cpu                 = 512
ecs_app_memory              = 1024
ecs_app_min_capacity        = 1
ecs_app_max_capacity        = 2  # Limited scaling for dev
ecs_celery_worker_count     = 1
ecs_celery_worker_cpu       = 256
ecs_celery_worker_memory    = 512
ecs_celery_beat_cpu         = 256
ecs_celery_beat_memory      = 512

# S3 Configuration
s3_enable_versioning        = false
s3_lifecycle_glacier_days   = 90
s3_lifecycle_expiration_days = 180

# CloudFront Configuration
cloudfront_price_class      = "PriceClass_100"  # US, Canada, Europe
cloudfront_min_ttl          = 0
cloudfront_default_ttl      = 3600
cloudfront_max_ttl          = 86400

# Monitoring
enable_detailed_monitoring  = false
alarm_email                 = "dev-alerts@socialflow.com"

# Tags
tags = {
  Environment = "development"
  ManagedBy   = "terraform"
  Project     = "social-flow"
  Owner       = "devops-team"
  CostCenter  = "engineering"
}
