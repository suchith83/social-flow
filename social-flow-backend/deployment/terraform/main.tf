# Main Terraform configuration for Social Flow infrastructure
terraform {
  required_version = ">= 1.5.0"
  
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
  
  backend "s3" {
    bucket         = "social-flow-terraform-state"
    key            = "prod/terraform.tfstate"
    region         = "us-east-1"
    encrypt        = true
    dynamodb_table = "terraform-lock"
  }
}

provider "aws" {
  region = var.aws_region
  
  default_tags {
    tags = {
      Project     = "SocialFlow"
      Environment = var.environment
      ManagedBy   = "Terraform"
      Owner       = "DevOps"
    }
  }
}

# Data sources
data "aws_availability_zones" "available" {
  state = "available"
}

data "aws_caller_identity" "current" {}

# VPC Module
module "vpc" {
  source = "./modules/vpc"
  
  environment         = var.environment
  vpc_cidr            = var.vpc_cidr
  availability_zones  = slice(data.aws_availability_zones.available.names, 0, 3)
  private_subnet_cidrs = var.private_subnet_cidrs
  public_subnet_cidrs  = var.public_subnet_cidrs
  
  enable_nat_gateway   = true
  single_nat_gateway   = var.environment == "dev" ? true : false
  enable_dns_hostnames = true
  enable_dns_support   = true
  
  tags = var.tags
}

# Security Groups Module
module "security_groups" {
  source = "./modules/security_groups"
  
  environment = var.environment
  vpc_id      = module.vpc.vpc_id
  vpc_cidr    = var.vpc_cidr
  
  tags = var.tags
}

# RDS PostgreSQL Database
module "database" {
  source = "./modules/rds"
  
  environment             = var.environment
  vpc_id                  = module.vpc.vpc_id
  database_subnet_ids     = module.vpc.private_subnet_ids
  security_group_ids      = [module.security_groups.database_sg_id]
  
  instance_class          = var.db_instance_class
  allocated_storage       = var.db_allocated_storage
  max_allocated_storage   = var.db_max_allocated_storage
  engine_version          = "15.4"
  database_name           = var.db_name
  master_username         = var.db_username
  
  backup_retention_period = var.environment == "prod" ? 7 : 1
  backup_window           = "03:00-04:00"
  maintenance_window      = "sun:04:00-sun:05:00"
  
  multi_az                = var.environment == "prod" ? true : false
  deletion_protection     = var.environment == "prod" ? true : false
  skip_final_snapshot     = var.environment == "dev" ? true : false
  
  enabled_cloudwatch_logs_exports = ["postgresql", "upgrade"]
  performance_insights_enabled    = true
  
  tags = var.tags
}

# ElastiCache Redis
module "redis" {
  source = "./modules/elasticache"
  
  environment         = var.environment
  vpc_id              = module.vpc.vpc_id
  subnet_ids          = module.vpc.private_subnet_ids
  security_group_ids  = [module.security_groups.redis_sg_id]
  
  node_type           = var.redis_node_type
  num_cache_nodes     = var.environment == "prod" ? 2 : 1
  engine_version      = "7.0"
  parameter_family    = "redis7"
  
  automatic_failover_enabled = var.environment == "prod" ? true : false
  multi_az_enabled           = var.environment == "prod" ? true : false
  
  tags = var.tags
}

# ECS Cluster
module "ecs" {
  source = "./modules/ecs"
  
  environment         = var.environment
  vpc_id              = module.vpc.vpc_id
  private_subnet_ids  = module.vpc.private_subnet_ids
  public_subnet_ids   = module.vpc.public_subnet_ids
  
  app_security_group_id = module.security_groups.app_sg_id
  alb_security_group_id = module.security_groups.alb_sg_id
  
  # Container configuration
  app_image           = var.app_image
  app_port            = 8000
  cpu                 = var.ecs_cpu
  memory              = var.ecs_memory
  desired_count       = var.ecs_desired_count
  
  # Database connection
  database_url        = "postgresql+asyncpg://${var.db_username}:${module.database.db_password}@${module.database.db_endpoint}/${var.db_name}"
  redis_url           = "redis://${module.redis.primary_endpoint_address}:6379/0"
  
  # Environment variables
  environment_vars = {
    ENVIRONMENT = var.environment
    LOG_LEVEL   = var.environment == "prod" ? "INFO" : "DEBUG"
  }
  
  # Secrets from AWS Secrets Manager
  secrets_arns = [
    module.secrets.secret_arn
  ]
  
  # Auto-scaling
  autoscaling_enabled     = true
  autoscaling_min_capacity = var.autoscaling_min_capacity
  autoscaling_max_capacity = var.autoscaling_max_capacity
  
  tags = var.tags
}

# S3 Buckets
module "s3" {
  source = "./modules/s3"
  
  environment = var.environment
  
  # Video storage bucket
  video_bucket_name = "${var.project_name}-videos-${var.environment}"
  
  # User uploads bucket
  uploads_bucket_name = "${var.project_name}-uploads-${var.environment}"
  
  # Static assets bucket  
  static_bucket_name = "${var.project_name}-static-${var.environment}"
  
  enable_versioning = var.environment == "prod" ? true : false
  enable_encryption = true
  
  lifecycle_rules = {
    uploads = {
      transition_days = 90
      storage_class   = "INTELLIGENT_TIERING"
    }
    videos = {
      transition_days = 30
      storage_class   = "GLACIER"
    }
  }
  
  tags = var.tags
}

# CloudFront CDN
module "cloudfront" {
  source = "./modules/cloudfront"
  
  environment = var.environment
  
  video_bucket_name   = module.s3.video_bucket_name
  static_bucket_name  = module.s3.static_bucket_name
  alb_dns_name        = module.ecs.alb_dns_name
  
  ssl_certificate_arn = var.ssl_certificate_arn
  domain_name         = var.domain_name
  
  tags = var.tags
}

# Secrets Manager
module "secrets" {
  source = "./modules/secrets"
  
  environment = var.environment
  
  secrets = {
    secret_key          = var.secret_key
    jwt_secret          = var.jwt_secret
    stripe_secret_key   = var.stripe_secret_key
    aws_access_key      = var.aws_access_key
    aws_secret_key      = var.aws_secret_key
    smtp_password       = var.smtp_password
  }
  
  tags = var.tags
}

# CloudWatch Monitoring
module "monitoring" {
  source = "./modules/monitoring"
  
  environment     = var.environment
  ecs_cluster_name = module.ecs.cluster_name
  ecs_service_name = module.ecs.service_name
  alb_arn_suffix   = module.ecs.alb_arn_suffix
  target_group_arn_suffix = module.ecs.target_group_arn_suffix
  
  # Alarm notifications
  alarm_email = var.alarm_email
  
  # Thresholds
  cpu_threshold    = 80
  memory_threshold = 85
  
  tags = var.tags
}

# Outputs
output "alb_dns_name" {
  description = "DNS name of the Application Load Balancer"
  value       = module.ecs.alb_dns_name
}

output "cloudfront_domain_name" {
  description = "CloudFront distribution domain name"
  value       = module.cloudfront.cloudfront_domain_name
}

output "database_endpoint" {
  description = "RDS database endpoint"
  value       = module.database.db_endpoint
  sensitive   = true
}

output "redis_endpoint" {
  description = "ElastiCache Redis endpoint"
  value       = module.redis.primary_endpoint_address
  sensitive   = true
}

output "video_bucket_name" {
  description = "S3 bucket for video storage"
  value       = module.s3.video_bucket_name
}

output "ecs_cluster_name" {
  description = "ECS cluster name"
  value       = module.ecs.cluster_name
}
