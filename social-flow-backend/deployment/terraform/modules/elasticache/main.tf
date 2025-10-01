# ElastiCache Redis Module
# Creates managed Redis cluster with replication and automatic failover

terraform {
  required_version = ">= 1.5.0"
}

# Variables
variable "cluster_id" {
  description = "Redis cluster identifier"
  type        = string
}

variable "engine_version" {
  description = "Redis engine version"
  type        = string
  default     = "7.0"
}

variable "node_type" {
  description = "Node instance type"
  type        = string
  default     = "cache.t3.medium"
}

variable "num_cache_nodes" {
  description = "Number of cache nodes"
  type        = number
  default     = 2
}

variable "parameter_group_family" {
  description = "Parameter group family"
  type        = string
  default     = "redis7"
}

variable "port" {
  description = "Redis port"
  type        = number
  default     = 6379
}

variable "subnet_ids" {
  description = "List of subnet IDs for cache subnet group"
  type        = list(string)
}

variable "security_group_ids" {
  description = "List of security group IDs"
  type        = list(string)
}

variable "automatic_failover_enabled" {
  description = "Enable automatic failover"
  type        = bool
  default     = true
}

variable "multi_az_enabled" {
  description = "Enable multi-AZ"
  type        = bool
  default     = true
}

variable "snapshot_retention_limit" {
  description = "Number of days to retain snapshots"
  type        = number
  default     = 5
}

variable "snapshot_window" {
  description = "Daily snapshot window"
  type        = string
  default     = "03:00-05:00"
}

variable "maintenance_window" {
  description = "Maintenance window"
  type        = string
  default     = "sun:05:00-sun:07:00"
}

variable "environment" {
  description = "Environment name"
  type        = string
}

variable "project_name" {
  description = "Project name"
  type        = string
}

variable "at_rest_encryption_enabled" {
  description = "Enable encryption at rest"
  type        = bool
  default     = true
}

variable "transit_encryption_enabled" {
  description = "Enable encryption in transit"
  type        = bool
  default     = true
}

# Cache Subnet Group
resource "aws_elasticache_subnet_group" "main" {
  name        = "${var.project_name}-${var.environment}-cache-subnet-group"
  description = "ElastiCache subnet group for ${var.project_name}"
  subnet_ids  = var.subnet_ids

  tags = {
    Name        = "${var.project_name}-${var.environment}-cache-subnet-group"
    Environment = var.environment
  }
}

# Cache Parameter Group
resource "aws_elasticache_parameter_group" "main" {
  name        = "${var.project_name}-${var.environment}-redis-params"
  family      = var.parameter_group_family
  description = "Custom parameter group for ${var.project_name} Redis"

  # Performance and memory optimization
  parameter {
    name  = "maxmemory-policy"
    value = "allkeys-lru"
  }

  parameter {
    name  = "timeout"
    value = "300"
  }

  parameter {
    name  = "tcp-keepalive"
    value = "300"
  }

  parameter {
    name  = "maxmemory-samples"
    value = "5"
  }

  tags = {
    Name        = "${var.project_name}-${var.environment}-redis-params"
    Environment = var.environment
  }

  lifecycle {
    create_before_destroy = true
  }
}

# ElastiCache Replication Group
resource "aws_elasticache_replication_group" "main" {
  replication_group_id = var.cluster_id
  description          = "Redis cluster for ${var.project_name}-${var.environment}"
  
  engine               = "redis"
  engine_version       = var.engine_version
  node_type            = var.node_type
  num_cache_clusters   = var.num_cache_nodes
  port                 = var.port
  
  # Parameter group
  parameter_group_name = aws_elasticache_parameter_group.main.name
  
  # Network configuration
  subnet_group_name  = aws_elasticache_subnet_group.main.name
  security_group_ids = var.security_group_ids
  
  # High availability
  automatic_failover_enabled = var.automatic_failover_enabled
  multi_az_enabled           = var.multi_az_enabled
  
  # Backup configuration
  snapshot_retention_limit = var.snapshot_retention_limit
  snapshot_window          = var.snapshot_window
  
  # Maintenance
  maintenance_window       = var.maintenance_window
  auto_minor_version_upgrade = true
  
  # Encryption
  at_rest_encryption_enabled = var.at_rest_encryption_enabled
  transit_encryption_enabled = var.transit_encryption_enabled
  
  # Notifications
  notification_topic_arn = aws_sns_topic.redis_notifications.arn
  
  # Logging
  log_delivery_configuration {
    destination      = aws_cloudwatch_log_group.redis_slow_log.name
    destination_type = "cloudwatch-logs"
    log_format       = "json"
    log_type         = "slow-log"
  }
  
  log_delivery_configuration {
    destination      = aws_cloudwatch_log_group.redis_engine_log.name
    destination_type = "cloudwatch-logs"
    log_format       = "json"
    log_type         = "engine-log"
  }
  
  tags = {
    Name        = var.cluster_id
    Environment = var.environment
  }
  
  depends_on = [
    aws_elasticache_subnet_group.main,
    aws_elasticache_parameter_group.main
  ]
}

# CloudWatch Log Groups
resource "aws_cloudwatch_log_group" "redis_slow_log" {
  name_prefix       = "/aws/elasticache/${var.project_name}-${var.environment}/slow-log-"
  retention_in_days = 7
  
  tags = {
    Name        = "${var.project_name}-${var.environment}-redis-slow-log"
    Environment = var.environment
  }
}

resource "aws_cloudwatch_log_group" "redis_engine_log" {
  name_prefix       = "/aws/elasticache/${var.project_name}-${var.environment}/engine-log-"
  retention_in_days = 7
  
  tags = {
    Name        = "${var.project_name}-${var.environment}-redis-engine-log"
    Environment = var.environment
  }
}

# SNS Topic for Notifications
resource "aws_sns_topic" "redis_notifications" {
  name_prefix = "${var.project_name}-${var.environment}-redis-"
  
  tags = {
    Name        = "${var.project_name}-${var.environment}-redis-notifications"
    Environment = var.environment
  }
}

# CloudWatch Alarms
resource "aws_cloudwatch_metric_alarm" "cache_cpu" {
  alarm_name          = "${var.cluster_id}-cpu-utilization"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 2
  metric_name         = "CPUUtilization"
  namespace           = "AWS/ElastiCache"
  period              = 300
  statistic           = "Average"
  threshold           = 75
  alarm_description   = "This metric monitors Redis CPU utilization"
  alarm_actions       = [aws_sns_topic.redis_notifications.arn]
  
  dimensions = {
    ReplicationGroupId = aws_elasticache_replication_group.main.id
  }
  
  tags = {
    Name        = "${var.cluster_id}-cpu-alarm"
    Environment = var.environment
  }
}

resource "aws_cloudwatch_metric_alarm" "cache_memory" {
  alarm_name          = "${var.cluster_id}-memory-utilization"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 2
  metric_name         = "DatabaseMemoryUsagePercentage"
  namespace           = "AWS/ElastiCache"
  period              = 300
  statistic           = "Average"
  threshold           = 80
  alarm_description   = "This metric monitors Redis memory utilization"
  alarm_actions       = [aws_sns_topic.redis_notifications.arn]
  
  dimensions = {
    ReplicationGroupId = aws_elasticache_replication_group.main.id
  }
  
  tags = {
    Name        = "${var.cluster_id}-memory-alarm"
    Environment = var.environment
  }
}

# Outputs
output "redis_endpoint" {
  description = "Redis primary endpoint"
  value       = aws_elasticache_replication_group.main.primary_endpoint_address
}

output "redis_reader_endpoint" {
  description = "Redis reader endpoint"
  value       = aws_elasticache_replication_group.main.reader_endpoint_address
}

output "redis_port" {
  description = "Redis port"
  value       = aws_elasticache_replication_group.main.port
}

output "redis_id" {
  description = "Redis replication group ID"
  value       = aws_elasticache_replication_group.main.id
}

output "redis_arn" {
  description = "Redis replication group ARN"
  value       = aws_elasticache_replication_group.main.arn
}
