variable "aws_region" {
  description = "AWS region"
  type        = string
  default     = "us-east-1"
}

variable "s3_bucket" {
  description = "S3 bucket name for video uploads"
  type        = string
  default     = "socialflow-uploads-dev"
}
