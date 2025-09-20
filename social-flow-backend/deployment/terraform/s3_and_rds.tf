resource "aws_s3_bucket" "uploads" {
  bucket = var.s3_bucket
  acl    = "private"

  versioning {
    enabled = true
  }

  lifecycle_rule {
    id      = "cleanup-old-objects"
    enabled = true
    expiration {
      days = 365
    }
  }
}

# RDS instance (placeholder) — for production replace with subnet group, security groups, parameter groups.
resource "aws_db_instance" "postgres" {
  allocated_storage    = 20
  engine               = "postgres"
  engine_version       = "15"
  instance_class       = "db.t3.micro"
  name                 = "socialflow"
  username             = "socialflow"
  password             = "change-me-in-prod"
  skip_final_snapshot  = true
  publicly_accessible  = false
  storage_encrypted    = true
}
