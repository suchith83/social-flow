# Makefile for Social Flow backend development

.PHONY: help install install-dev test test-unit test-integration test-performance test-security test-all lint format clean run dev docker-build docker-run docker-stop docs

# Default target
help:
	@echo "Social Flow Backend - Development Commands"
	@echo "=========================================="
	@echo ""
	@echo "Installation:"
	@echo "  install      Install production dependencies"
	@echo "  install-dev  Install development dependencies"
	@echo ""
	@echo "Testing:"
	@echo "  test         Run all tests"
	@echo "  test-unit    Run unit tests only"
	@echo "  test-integration Run integration tests only"
	@echo "  test-performance Run performance tests only"
	@echo "  test-security Run security tests only"
	@echo "  test-coverage Run tests with coverage report"
	@echo ""
	@echo "Code Quality:"
	@echo "  lint         Run all linting checks"
	@echo "  format       Format code with black and isort"
	@echo "  type-check   Run type checking with mypy"
	@echo "  security     Run security checks"
	@echo ""
	@echo "Development:"
	@echo "  run          Run the application"
	@echo "  dev          Run in development mode with hot reload"
	@echo "  clean        Clean up temporary files"
	@echo ""
	@echo "Docker:"
	@echo "  docker-build Build Docker image"
	@echo "  docker-run   Run Docker container"
	@echo "  docker-stop  Stop Docker container"
	@echo ""
	@echo "Documentation:"
	@echo "  docs         Generate documentation"
	@echo "  docs-serve   Serve documentation locally"

# Installation
install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements.txt
	pip install -r requirements-dev.txt

# Testing
test: test-all

test-unit:
	python -m pytest tests/unit/ -v

test-integration:
	python -m pytest tests/integration/ -v

test-performance:
	python -m pytest tests/performance/ -v

test-security:
	python -m pytest tests/security/ -v

test-all:
	python -m pytest tests/ -v

test-coverage:
	python -m pytest tests/ --cov=app --cov-report=html --cov-report=term

# Code Quality
lint:
	python -m flake8 app/
	python -m mypy app/
	python -m bandit -r app/

format:
	python -m black app/
	python -m isort app/

type-check:
	python -m mypy app/

security:
	python -m bandit -r app/
	python -m safety check

# Development
run:
	python -m uvicorn app.main:app --host 0.0.0.0 --port 8000

dev:
	python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .bandit/

# Docker
docker-build:
	docker build -t social-flow-backend .

docker-run:
	docker-compose up -d

docker-stop:
	docker-compose down

# Documentation
docs:
	cd docs && make html

docs-serve:
	cd docs && python -m http.server 8001

# Database
db-migrate:
	alembic upgrade head

db-downgrade:
	alembic downgrade -1

db-revision:
	alembic revision --autogenerate -m "$(message)"

# Pre-commit hooks
pre-commit-install:
	pre-commit install

pre-commit-run:
	pre-commit run --all-files

# CI/CD
ci-test:
	python -m pytest tests/ --cov=app --cov-report=xml --junitxml=test-results.xml

ci-lint:
	python -m flake8 app/
	python -m mypy app/
	python -m bandit -r app/

ci-format:
	python -m black --check app/
	python -m isort --check-only app/

# Performance
benchmark:
	python -m pytest tests/performance/ --benchmark-only

load-test:
	locust -f tests/load/locustfile.py --host=http://localhost:8000

# Monitoring
logs:
	docker-compose logs -f

monitor:
	docker stats

# Deployment
deploy-staging:
	@echo "Deploying to staging..."
	# Add staging deployment commands here

deploy-production:
	@echo "Deploying to production..."
	# Add production deployment commands here

# Health checks
health:
	curl -f http://localhost:8000/health || exit 1

# Database setup
db-setup:
	createdb social_flow_dev
	alembic upgrade head

# Seed data
seed:
	python scripts/seed_data.py

# Backup
backup:
	pg_dump social_flow_dev > backup_$(shell date +%Y%m%d_%H%M%S).sql

# Restore
restore:
	psql social_flow_dev < $(file)

# Environment setup
env-setup:
	cp .env.example .env
	@echo "Please edit .env file with your configuration"

# Full setup for new developers
setup: env-setup install-dev db-setup pre-commit-install
	@echo "Setup complete! Run 'make dev' to start the development server."
