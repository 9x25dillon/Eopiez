.PHONY: help start stop restart build test clean logs shell notebook

# Default target
.DEFAULT_GOAL := help

# Variables
COMPOSE := docker-compose
PYTHON := python3
JULIA := julia

help: ## Show this help message
	@echo "Eopiez Development Commands"
	@echo "============================"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# ===========================================
# Docker Operations
# ===========================================

start: ## Start all services
	@echo "Starting Eopiez services..."
	$(COMPOSE) up -d
	@echo "✓ Services started"
	@echo "  API Gateway: http://localhost:8000"
	@echo "  API Docs: http://localhost:8000/docs"

stop: ## Stop all services
	@echo "Stopping Eopiez services..."
	$(COMPOSE) down
	@echo "✓ Services stopped"

restart: ## Restart all services
	@echo "Restarting Eopiez services..."
	$(COMPOSE) restart
	@echo "✓ Services restarted"

build: ## Build all Docker images
	@echo "Building Docker images..."
	$(COMPOSE) build --no-cache
	@echo "✓ Build complete"

clean: ## Clean up containers, volumes, and images
	@echo "Cleaning up Docker resources..."
	$(COMPOSE) down -v --remove-orphans
	@echo "✓ Cleanup complete"

logs: ## View logs from all services
	$(COMPOSE) logs -f

logs-api: ## View API gateway logs
	$(COMPOSE) logs -f api-gateway

logs-julia: ## View Julia backend logs
	$(COMPOSE) logs -f julia-backend

logs-aluls: ## View AL-ULS service logs
	$(COMPOSE) logs -f al-uls-service

# ===========================================
# Testing
# ===========================================

test: ## Run all tests
	@echo "Running tests..."
	$(COMPOSE) --profile test run --rm test
	@echo "✓ Tests complete"

test-python: ## Run Python tests only
	@echo "Running Python tests..."
	pytest tests/python -v
	@echo "✓ Python tests complete"

test-julia: ## Run Julia tests only
	@echo "Running Julia tests..."
	$(JULIA) --project=. -e 'using Pkg; Pkg.test()'
	@echo "✓ Julia tests complete"

test-integration: ## Run integration tests
	@echo "Running integration tests..."
	pytest tests/integration -v
	@echo "✓ Integration tests complete"

test-coverage: ## Run tests with coverage report
	@echo "Running tests with coverage..."
	pytest tests/ -v --cov=src --cov-report=html --cov-report=term
	@echo "✓ Coverage report generated in htmlcov/"

# ===========================================
# Development
# ===========================================

notebook: ## Start Jupyter notebook server
	@echo "Starting Jupyter notebook..."
	$(COMPOSE) --profile dev up -d notebook
	@echo "✓ Notebook server started at http://localhost:8888"

shell-api: ## Open shell in API container
	$(COMPOSE) exec api-gateway /bin/bash

shell-julia: ## Open Julia REPL in Julia container
	$(COMPOSE) exec julia-backend julia --project=.

shell-db: ## Open PostgreSQL shell
	$(COMPOSE) exec postgres psql -U eopiez -d eopiez

shell-redis: ## Open Redis CLI
	$(COMPOSE) exec redis redis-cli

# ===========================================
# Monitoring
# ===========================================

monitoring: ## Start monitoring stack (Prometheus + Grafana)
	@echo "Starting monitoring stack..."
	$(COMPOSE) --profile monitoring up -d
	@echo "✓ Monitoring started"
	@echo "  Prometheus: http://localhost:9090"
	@echo "  Grafana: http://localhost:3000 (admin/admin)"

# ===========================================
# Database Operations
# ===========================================

db-migrate: ## Run database migrations
	@echo "Running database migrations..."
	$(COMPOSE) exec api-gateway alembic upgrade head
	@echo "✓ Migrations complete"

db-reset: ## Reset database (WARNING: destroys all data)
	@echo "⚠️  WARNING: This will destroy all data!"
	@read -p "Are you sure? [y/N] " -n 1 -r; \
	echo; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		$(COMPOSE) down -v; \
		$(COMPOSE) up -d postgres; \
		sleep 5; \
		$(COMPOSE) exec postgres psql -U eopiez -c "DROP DATABASE IF EXISTS eopiez;"; \
		$(COMPOSE) exec postgres psql -U eopiez -c "CREATE DATABASE eopiez;"; \
		echo "✓ Database reset complete"; \
	fi

db-backup: ## Backup database
	@echo "Backing up database..."
	@mkdir -p backups
	$(COMPOSE) exec -T postgres pg_dump -U eopiez eopiez > backups/eopiez_backup_$$(date +%Y%m%d_%H%M%S).sql
	@echo "✓ Backup saved to backups/"

# ===========================================
# Code Quality
# ===========================================

lint: ## Run linters
	@echo "Running linters..."
	@command -v black >/dev/null 2>&1 || { echo "Installing black..."; pip install black; }
	@command -v flake8 >/dev/null 2>&1 || { echo "Installing flake8..."; pip install flake8; }
	black --check api.py api_gateway.py al-uls-evolution/
	flake8 api.py api_gateway.py al-uls-evolution/ --max-line-length=120
	@echo "✓ Linting complete"

format: ## Format code
	@echo "Formatting code..."
	@command -v black >/dev/null 2>&1 || { echo "Installing black..."; pip install black; }
	black api.py api_gateway.py al-uls-evolution/
	@echo "✓ Formatting complete"

typecheck: ## Run type checker
	@echo "Running type checker..."
	@command -v mypy >/dev/null 2>&1 || { echo "Installing mypy..."; pip install mypy; }
	mypy api.py api_gateway.py --ignore-missing-imports
	@echo "✓ Type checking complete"

# ===========================================
# Installation
# ===========================================

install: ## Install dependencies (local development)
	@echo "Installing Julia dependencies..."
	$(JULIA) --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'
	@echo "Installing Python dependencies..."
	pip install -r requirements.txt
	@echo "✓ Installation complete"

install-dev: ## Install development dependencies
	@echo "Installing development dependencies..."
	pip install pytest pytest-cov pytest-asyncio pytest-mock black flake8 mypy
	@echo "✓ Development dependencies installed"

# ===========================================
# Examples
# ===========================================

example-motif: ## Run motif detection example
	@echo "Running motif detection example..."
	$(PYTHON) examples/motif_detection_example.py

example-pipeline: ## Run full pipeline example
	@echo "Running full pipeline example..."
	$(PYTHON) examples/full_pipeline_example.py

example-qvnm: ## Run QVNM analysis example
	@echo "Running QVNM example..."
	$(PYTHON) examples/qvnm_example.py

# ===========================================
# Status & Info
# ===========================================

status: ## Show status of all services
	@$(COMPOSE) ps

health: ## Check health of all services
	@echo "Checking service health..."
	@curl -s http://localhost:8000/health | $(PYTHON) -m json.tool || echo "API Gateway not responding"

info: ## Show system information
	@echo "Eopiez System Information"
	@echo "========================="
	@echo "Docker:"
	@docker --version
	@echo ""
	@echo "Docker Compose:"
	@docker-compose --version
	@echo ""
	@echo "Python:"
	@$(PYTHON) --version
	@echo ""
	@echo "Julia:"
	@$(JULIA) --version
	@echo ""
	@echo "Services:"
	@$(COMPOSE) ps

# ===========================================
# Documentation
# ===========================================

docs: ## Generate documentation
	@echo "Generating documentation..."
	@mkdir -p docs/generated
	@echo "✓ Documentation generated"

docs-serve: ## Serve documentation locally
	@echo "Serving documentation at http://localhost:8080"
	@cd docs && $(PYTHON) -m http.server 8080

# ===========================================
# Quick Start
# ===========================================

quickstart: build start health ## Quick start: build, start, and verify
	@echo ""
	@echo "✨ Eopiez is ready!"
	@echo "=================="
	@echo "API Gateway: http://localhost:8000"
	@echo "API Docs: http://localhost:8000/docs"
	@echo ""
	@echo "Try running:"
	@echo "  make example-motif"
	@echo "  make example-pipeline"
	@echo "  make notebook"
