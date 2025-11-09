# Makefile for Research Assistant

.PHONY: help setup install test clean run grobid-start grobid-stop check-services example models grobid-restart lint format lint-fix test-all clean-all dry-run

help:
	@echo "Research Assistant - Available Commands"
	@echo "========================================"
	@echo "Setup:"
	@echo "  make setup          - Initial setup (venv, dependencies, services)"
	@echo "  make install        - Install Python dependencies"
	@echo ""
	@echo "Services:"
	@echo "  make check-services - Check if GROBID and Ollama are running"
	@echo "  make grobid-start   - Start GROBID Docker container"
	@echo "  make grobid-stop    - Stop GROBID Docker container"
	@echo "  make grobid-restart - Restart GROBID"
	@echo "  make models         - Pull required Ollama models (deepseek-r1:8b, nomic-embed-text)"
	@echo ""
	@echo "Running:"
	@echo "  make example        - Run example pipeline"
	@echo "  make run            - Run pipeline (requires ROOT_DIR and TOPIC)"
	@echo "  make dry-run        - Dry run (no file moves)"
	@echo ""
	@echo "Testing:"
	@echo "  make test           - Run test suite"
	@echo "  make test-coverage  - Run tests with coverage"
	@echo ""
	@echo "Maintenance:"
	@echo "  make clean          - Clean cache and outputs"
	@echo "  make clean-all      - Clean everything including venv"
	@echo ""
	@echo "Example usage:"
	@echo "  make run ROOT_DIR=/path/to/papers TOPIC='machine learning in healthcare'"

setup:
	@bash setup.sh

install:
	@echo "Installing dependencies..."
	pip install -r requirements.txt

grobid-start:
	@echo "Starting GROBID service..."
	@if ! command -v docker >/dev/null 2>&1; then \
		echo "Docker not installed. Install Docker Desktop."; \
		exit 1; \
	fi
	@if docker ps -a --format '{{.Names}}' | grep -q '^grobid$$'; then \
		if docker ps --format '{{.Names}}' | grep -q '^grobid$$'; then \
			echo "GROBID is already running"; \
		else \
			echo "Starting existing GROBID container..."; \
			docker start grobid >/dev/null; \
			echo "GROBID started on port 8070"; \
		fi; \
	else \
		echo "Starting new GROBID Docker container..."; \
		docker run -d -p 8070:8070 --name grobid lfoppiano/grobid:0.8.0 >/dev/null; \
		echo "GROBID started on port 8070"; \
	fi

grobid-stop:
	@echo "Stopping GROBID service..."
	@docker stop grobid || true
	@docker rm grobid || true
	@echo "GROBID stopped"

grobid-restart: grobid-stop grobid-start

test:
	@echo "Running tests..."
	pytest tests/ -v

clean:
	@echo "Cleaning cache and temporary files..."
	rm -rf cache/*
	rm -rf outputs/*
	rm -rf __pycache__
	rm -rf **/__pycache__
	rm -rf .pytest_cache
	find . -name "*.pyc" -delete
	@echo "Clean complete"

run:
	@if [ -z "$(ROOT_DIR)" ]; then \
		echo "Error: ROOT_DIR not specified"; \
		echo "Usage: make run ROOT_DIR=/path/to/papers TOPIC='your topic'"; \
		exit 1; \
	fi
	@if [ -z "$(TOPIC)" ]; then \
		echo "Error: TOPIC not specified"; \
		echo "Usage: make run ROOT_DIR=/path/to/papers TOPIC='your topic'"; \
		exit 1; \
	fi
	@echo "Running pipeline..."
	python cli.py process --root-dir "$(ROOT_DIR)" --topic "$(TOPIC)"

dry-run:
	@if [ -z "$(ROOT_DIR)" ]; then \
		echo "Error: ROOT_DIR not specified"; \
		exit 1; \
	fi
	@if [ -z "$(TOPIC)" ]; then \
		echo "Error: TOPIC not specified"; \
		exit 1; \
	fi
	@echo "Running pipeline in dry-run mode..."
	python cli.py process --root-dir "$(ROOT_DIR)" --topic "$(TOPIC)" --dry-run

check-services:
	@echo "Checking required services..."
	@echo ""
	@echo "GROBID:"
	@curl -s http://localhost:8070/api/isalive > /dev/null && echo "  ✓ Running on port 8070" || echo "  ✗ Not running (use: make grobid-start)"
	@echo ""
	@echo "Ollama:"
	@ollama list > /dev/null 2>&1 && echo "  ✓ Running" || echo "  ✗ Not running (install from https://ollama.ai)"
	@if command -v ollama >/dev/null 2>&1; then \
		REQ="deepseek-r1:8b nomic-embed-text"; \
		for M in $$REQ; do \
			if ollama list | grep -q $$M; then echo "    ✓ $$M available"; else echo "    ✗ $$M missing (run: ollama pull $$M)"; fi; \
		done; \
	fi
	@echo ""
	@echo "Tesseract:"
	@which tesseract > /dev/null && echo "  ✓ Installed: $$(tesseract --version | head -n1)" || echo "  ✗ Not installed (brew install tesseract)"
	@echo ""

example:
	@echo "Running example pipeline..."
	python example.py

test-coverage:
	@echo "Running tests with coverage..."
	pytest tests/ --cov=core --cov=utils --cov=cache --cov-report=html --cov-report=term

lint:
	@echo "Running linters..."
	@echo "Black..."
	black --check core/ utils/ cache/ tests/ *.py
	@echo "isort..."
	isort --check-only core/ utils/ cache/ tests/ *.py
	@echo "Flake8..."
	flake8 core/ utils/ cache/ tests/ *.py
	@echo "MyPy..."
	mypy core/ utils/ cache/

format:
	@echo "Formatting code..."
	black core/ utils/ cache/ tests/ *.py
	isort core/ utils/ cache/ tests/ *.py

lint-fix: format
	@echo "Code formatted"

test-all: lint test test-coverage
	@echo "All tests and linting complete"

clean-all: clean
	@echo "Removing virtual environment..."
	rm -rf venv/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf htmlcov/
	rm -f .coverage
	rm -f coverage.xml

models:
	@echo "Pulling required Ollama models..."
	@if command -v ollama >/dev/null 2>&1; then \
		ollama pull deepseek-r1:8b || true; \
		ollama pull nomic-embed-text || true; \
	else \
		echo "Ollama not installed. Install from https://ollama.ai"; \
		false; \
	fi
