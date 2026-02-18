.PHONY: help lint format format-check type-check test test-cov clean pre-commit-run

help:
	@echo "Commands: lint format format-check type-check test test-cov clean pre-commit-run"

lint:
	ruff check src tests && mypy src

format:
	black src tests && isort src tests

format-check:
	black --check src tests && isort --check-only src tests

type-check:
	mypy src

test:
	pytest

test-cov:
	pytest --cov=src --cov-report=html --cov-report=term-missing

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .mypy_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .ruff_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name htmlcov -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name .coverage -delete 2>/dev/null || true

pre-commit-run:
	pre-commit run --all-files
