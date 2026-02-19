.PHONY: help check setup install-extensions lint format format-check type-check test test-cov clean pre-commit-run

help:
	@echo "Max LLM Development Commands:"
	@echo ""
	@echo "  Setup & Environment:"
	@echo "    make check                - Check system requirements"
	@echo "    make setup                - Run full setup (setup.sh)"
	@echo "    make install-extensions   - Install required VSCode extensions"
	@echo ""
	@echo "  Code Quality:"
	@echo "    make lint                 - Run ruff and mypy checks"
	@echo "    make format               - Format code with black and isort"
	@echo "    make format-check         - Check code formatting without changes"
	@echo "    make type-check           - Run mypy type checker"
	@echo ""
	@echo "  Testing & Verification:"
	@echo "    make test                 - Run pytest"
	@echo "    make test-cov             - Run pytest with coverage report"
	@echo "    make pre-commit-run       - Run pre-commit on all files"
	@echo ""
	@echo "  Maintenance:"
	@echo "    make clean                - Remove build artifacts and caches"
	@echo ""

check:
	python3 check_system.py

setup:
	bash setup.sh

install-extensions:
	bash install_vscode_extensions.sh

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
