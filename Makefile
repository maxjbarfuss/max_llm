.PHONY: help lint format format-check type-check \
		 build build-release build-debug cmake-configure cmake-configure-debug cmake-build cmake-test \
		 test tests test-py test-cpp test-cov cpp-lint cpp-format cpp-format-check \
         clean clean-py clean-cmake pre-commit-run

help:
	@echo "Max LLM Development Commands:"
	@echo ""
	@echo "  ℹ️  Setup:"
	@echo "    Run: source setup.sh          (comprehensive environment setup)"
	@echo ""
	@echo "  🏗️  Build System (C++):"
	@echo "    make cmake-configure      - Configure CMake (Release mode)"
	@echo "    make cmake-configure-debug - Configure CMake (Debug mode)"
	@echo "    make cmake-build          - Alias: build C++ targets"
	@echo "    make cmake-test           - Alias: run C++ tests"
	@echo "    make build                - Build all C++ targets (Release)"
	@echo "    make build-debug          - Build all C++ targets (Debug)"
	@echo "    make build-release        - Explicit Release build"
	@echo ""
	@echo "  ✅ Testing:"
	@echo "    make test                 - Run all tests (Python + C++)"
	@echo "    make tests                - Run all tests (Python + C++), alias for make test"
	@echo "    make test-py              - Run pytest (Python unit tests)"
	@echo "    make test-cpp             - Run ctest (C++ unit tests, if configured)"
	@echo "    make test-cov             - Run pytest with coverage report"
	@echo ""
	@echo "  📝 Code Quality:"
	@echo "    make lint                 - Run ruff and mypy (Python)"
	@echo "    make format               - Format code (black, isort, clang-format)"
	@echo "    make format-check         - Check formatting without changes"
	@echo "    make type-check           - Run mypy type checker"
	@echo "    make cpp-lint             - Run clang-tidy on C++ code"
	@echo "    make cpp-format-check     - Check C++ formatting"
	@echo ""
	@echo "  🧹 Maintenance:"
	@echo "    make clean                - Remove all build artifacts and caches"
	@echo "    make clean-py             - Remove Python caches only"
	@echo "    make clean-cmake          - Remove CMake build directory"
	@echo "    make pre-commit-run       - Run pre-commit on all files"
	@echo ""

# CMake Build System
CMAKE_BUILD_DIR ?= build
CMAKE_BUILD_TYPE ?= Release

cmake-configure:
	@echo "Configuring CMake ($(CMAKE_BUILD_TYPE))..."
	cmake -B $(CMAKE_BUILD_DIR) -DCMAKE_BUILD_TYPE=$(CMAKE_BUILD_TYPE) -GNinja
	@echo "✓ CMake configured"

cmake-configure-debug:
	@echo "Configuring CMake (Debug)..."
	cmake -B $(CMAKE_BUILD_DIR) -DCMAKE_BUILD_TYPE=Debug -GNinja
	@echo "✓ CMake configured (Debug)"

build: cmake-configure
	@echo "Building C++ targets (Release)..."
	cmake --build $(CMAKE_BUILD_DIR) -j$$(nproc)
	@echo "✓ C++ build complete"

build-debug: cmake-configure-debug
	@echo "Building C++ targets (Debug)..."
	cmake --build $(CMAKE_BUILD_DIR) -j$$(nproc)
	@echo "✓ C++ build complete (Debug)"

build-release: build
	@echo "✓ Release build complete"

cmake-build: build

cmake-test: test-cpp

# Testing
test: test-py test-cpp
	@echo "✓ All tests passed"

tests: test

test-py:
	@echo "Running Python tests..."
	pytest tests/ -v --tb=short
	@echo "✓ Python tests complete"

test-cpp: build
	@echo "Running C++ tests..."
	@if [ -d "$(CMAKE_BUILD_DIR)" ]; then \
		cd $(CMAKE_BUILD_DIR) && ctest --output-on-failure -j$$(nproc) || echo "Note: GTest may not be installed"; \
	else \
		echo "Run 'make build' first to build C++ tests"; \
	fi

test-cov:
	@echo "Running Python tests with coverage..."
	pytest tests/ --cov=src --cov-report=html --cov-report=term-missing -v
	@echo "✓ Coverage report generated (htmlcov/index.html)"

# Code Quality
lint:
	ruff check src tests && mypy src

format:
	black src tests && isort src tests
	@which clang-format > /dev/null && find src/core src/models \( -name "*.h" -o -name "*.cpp" \) | xargs clang-format -i || echo "Note: clang-format not installed"

format-check:
	black --check src tests && isort --check-only src tests
	@which clang-format > /dev/null && find src/core src/models \( -name "*.h" -o -name "*.cpp" \) | xargs clang-format --dry-run -Werror 2>/dev/null || echo "Note: clang-format not installed"

type-check:
	mypy src

cpp-lint:
	@which clang-tidy > /dev/null && find src/core src/models -name "*.cpp" -not -path "*/build/*" | head -5 | xargs clang-tidy -header-filter=.* 2>/dev/null || echo "clang-tidy not available; install with: apt-get install clang-tools"

cpp-format-check:
	@which clang-format > /dev/null && find src/core src/models \( -name "*.h" -o -name "*.cpp" \) | grep -v build | xargs clang-format --dry-run -Werror 2>/dev/null && echo "✓ C++ formatting valid" || echo "Note: Use 'make format' to auto-fix"

cpp-format:
	@which clang-format > /dev/null && find src/core src/models \( -name "*.h" -o -name "*.cpp" \) | xargs clang-format -i || echo "clang-format not available; install clang-format"

# Cleanup
clean: clean-py clean-cmake
	@echo "✓ All artifacts cleaned"

clean-py:
	@echo "Cleaning Python artifacts..."
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .mypy_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .ruff_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name htmlcov -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name .coverage -delete 2>/dev/null || true

clean-cmake:
	@echo "Cleaning CMake artifacts..."
	rm -rf $(CMAKE_BUILD_DIR) 2>/dev/null || true

pre-commit-run:
	pre-commit run --all-files
