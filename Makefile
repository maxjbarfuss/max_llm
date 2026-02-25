.PHONY: help lint format format-check type-check check \
		 build build-release build-debug cmake-configure cmake-configure-debug cmake-build cmake-test \
		 test tests test-quick test-py test-py-quick test-cpp test-cov test-report cpp-lint cpp-format cpp-format-check \
         clean clean-py clean-cmake pre-commit-run

ARTIFACTS_DIR ?= artifacts/test
PYTEST_JUNIT ?= $(ARTIFACTS_DIR)/junit-py.xml
PYTEST_COV_XML ?= $(ARTIFACTS_DIR)/coverage-py.xml
PYTEST_SUMMARY ?= $(ARTIFACTS_DIR)/summary-py.md
PYTEST_QUICK_JUNIT ?= $(ARTIFACTS_DIR)/junit-py-quick.xml
PYTEST_QUICK_COV_XML ?= $(ARTIFACTS_DIR)/coverage-py-quick.xml
PYTEST_QUICK_SUMMARY ?= $(ARTIFACTS_DIR)/summary-py-quick.md
PYTEST_COV_JUNIT ?= $(ARTIFACTS_DIR)/junit-py-cov.xml
PYTEST_COV_XML ?= $(ARTIFACTS_DIR)/coverage-py-cov.xml
PYTEST_COV_SUMMARY ?= $(ARTIFACTS_DIR)/summary-py-cov.md
PYTEST_COV_HTML ?= $(ARTIFACTS_DIR)/htmlcov
CTEST_JUNIT ?= $(ARTIFACTS_DIR)/junit-cpp.xml
CTEST_SUMMARY ?= $(ARTIFACTS_DIR)/summary-cpp.md
TEST_QUICK_SUMMARY ?= $(ARTIFACTS_DIR)/summary-quick.md

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
	@echo "    make test-quick            - Run fast mixed tests (Python unit + C++ tests)"
	@echo "    make test-py              - Run pytest (Python unit tests)"
	@echo "    make test-py-quick        - Run pytest on tests/unit only"
	@echo "    make test-cpp             - Run ctest (C++ unit tests, if configured)"
	@echo "    make test-cov             - Run pytest with coverage report"
	@echo "    make test-report          - Display aggregated test results summary"
	@echo ""
	@echo "  📝 Code Quality:"
	@echo "    make check                - Pre-commit gate (format, lint, type, fast tests)"
	@echo "    make lint                 - Run black, ruff, and mypy (Python)"
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
	cmake -B $(CMAKE_BUILD_DIR) -DCMAKE_BUILD_TYPE=$(CMAKE_BUILD_TYPE) -DCMAKE_CUDA_ARCHITECTURES=all -GNinja
	@echo "✓ CMake configured"

cmake-configure-debug:
	@echo "Configuring CMake (Debug)..."
	cmake -B $(CMAKE_BUILD_DIR) -DCMAKE_BUILD_TYPE=Debug -DCMAKE_CUDA_ARCHITECTURES=all -GNinja
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

test-quick: test-py-quick test-cpp
	@echo "✓ Quick mixed tests complete"
	@printf "# Test Summary (Quick)\n\n- Python: $(PYTEST_QUICK_JUNIT)\n- Python coverage: $(PYTEST_QUICK_COV_XML)\n- C++: $(CTEST_JUNIT)\n" > $(TEST_QUICK_SUMMARY)
	@echo "✓ Summary written to $(TEST_QUICK_SUMMARY)"

test-py:
	@echo "Running Python tests..."
	@mkdir -p $(ARTIFACTS_DIR)
	@status=0; \
	python -m pytest tests/ -v --tb=short --junitxml=$(PYTEST_JUNIT) \
		--cov=src --cov-report=xml:$(PYTEST_COV_XML) --cov-report=term-missing || status=$$?; \
	printf "# Test Summary (Python)\n\n- JUnit: $(PYTEST_JUNIT)\n- Coverage: $(PYTEST_COV_XML)\n" > $(PYTEST_SUMMARY); \
	exit $$status
	@echo "✓ Python tests complete"

test-py-quick:
	@echo "Running Python unit tests (quick)..."
	@mkdir -p $(ARTIFACTS_DIR)
	@status=0; \
	python -m pytest tests/unit -v --tb=short --junitxml=$(PYTEST_QUICK_JUNIT) \
		--cov=src --cov-report=xml:$(PYTEST_QUICK_COV_XML) --cov-report=term-missing || status=$$?; \
	printf "# Test Summary (Python Quick)\n\n- JUnit: $(PYTEST_QUICK_JUNIT)\n- Coverage: $(PYTEST_QUICK_COV_XML)\n" > $(PYTEST_QUICK_SUMMARY); \
	exit $$status
	@echo "✓ Python quick tests complete"

test-cpp:
	@echo "Running C++ tests..."
	@mkdir -p $(ARTIFACTS_DIR)
	@if command -v nvcc >/dev/null 2>&1 || command -v nvcc.exe >/dev/null 2>&1 || test -n "${CUDACXX}"; then \
		$(MAKE) cmake-configure; \
		status=0; \
		PROJECT_ROOT=$$(pwd); \
		cd $(CMAKE_BUILD_DIR) && ctest --output-on-failure -j$$(nproc) --output-junit "$$PROJECT_ROOT/$(CTEST_JUNIT)" || status=$$?; \
		printf "# Test Summary (C++)\n\n- JUnit: $(CTEST_JUNIT)\n" > "$$PROJECT_ROOT/$(CTEST_SUMMARY)"; \
		exit $$status; \
	else \
		printf "# Test Summary (C++)\n\n- JUnit: skipped (no CUDA compiler)\n" > $(CTEST_SUMMARY); \
		echo "CUDA compiler not found; skipping C++ tests"; \
	fi
	@echo "✓ C++ tests complete"

test-cov:
	@echo "Running Python tests with coverage..."
	@mkdir -p $(ARTIFACTS_DIR)
	@status=0; \
	python -m pytest tests/ -v --tb=short --junitxml=$(PYTEST_COV_JUNIT) \
		--cov=src --cov-report=xml:$(PYTEST_COV_XML) --cov-report=html:$(PYTEST_COV_HTML) --cov-report=term-missing || status=$$?; \
	printf "# Test Summary (Python Coverage)\n\n- JUnit: $(PYTEST_COV_JUNIT)\n- Coverage XML: $(PYTEST_COV_XML)\n- Coverage HTML: $(PYTEST_COV_HTML)/index.html\n" > $(PYTEST_COV_SUMMARY); \
	exit $$status
	@echo "✓ Coverage report generated ($(PYTEST_COV_HTML)/index.html)"

test-report:
	@echo "\n=== Test Results Summary ==="
	@echo ""
	@if [ -f "$(PYTEST_SUMMARY)" ]; then echo "📊 Python Tests:"; cat "$(PYTEST_SUMMARY)"; echo ""; else echo "❌ Python test summary not found (run 'make test-py' first)"; fi
	@if [ -f "$(CTEST_SUMMARY)" ]; then echo "📊 C++ Tests:"; cat "$(CTEST_SUMMARY)"; echo ""; else echo "❌ C++ test summary not found (run 'make test-cpp' first)"; fi
	@if [ -f "$(PYTEST_COV_SUMMARY)" ]; then echo "📊 Coverage Report:"; cat "$(PYTEST_COV_SUMMARY)"; echo ""; fi
	@echo "Use make test-cov to generate coverage HTML report"

# Code Quality
check: lint test-py-quick
	@echo "✓ Pre-commit checks passed (format, lint, type, fast tests)"

lint:
	.venv/bin/black --check src tests
	.venv/bin/ruff check src tests
	.venv/bin/mypy src

format:
	.venv/bin/black src tests
	.venv/bin/isort src tests
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
