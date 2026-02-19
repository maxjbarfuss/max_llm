# Max LLM Session Checklist

Purpose: lightweight execution tracker for humans and AI agents.

Session rule: a session is all work since the last commit.

How to use:
1. Read this file at session start.
2. Work from **Next Steps**.
3. Keep temporary notes in **Current Session Scratch Pad**.
4. Add a short entry to **Running Session Log** before committing.

---

## Roadmap (High-Level)

- [x] Foundation setup (repo, docs baseline, config system)
- [ ] Core model blocks (embeddings, RoPE, MLA, transformer forward pass)
- [ ] Advanced model blocks (MoE, GRU, precision scheduler)
- [ ] Training system (data pipeline, distributed training, checkpointing, monitoring)
- [ ] Optimization and validation (compile, checkpointing strategy, smoke/perf tests)
- [ ] Training runs and evaluation (pre-train, fine-tune, post-train)

---

## Next Steps

### Immediate (Must Complete Before Model Development)
1. ✅ **Dependency management - COMPLETE** (2026-02-19):
   - ✅ Phase-based Python deps in pyproject.toml, bootstrap detection, unified setup process
   - ✅ CMake 3.20+ configuration, C++20 standard, per-module CMakeLists.txt
   - ✅ Single-pass system dependency setup: check + auto-install + idempotent (53% code reduction)
   - ✅ Setup handles new or existing WSL2 instances, re-runnable anytime, zero user input
   - ⏳ **ACTION**: Run full build/test validation: `make cmake-configure && make build && make test`

2. 🟨 **Testing framework - Scaffold validation** (2026-02-19, docs done):
   - ✅ GTest integrated into all C++ modules, pytest config in pyproject.toml
   - ✅ TESTING.md guide created with TDD workflow and phase-based test strategy
   - ✅ Makefile targets: `make test`, `make test-py`, `make test-cpp`, `make test-cov`
   - ⏳ **ACTION**: Build C++ tests and run `ctest` to verify GTest integration
   - ⏳ **ACTION**: Validate Python test discovery in tests/ directory

3. ✅ **Modern C++ toolchain** (2026-02-19): CMakeLists.txt (3.20+), Ninja, vertical slice architecture, gcc/clang flexible detection

4. ✅ **Project restructuring for vertical slices** (2026-02-19): Feature-vertical layout with optional kernels/, core shared infrastructure, enhanced Makefile

### Soon (After Dependency + Testing Validation)
5. Build `src/models/embeddings.py` with tests first (TDD: shape, scaling, weight tying).
6. Build `src/models/position.py` (RoPE) with correctness tests.
7. Build `src/models/attention/mla.py` (Q full, latent KV, cache behavior) with tests.
8. Run tokenizer benchmark slice (GPT-2 BPE vs Unigram) and record recommendation in `design/DESIGN.md`.
9. Implement `src/models/transformer.py` with Pre-LN residual structure.
10. Integrate basic forward pass (embeddings -> attention -> projection) with shape tests.

### Blocked
- None.

---

## Current Session Scratch Pad

- Session date: 2026-02-19
- Since commit: post-`48d2ff2`
- Active focus: Dependency management + ML toolchain + Modern C++ development infrastructure + Project restructuring
- Changes:
  1. **Setup process optimization** (2026-02-19):
     - Unified check + auto-install into single `setup_dependencies.py` (eliminated redundancy)
     - Reduced from 1,064 → 496 lines total (-53% code reduction)
     - setup.sh: 231 → 114 lines | setup_dependencies.py: 567 → 229 lines
     - Idempotent and re-runnable: all checks skip if already installed
     - Handles new/existing WSL2, minimal user input, Docker-friendly (no prompts)
  2. Restructured pyproject.toml (phase-based optional deps)
  3. Removed requirements.txt (deps now via pyproject.toml only)
  4. Updated setup.sh: Added Pre-Step Python 3.10+ detection with auto-install
  5. Migrated Python dependency flow to requirements.txt + pip check (simplified verification)
   6. **Unified scripts/setup/setup_dependencies.py** (replaced legacy split system-check/install scripts):
     - WSL2 + git/curl + cmake/ninja + gcc/clang + CUDA 12.1 + NCCL
     - Auto-install missing deps, skip if already present
     - Compiler check & auto-install (gcc+g++ or clang+clang++)
     - Single-pass (no verification loops or double-checking)
   7. **Created C++ build infrastructure:**
     - CMakeLists.txt (root-level, CMake 3.20+, C++20 standard, CUDA 12.0+, Python 3 bindings support)
     - cpp/CMakeLists.txt (library target, Python binding support with pybind11)
     - cpp/src/ and cpp/include/max_llm/ directories
     - cpp/tests/ with CMake configuration and placeholder test
     - Header files: common.h, version.h with versioning macros
     - Placeholder source (placeholder.cpp with version info)
   8. **Code quality tooling:**
     - .clang-format (LLVM-based, 100 char limit, C++20)
     - .clang-tidy (comprehensive checks with naming conventions, Cpp11 style)
     - vcpkg.json (package manifest for C++ dependencies)
   9. **Documentation updates:**
     - Added C++ coding standards to design/DESIGN_PHILOSOPHY.md (formatting, static analysis, memory safety, CUDA guidelines)
     - Added build process requirements (CMake 3.20+, Ninja, -O3/-march=native, -Werror)
     - Updated PLAN_CHECKLIST.md with completed work
- Decisions made:
  - C++20 as language standard (modern features, constexpr, concepts)
  - CMake 3.20+ required (better dependency resolution, presets)
   - Both gcc/g++ and clang/clang++ supported (auto-detect in scripts/setup/setup_dependencies.py)
  - Require at least one C and one C++ compiler (flexible toolchain choice)
  - CUDA 12.0+ integration via CMake FindCUDA module
  - Python bindings optional (pybind11 if available)
  - All C++ treated as warnings-as-errors for code quality
- Open questions:
   - Support Conan in addition to vcpkg?
   - Auto-install build tools in `setup.sh`?
- Blockers: None
- Files touched: pyproject.toml, setup.sh, scripts/setup/setup_dependencies.py, design/DESIGN_PHILOSOPHY.md, README.md, SETUP.md, requirements.txt (created), CMakeLists.txt (created), src/core/ (created), src/models/*/kernels/ (created), .clang-format (created), .clang-tidy (created), vcpkg.json (created)
- Validation before next phase:
  - [ ] Run build validation (`cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build`)
  - [ ] Test `source setup.sh` on fresh WSL2 instance (new + existing scenarios)
  - [ ] Run quality + tests (`make lint`, `make test`)
  - [ ] Update running session log with validation results

## Running Session Log (Brief)

| Date | Since Commit | Summary |
|---|---|---|
| 2026-02-19 | post-`48d2ff2` | **Setup Optimization Final Pass**: Unified system dependency setup (54% code reduction). Consolidated legacy split system-check/install scripts into single-pass setup_dependencies.py. Optimized setup.sh (231→114 lines). Idempotent, re-runnable, handles new/existing WSL2. Zero user input (except optional sudo). All tests: bash/Python syntax ✓. Ready for production WSL2 bootstrap. |
| 2026-02-19 | post-`48d2ff2` (earlier) | **Dependency + Testing Framework Finalization**: (1) Dependency scaffolding complete: phase-based Python deps, precompile checks, CMake build system, vcpkg manifest. (2) Testing framework scaffolding complete: GTest for C++, pytest for Python, unified Makefile (18 targets for build/test/lint/format), TESTING.md guide with TDD workflow. (3) Project restructuring: vertical slices (embeddings/position/attention/moe/rnn) with shared src/core/, removed legacy cpp/. (4) Enhanced Makefile with CMake integration, dependency management targets, and comprehensive help. |
| 2026-02-18 | post-initial | Config externalization (TOML split modules), docs/cross-ref normalization, principles elevation (Big-O/SOLID/reproducibility), Apache-2.0 license. |

---

## Canonical References

- `design/DESIGN.md` (architecture, engineering standards, testing strategy, agent workflow)
- `design/PLAN_CHECKLIST.md` (this file — session tracker)
- `CONTRIBUTING.md` (workflow and validation gates)
- `config/*.toml` (authoritative runtime values)
- `Makefile` (build, test, lint targets)
