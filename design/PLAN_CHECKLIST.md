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
   - ✅ Full setup validation passed on WSL2 Ubuntu 24.04 (Python 3.13, CUDA 12.9, PyTorch 2.10.0+cu128)
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

- Session date: 2026-02-20
- Since commit: post-`5aced98`
- Active focus: Setup hardening — tested and fixed all setup failures on WSL2 Ubuntu 24.04
- Changes:
  1. **setup.sh hardening** (2026-02-20):
     - Python selection: prefer highest available 3.10–3.13 w/ `import ensurepip` check; install 3.13 from deadsnakes PPA only as fallback
     - Pre-cache `sudo -v` before Python subprocess (avoids blocking in non-interactive subshell)
     - Venv recovery: detect corrupt .venv (missing activate script), recreate automatically
     - CUDA PATH: glob `/usr/local/cuda-12.*` instead of hardcoded 12.1
     - GPU packages (xformers, deepspeed, bitsandbytes) installed AFTER torch (Step 4)
     - All bare `python` references → `$PYTHON_CMD`
  2. **setup_dependencies.py hardening** (2026-02-20):
     - `dpkg_repair()`: pre-creates `/usr/local/cuda-*` directories expected by config packages, runs `dpkg --configure -a` — called before every `apt_install()` for crash resilience
     - `find_best_cuda_toolkit()`: queries `apt-cache search` for best available `cuda-toolkit-12-x` package (Ubuntu 24.04 has 12-5 through 13-1, not 12-1)
     - `detect_ubuntu_version()`: reads `/etc/os-release` for CUDA repo URL (no more hardcoded `ubuntu2204`)
     - cuDNN auto-install via `apt install cudnn9-cuda-12` (was manual-only)
     - CUDA/nvcc treated as optional; only `nvidia-smi` is critical failure
     - `wget` added to system dependency CHECKS
     - Bare `except:` → `except Exception:`
     - Default subprocess timeout 30s → 300s
  3. **requirements.txt**: removed xformers, deepspeed, bitsandbytes (now in setup.sh Step 4 after torch)
  4. **ci.yml**: added CPU torch install for CI runners
- Final test result (ALL GREEN):
  - nvidia-smi ✓, nvcc ✓ (12.9), NCCL ✓, cuDNN ✓ (9.19.0.56)
  - Python 3.13.12, PyTorch 2.10.0+cu128, CUDA available: True
  - UV 0.10.4, 12 VSCode extensions, pip check clean
- Decisions made:
  - Python 3.13 from deadsnakes PPA (only installed if no suitable dev Python exists)
  - CUDA toolkit auto-detected from apt-cache (best available 12.x)
  - dpkg_repair() adds resilience against interrupted package installs
- Open questions: None
- Blockers: None
- Files touched: setup.sh, scripts/setup/setup_dependencies.py, requirements.txt, .github/workflows/ci.yml, SETUP.md, design/PLAN_CHECKLIST.md
- Validation before next phase:
  - [x] Test `source setup.sh` on WSL2 instance — ALL GREEN
  - [ ] Run build validation (`cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build`)
  - [ ] Run quality + tests (`make lint`, `make test`)

## Running Session Log (Brief)

| Date | Since Commit | Summary |
|---|---|---|
| 2026-02-20 | post-`5aced98` | **Setup Hardening & Validation**: Tested `source setup.sh` on WSL2 Ubuntu 24.04 and fixed all failures. Python: prefer 3.13→3.10 w/ ensurepip check, deadsnakes fallback. CUDA: auto-detect best toolkit via apt-cache (12.9), dpkg_repair() for crash resilience, cuDNN auto-install. Venv: detect corrupt .venv, recreate. All tests pass: Python 3.13.12, PyTorch 2.10.0+cu128, CUDA available, UV 0.10.4, pip check clean. |
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
