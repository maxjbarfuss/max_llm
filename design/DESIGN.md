# Max LLM Design

Hybrid LLM (100–500M params), local-first on dual 24GB GPUs, inference on single GPU or CPU.

---

## Architecture

Flow: `Text → Tokenizer → Embeddings → Input FFN → Transformer (MLA + RoPE) → MoE → GRU → Projection (tied)`

- **Tokenizer**: GPT-2 BPE baseline, vocab 50,304; Unigram as exploration candidate
- **Embeddings**: BF16, scaled by `sqrt(hidden_size)`, tied with output projection
- **Context**: 2048 tokens (extensible)
- **MLA**: Full-size Q, latent K/V (`latent_dim=512`), RoPE in latent space
- **MoE**: 16–32 experts, Top-2 routing, BF16 experts + gate
- **Output**: GRU stage for sequential continuity

Architecture is intentionally adjustable. Changes require tests and benchmark evidence.

---

## Training Efficiency

Stable baseline: **BF16 + AMP** throughout (well-supported, numerically safe). Layer in optimizations as the project matures.

**Progressive precision schedule** (adjustable; phase boundaries tunable on stability/performance):
| Phase | Steps | FFN/RNN | Notes |
|-------|-------|---------|-------|
| 1 | 0–30% | BF16 + FP32 master weights | Stable AMP baseline |
| 2 | 30–80% | FP8 (via `torchao` on Ada/Hopper) | Hardware-dependent |
| 3 | 80–100% | FP8/BF16 mixed runtime | Tune to stability |

- **Always BF16**: attention, MoE, embeddings
- **Inference KV cache**: FP8
- **Rollback**: automatic on divergence detection

**Throughput and memory optimizations** (implement in order of impact):
1. Flash Attention 2 — 2–4× attention memory reduction, significant speedup
2. `torch.compile(mode="max-autotune")` — ~20–30% throughput gain
3. Selective gradient checkpointing — trades compute for activation memory
4. FP8 linear/FFN layers — `torchao` or `transformer-engine` on RTX 4090 / H100
5. Activation offloading — CPU offload during forward for memory headroom on deep models

**Exploration track**: FP4 training — evaluate when `torchao` FP4 support matures or target hardware warrants it.

---

## Training Infrastructure

- **Data**: Streaming-first for multi-TB corpus; RAM LRU + SSD token cache
- **Dataloader**: 6 workers, prefetch 2, pinned memory, persistent workers
- **Distributed**: DDP for 100–300M params; FSDP full sharding for 300–500M
- **Scale**: microbatching + gradient accumulation

**Training program:**
1. Pre-train: large mixed corpus, long schedule, precision progression
2. Fine-tune: curated domain data, lower LR, earlier BF16 transition
3. Post-train: SFT, optional DPO/RLHF, calibration, export-ready weights

**Reliability:**
- Required metrics: loss, perplexity, throughput, memory, quantization error, expert utilization
- Required alerts: NaN/Inf, loss spikes, OOM, disk pressure, thermal
- Checkpoints: rolling recent + best, atomic writes, include RNG + precision phase

---

## Tokenizer Exploration Track

Compare GPT-2 BPE vs Unigram on identical data slices: perplexity, throughput, sequence length efficiency, memory. Promote Unigram only if quality/performance is neutral or better. Keep vocabulary alignment constraints explicit when swapping.

---

## Validation Gates

- Smoke overfit test (small model, quick convergence check)
- Precision phase transition stability checks
- Multi-GPU consistency checks
- Attention backend equivalence checks
- Periodic generation regression samples

---

## Delivery Order

1. Base components: embeddings, RoPE, MLA, transformer block ← **current phase**
2. MoE + GRU integration
3. Data pipeline and training loop
4. Distributed training + checkpointing
5. Monitoring + optimization

**Definition of Done**: End-to-end pre-train/fine-tune/post-train reproducible locally; checkpoint recovery reliable; quality/performance stable across precision phases.

---

## Engineering Principles

- **KISS first**: simplest correct design wins
- **TDD-first**: write a failing test before every new behavior
- **Type-driven**: typed public interfaces, validated configs
- **Complexity awareness**: document Big-O for non-trivial paths; set performance budgets for hot paths; prefer asymptotically better algorithms over micro-optimizations
- **SOLID**: SRP, OCP, LSP, ISP, DIP enforced; exceptions require documented benchmark-backed justification
- **Composition over indirection**: clear data flow over deep abstraction chains
- **Reproducibility by default**: deterministic seeds, explicit configs, checkpoint-compatible changes
- **Fail fast**: validate early with precise error messages
- **API discipline**: behavior/interface changes update tests and docs in the same PR
- **Measure before optimizing**: profile first; evidence-backed decisions only

---

## Code Structure

**Vertical slices**: each feature owns both Python and optional C++/CUDA layers.

```
src/config/                        # Phase 0: Configuration
src/models/{feature}/
  {feature}.py                     # Phase 1: Python implementation
  kernels/                         # Optional: C++ acceleration
    include/max_llm/{feature}/
    src/
    tests/
src/core/                          # Shared C++ infrastructure
src/training/                      # Phase 2+: Training pipeline
```

**Adding a feature:**
1. Create `src/models/my_feature/my_feature.py` with PyTorch module
2. Write Phase 1 tests in `tests/unit/test_my_feature.py`
3. If CUDA needed: add `kernels/` with headers, implementation, tests, and CMakeLists.txt
4. Register in root CMakeLists.txt only if kernels exist

---

## Testing Strategy

Layers: unit (Python + C++) → integration (Python) → shape/contract → performance → smoke.

**Phase alignment:**
- Phase 0: Config validation only (no torch dependency)
- Phase 1: Model layer tests (requires torch)
- Phase 2+: Training, distributed, checkpoint tests

**TDD cycle**: Red → Green → Refactor → Coverage. Every new behavior gets a failing test first.

**Python:**
- One test file per module; use fixtures; avoid duplicating setup
- Assert shapes and dtypes; avoid brittle float equality
- Parametrize for CPU and GPU; skip GPU if CUDA unavailable

**C++ (GTest):** Tests live in `src/*/kernels/tests/`. Run via `make test-cpp`.

```bash
make test        # All tests (Python + C++)
make test-py     # pytest only
make test-cpp    # ctest only
make test-cov    # Coverage report
```

---

## Agent Workflow

**Session start:**
1. Read `design/PLAN_CHECKLIST.md` (`Next Steps`, `Running Session Log`)
2. Read the relevant section of this document
3. Check existing tests and interfaces before modifying code

**During work:**
- Keep edits focused and atomic
- Preserve API stability unless the change is intentional and documented
- Add/adjust tests with each behavior change

**Session end:**
- Update `PLAN_CHECKLIST.md` (`Scratch Pad` + `Running Session Log`) with completed work, next step, and any blockers
- Update docs impacted by the change in the same PR

---

## Coding Standards

### Python
- Python 3.10+ idioms only
- Type hints for all public classes and functions
- Minimal inline comments; prefer clear names and docstrings
- Import order: stdlib → third-party → local
- Avoid global state when deterministic behavior is required

### C++
- C++20 required (constexpr, concepts, modern idioms)
- `clang-format` (LLVM config) + `clang-tidy` with zero warnings
- Smart pointers only (`std::unique_ptr`, `std::shared_ptr`); no raw `new`/`delete`
- `const` correctness throughout; prefer pure functions
- Export limited public surface in `max_llm::` namespace; document preconditions in headers
- GPU compute in `.cu` files; CPU fallback always provided
- CUDA error checking macros on all CUDA calls

### Build
- CMake 3.20+, Ninja, C++20 enforced at configure time
- `-O3 -march=native` (release), `-g` always included
- LTO enabled for release; `-Werror` on all targets

---

## Quality Gates

**Before merge:**
```bash
make lint      # ruff + mypy + black check
make test      # Python + C++ tests
make format    # Auto-format Python + C++
```

**Delivery readiness (per new model component):**
1. Interface contract: typed constructor/forward signature + expected tensor shapes
2. Config contract: Pydantic config with defaults and validation constraints
3. Test contract: at least one failing unit test before implementation
4. Performance budget: baseline latency/memory/throughput target
5. Failure modes: documented edge conditions (OOM, NaN/Inf, device mismatch)

---

## Reproducibility Contract

Every training/benchmark run must capture:
- Seed values (Python, NumPy, PyTorch CPU/GPU)
- Config snapshot (commit hash + resolved config values)
- Dataset fingerprint (path, revision, or hash)
- Environment fingerprint (Python, PyTorch, CUDA, GPU model, driver)
- Artifact naming: `run_<date>_<component>_<commit>_<seed>`

Runs missing any of these are exploratory only — not baseline-comparable.

---

## CI Strategy

- **lint-fast (every PR)**: ruff, mypy, black check
- **test (every PR)**: Phase 0/1 Python unit tests
- **gpu-build (nightly/manual)**: CUDA build, GTest, integration/perf smoke tests

---

## Dependency Phases

| Phase | Deps | Use Case |
|-------|------|----------|
| 0 (Base) | `pydantic` | Config validation |
| 1 (Model) | + `torch`, `numpy` | Model building |
| 2+ (Training) | + `transformers`, `datasets`, `accelerate`, `deepspeed`, `xformers`, `bitsandbytes` | Full training stack |
| Dev | `black`, `ruff`, `mypy`, `pytest` | Code quality |

Install: `source setup.sh` (full local stack) or `pip install -r requirements.txt && pip install -e .`.
