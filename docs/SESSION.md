# Session Notes

> Quick-start for any contributor (human or agent). Updated every session.
> For execution detail: [PLAN.md](PLAN.md). For architecture: [DESIGN.md](DESIGN.md).

---

## Current Focus

**Phase 2 — data sourcing and end-to-end training validation**

Infrastructure is complete (tokenizer modes, config system, data pipeline framework). Next step is downloading real datasets and proving the learning signal end-to-end.

## Immediate Next Steps

1. 🎯 TinyStories data path + YAML: add `scripts/data/tinystories/default_utf8.yaml`; run normalize/tokenize using `run_data_prep.py`
2. Validate TinyStories token counts (chars/token sanity) and boundary handling (`<|endoftext|>`)
3. Run `train.py` on TinyStories subset; capture loss curve and compare vs WikiText small
4. (Stretch) Add simple mixed-source sampler: interleave TinyStories + WikiText caches by weight; run short training for combined sample quality
5. Seed + checkpoint hardening (Python/NumPy/PyTorch CPU/CUDA; save/restore; deterministic replay)
6. Option B placeholder boundary tests (Phase 9-shaped stubs: import clean, raise `NotImplementedError`)

## Quick Commands

```bash
make test-quick   # fast Python + C++ gate (<3 min)
make test-py      # Python only
make test-cpp     # C++ only
make test         # full suite
make test-cov     # with coverage
make lint         # ruff + mypy
make format-check # black + isort + clang-format
make format       # auto-fix formatting
```

---

## Current Session Scratch Pad

> Ephemeral — clear this section at commit time. Use for in-progress notes only.

**Session: 2026-02-24 | Documentation Reorganization + Data Pipeline Refactor**

- Deleted docs/data/DATA_WORKFLOW.md and docs/data/DATA_PLAN.md; distributed content: workflow diagram + design decisions → DESIGN.md; operational quickstart → scripts/data/README.md; phase-appropriate implementation tasks → each phase's Data section in PLAN.md
- Created docs/SESSION.md for live session state (current focus, next steps, scratch pad, log)
- Created scripts/data/README.md with all operational data prep docs (quickstart, config anatomy, size guide, tips)
- Merged .github/CONTRIBUTORS.md into CONTRIBUTING.md (Authorization section); deleted CONTRIBUTORS.md
- Moved SETUP.md → scripts/setup/README.md; updated all references
- Updated Agent Workflow in DESIGN.md and SKILLS.md to read SESSION.md first
- Fixed broken anchors: README.md `#data-strategy-summary` → `#data-strategy`; PLAN.md `#workflow` → `#standard-workflow`
- Restructured src/data/: `cli/` → `pipeline/` (generic tools); `processors/` → `datasets/wikitext/` (dataset-specific); added `datasets/tinystories/` placeholder; moved `prepare_training_data.py` → `scripts/data/`
- Created `src/data/datasets/boundary.py`: `BoundaryDetector` ABC + `WikiTextBoundary`, `TinyStoriesBoundary`, `PatternBoundary`, `NoBoundary`, `get_boundary_detector()` factory
- Refactored `extract_text.py`: removed hardcoded WikiText default + `--no-boundaries`; now uses `--dataset NAME` or `--boundary-pattern REGEX`; YAML configs gain `cutoff.dataset: wikitext`
- Fixed stale test import in `test_normalize_wikitext.py` (`normalize_wikitext` → `src.data.datasets.wikitext.normalize`)
- Ran full test suite (189 tests passing) after refactor
- Ran `train.py --config config/experiment.toml` on `data/fast/wikitext_100k_tokens__utf8.npy`: loss 16.01 → 2.61 over 500 steps (train_samples=697, val_samples=78)
- WikiText-103 small subset: normalized/tokenized/token-subset done; training signal verified
- **Next priority**: TinyStories prep + comparison; explore mixed-source sampling

---

## Running Session Log

**Retention policy**: Keep last 5 sessions. Archive older entries to `docs/archive/` or trim after merge. One row per unique commit marker; update existing row rather than duplicating.

| Date | Commit | Summary |
|---|---|---|
| 2026-02-24 | post-`main`+unstaged | **Docs reorganization + Phase 2 status update**: Renamed design/ → docs/; moved data-specific docs to docs/data/ (DATA_PLAN.md, DATA_WORKFLOW.md); updated 9+ cross-references in main docs; modernized WORKFLOW_DATA_PREP diagram (ASCII → Mermaid flowchart); removed legacy manual CLI sections; refactored docs to config-first approach. Phase 2 status updated to reflect completed tokenizer modes (UTF-8/16/32/codepoint via TokenizerFactory), config-driven data tools, and pipeline infrastructure (normalize/tokenize/extract runners). Consolidated DATA_WORKFLOW + DATA_PLAN into DESIGN.md and PLAN.md; created SESSION.md. Ready for commit. |
| 2026-02-23 | `ad510b1` (Phase 2 steps 1–4) | Steps 1–3 as before. Step 4: `TextChunkDataset` + `make_data_loaders` (12 tests); `train_step` + `train` loop (8 tests). 82 Python tests passing, lint clean. |
| 2026-02-23 | post-`main`+unstaged | Consolidated session summary: setup hardening landed (CUDA preflight, post-venv CUDA path export, Step 4 torch-dependent install ordering, and improved build parallelism controls), then Phase 1 closure/docs governance cleanup completed (`CONTRIBUTING` refactor, checklist canonicalized there, `CONTRIBUTORS` now policy + guidance pointer, torchao coverage added to acceleration tests). Local validation passed: `make lint`, `make format-check`, `make test` (35 Python + 1 C++). |
| 2026-02-20 | `main` | ✅ **Phase 1 near-complete**: Fixed Makefile pytest invocation (`python -m pytest` instead of bare `pytest`) resolving test-py-quick ImportError. Verified: tests (30 Python + 1 C++ PASSED, 1.990s total), quick sanity (0.440s <3 min), CI passing (Run #13), documentation consistent. Acceleration libs still pending functional tests at that time. |
| 2026-02-20 | post-`f31f5cc`+unstaged | CI consolidation final pass: Merged test-ci→main; deleted test-ci branch; trimmed Python matrix to 3.12 only; consolidated lint+test into single `ci` job to eliminate redundant pip installs. Run #13 passed with unified job structure. Updated PLAN.md Phase 1 status (CI pipeline ☑, lint rules ☑, tests ✓30/30). |
