# Session Notes

> Quick-start for any contributor (human or agent). Updated every session.
> For execution detail: [PLAN.md](PLAN.md). For architecture: [DESIGN.md](DESIGN.md).

---

## Current Focus

**Phase 2 — data sourcing and end-to-end training validation**

Infrastructure is complete (tokenizer modes, config system, data pipeline framework). Next step is downloading real datasets and proving the learning signal end-to-end.

## Phase 2 Status

Phase 1 complete. Phase 2 ~93% done: infrastructure, tokenizers, config, training loop, checkpointing, and inference all working. WikiText-103 validates end-to-end. Remaining: TinyStories pipeline, seed hardening, overfit test.

**Completed sequence:**
1. ✅ Tokenizer + config system (done — p2-step1)
2. ✅ Data pipeline framework — YAML-driven normalize/tokenize/extract runners (done — p2-step2 through refactors)
3. ✅ CharTokenizer: UTF-8/16/32 + codepoint modes, roundtrip tests (done — p2-step2)
4. ✅ Config system: `config/experiment.toml` fully populated, `src.training.train` wired (done — p2-step1)
5. ✅ BaseLearningModel ABC + SimpleLM: token embedding → GELU MLP → LM head (done — p2-step3)
6. ✅ Data loader + training loop: DataLoader, train_step, loss computation (done — p2-step4)
7. ✅ WikiText-103 pipeline: normalize → tokenize → extract subset (done — recent refactors)
8. ✅ Training validated: 100k-token subset, loss 16.01→2.61 over 500 steps (done)
9. ✅ Checkpointing: `save_checkpoint` saves model + optimizer state to `output_dir/checkpoint.pt`; called from `train` entrypoint (done — this session)
10. ✅ Inference: `src/inference/run.py` — temperature, top-k, top-p sampling; loads checkpoint; text-in → text-out (done — this session)

## Next Steps (Priority Order)

1. 🎯 TinyStories data path + YAML: add `scripts/data/tinystories/default_utf8.yaml`; run normalize/tokenize using `run_data_prep.py`
2. Validate TinyStories token counts (chars/token sanity) and boundary handling (`<|endoftext|>`)
3. Run `python -m src.training.train` on TinyStories subset; capture loss curve and compare vs WikiText small
4. Seed hardening (Python/NumPy/PyTorch CPU/CUDA; deterministic replay)
5. Overfit 10K-token subset (target: loss < 0.1 within 500 steps)
6. (Stretch) Add simple mixed-source sampler: interleave TinyStories + WikiText caches by weight

**Config-driven execution**: All data-prep driven by YAML configs in `scripts/data/<dataset>/`. Single runner:
```bash
python scripts/data/run_data_prep.py --config scripts/data/<dataset>/<config>.yaml
```
See [DESIGN.md — Data Pipeline Reference](DESIGN.md#data-pipeline-reference) for config anatomy and size guide.

---

## Current Session Scratch Pad

> Ephemeral — clear this section at commit time. Use for in-progress notes only.

---

## Running Session Log

**Retention policy**: Keep last 5 sessions. Archive older entries to `docs/archive/` or trim after merge. One row per unique commit marker; update existing row rather than duplicating.

| Date | Commit | Summary |
|---|---|---|
| 2026-02-24 | post-phase2+unstaged | **Checkpoint, inference, quality gate, coverage, doc sync**: Added `save_checkpoint` to `train.py`; inference end-to-end (temp/top-k/top-p). Fixed 9 lint/mypy violations. Refactored `extract_subset` helpers. 31 new unit tests (checkpoint, inference, boundary detectors, `create_simple_loaders`). Moved `install_vscode_extensions.sh` → `scripts/setup/`. Added `.claude/CLAUDE.md` bootstrap. Doc sync: Phase 2 diagram fixed (training/inference split), Option B stubs removed from PLAN.md + DESIGN.md, README stripped of mutable status (L005 added to LESSONS.md), SESSION.md next steps updated. Lint clean, mypy clean, 219 tests, 58% coverage. |
| 2026-02-24 | `48c09c6` | **SKILLS.md consolidation + governance**: Refactored `.github/SKILLS.md` (71→62 lines, 7→4 sections). Added explicit Persona section (5 roles); consolidated Working Discipline, Session Workflow (Start/During/End/Done), Project Patterns (absorbed Protected Files), and Technical Skills (Core + When-relevant, added design principles: SOLID/DRY/KISS/YAGNI/composition). Updated frontmatter (v1.2→v1.3, agentskills.io compatible). Updated CONTRIBUTING.md to route agents to SKILLS.md + LESSONS.md. |
| 2026-02-24 | post-`main`+unstaged | **Docs reorganization + Phase 2 status update**: Renamed design/ → docs/; moved data-specific docs to docs/data/ (DATA_PLAN.md, DATA_WORKFLOW.md); updated 9+ cross-references in main docs; modernized WORKFLOW_DATA_PREP diagram (ASCII → Mermaid flowchart); removed legacy manual CLI sections; refactored docs to config-first approach. Phase 2 status updated to reflect completed tokenizer modes (UTF-8/16/32/codepoint via TokenizerFactory), config-driven data tools, and pipeline infrastructure (normalize/tokenize/extract runners). Consolidated DATA_WORKFLOW + DATA_PLAN into DESIGN.md and PLAN.md; created SESSION.md. Ready for commit. |
| 2026-02-23 | `ad510b1` (Phase 2 steps 1–4) | Steps 1–3 as before. Step 4: `TextChunkDataset` + `make_data_loaders` (12 tests); `train_step` + `train` loop (8 tests). 82 Python tests passing, lint clean. |
| 2026-02-23 | post-`main`+unstaged | Consolidated session summary: setup hardening landed (CUDA preflight, post-venv CUDA path export, Step 4 torch-dependent install ordering, and improved build parallelism controls), then Phase 1 closure/docs governance cleanup completed (`CONTRIBUTING` refactor, checklist canonicalized there, `CONTRIBUTORS` now policy + guidance pointer, torchao coverage added to acceleration tests). Local validation passed: `make lint`, `make format-check`, `make test` (35 Python + 1 C++). |
| 2026-02-20 | `main` | ✅ **Phase 1 near-complete**: Fixed Makefile pytest invocation (`python -m pytest` instead of bare `pytest`) resolving test-py-quick ImportError. Verified: tests (30 Python + 1 C++ PASSED, 1.990s total), quick sanity (0.440s <3 min), CI passing (Run #13), documentation consistent. Acceleration libs still pending functional tests at that time. |
| 2026-02-20 | post-`f31f5cc`+unstaged | CI consolidation final pass: Merged test-ci→main; deleted test-ci branch; trimmed Python matrix to 3.12 only; consolidated lint+test into single `ci` job to eliminate redundant pip installs. Run #13 passed with unified job structure. Updated PLAN.md Phase 1 status (CI pipeline ☑, lint rules ☑, tests ✓30/30). |
