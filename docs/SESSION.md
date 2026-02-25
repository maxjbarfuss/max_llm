# Session Notes

> Quick-start for any contributor (human or agent). Updated every session.
> For execution detail: [PLAN.md](PLAN.md). For architecture: [DESIGN.md](DESIGN.md).

---

## Current Focus

**Phase 2 — Seed hardening complete**

Comprehensive seed management implemented with `seed_everything` utility. All random sources (Python, NumPy, PyTorch CPU/CUDA) are now properly seeded. Deterministic training verified with 8 unit tests confirming bit-identical replay across multiple steps.

## Phase 2 Status

Phase 1 complete. Phase 2 ~99% done: infrastructure, tokenizers, config, training loop, checkpointing, inference, perplexity metrics, and seed hardening all working. WikiText & TinyStories datasets validated end-to-end with proper boundary detection. Remaining: overfit test.

**Completed sequence (this session additions)**:
- ✅ Seed hardening: `src/utils/seed.py` with `seed_everything` utility
- ✅ Deterministic training: Python/NumPy/PyTorch CPU/CUDA seeds properly managed
- ✅ DataLoader reproducibility: Generator-based seeding + worker_init_fn
- ✅ Deterministic replay verification: 8 unit tests (`tests/unit/test_seed.py`) confirming bit-identical training trajectories
- ✅ Training entrypoint integration: `main()` calls `seed_everything` at startup

**Completed sequence (previous sessions)**:
- ✅ TinyStories pipeline: YAML configs + prepare script (scripts/data/tinystories/)
- ✅ Boundary detector enhancement: `BlankLineBoundary` + updated `TinyStoriesBoundary` to handle both formats
- ✅ Dataset comparison: WikiText (loss 2.61) vs TinyStories (loss 2.74) — similar convergence, both valid
- ✅ TinyStories extraction now respects story boundaries: **5 complete stories** in 100K token subset

**Completed sequence:**
1. ✅ Tokenizer + config system (done — p2-step1)
2. ✅ Data pipeline framework — YAML-driven normalize/tokenize/extract runners (done — p2-step2 through refactors)
3. ✅ CharTokenizer: UTF-8/16/32 + codepoint modes, roundtrip tests (done — p2-step2)
4. ✅ Config system: `config/experiment.toml` fully populated, `src.training.train` wired (done — p2-step1)
5. ✅ BaseLearningModel ABC + SimpleLM: token embedding → GELU MLP → LM head (done — p2-step3)
6. ✅ Data loader + training loop: DataLoader, train_step, loss computation (done — p2-step4)
7. ✅ WikiText-103 pipeline: normalize → tokenize → extract subset (done — recent refactors)
8. ✅ Training validated: 100k-token subset, loss 16.01→2.61 over 500 steps (done)
9. ✅ Checkpointing: `save_checkpoint` saves model + optimizer state to `output_dir/checkpoint.pt`; called from `train` entrypoint (done)
10. ✅ Inference: `src/inference/run.py` — temperature, top-k, top-p sampling; loads checkpoint; text-in → text-out (done)
11. ✅ Perplexity metrics: `train()` returns dict with losses + perplexities; `compute_perplexity()` helper (done — this session)
12. ✅ Integration tests: `tests/unit/test_integration_p2.py` — end-to-end pipeline validation (done — this session)
13. ✅ Evaluation script: `scripts/evaluate_p2.py` — checkpoint assessment with generation samples (done — this session)

## Next Steps (Priority Order)

1. ☐ Overfit 10K-token subset (target: loss < 0.1 within 500 steps)
2. ☐ (Stretch) Add simple mixed-source sampler: interleave TinyStories + WikiText caches by weight
3. ⏭️ **Phase 3 begins**: Minimal Transformer (multi-head causal attention, FFN blocks)

**Config-driven execution**: All data-prep driven by YAML configs in `scripts/data/<dataset>/`. Single runner:
```bash
python scripts/data/run_data_prep.py --config scripts/data/<dataset>/<config>.yaml
```
See [DESIGN.md — Data Pipeline Reference](DESIGN.md#data-pipeline-reference) for config anatomy and size guide.

---

## Current Session Scratch Pad

> Ephemeral — clear this section at commit time. Use for in-progress notes only.

(Cleared for commit)

---

## Running Session Log

**Retention policy**: Keep last 5 sessions. Archive older entries to `docs/archive/` or trim after merge. One row per unique commit marker; update existing row rather than duplicating.

| Date | Commit | Summary |
|---|---|---|
| 2026-02-25 | post-`main`+unstaged | **Seed hardening complete**: Implemented comprehensive seed management via `src/utils/seed.py` with `seed_everything` utility (Python/NumPy/PyTorch CPU/CUDA). Added deterministic training with generator-based DataLoader seeding + worker_init_fn. Created `tests/unit/test_seed.py` with 8 unit tests verifying bit-identical replay across single/multi-step training. Updated `src/training/train.py` to call `seed_everything` at startup. All 237 tests passing. Phase 2 exit criteria updated: seed hardening ✅ (deterministic replay verified). PLAN.md + SESSION.md status updated to ~99% complete. |
| 2026-02-24 | `b69cd4b` | **Perplexity metrics + integration tests + evaluation script**: Added `compute_perplexity()` helper; `train()` now returns dict with losses + perplexities (all call sites updated). Created `tests/unit/test_integration_p2.py` (5 integration tests for full pipeline). Added `scripts/evaluate_p2.py` for checkpoint assessment with generation samples. Updated training loop to log perplexity. All 224 tests passing. Phase 2 exit criteria updated: perplexity metrics ✅, evaluation script ✅. |
| 2026-02-24 | `548da84` | **Checkpoint, inference, quality gate, coverage, doc sync**: Added `save_checkpoint` to `train.py`; inference end-to-end (temp/top-k/top-p). Fixed 9 lint/mypy violations. Refactored `extract_subset` helpers. 31 new unit tests (checkpoint, inference, boundary detectors, `create_simple_loaders`). Moved `install_vscode_extensions.sh` → `scripts/setup/`. Added `.claude/CLAUDE.md` bootstrap. Doc sync: Phase 2 diagram fixed (training/inference split), Option B stubs removed from PLAN.md + DESIGN.md, README stripped of mutable status (L005 added to LESSONS.md), SESSION.md next steps updated. Lint clean, mypy clean, 219 tests, 58% coverage. |
| 2026-02-24 | `48c09c6` | **SKILLS.md consolidation + governance**: Refactored `.github/SKILLS.md` (71→62 lines, 7→4 sections). Added explicit Persona section (5 roles); consolidated Working Discipline, Session Workflow (Start/During/End/Done), Project Patterns (absorbed Protected Files), and Technical Skills (Core + When-relevant, added design principles: SOLID/DRY/KISS/YAGNI/composition). Updated frontmatter (v1.2→v1.3, agentskills.io compatible). Updated CONTRIBUTING.md to route agents to SKILLS.md + LESSONS.md. |
| 2026-02-24 | post-`main`+unstaged | **Docs reorganization + Phase 2 status update**: Renamed design/ → docs/; moved data-specific docs to docs/data/ (DATA_PLAN.md, DATA_WORKFLOW.md); updated 9+ cross-references in main docs; modernized WORKFLOW_DATA_PREP diagram (ASCII → Mermaid flowchart); removed legacy manual CLI sections; refactored docs to config-first approach. Phase 2 status updated to reflect completed tokenizer modes (UTF-8/16/32/codepoint via TokenizerFactory), config-driven data tools, and pipeline infrastructure (normalize/tokenize/extract runners). Consolidated DATA_WORKFLOW + DATA_PLAN into DESIGN.md and PLAN.md; created SESSION.md. Ready for commit. |
