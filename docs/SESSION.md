# Session Notes

> Quick-start for any contributor (human or agent). Updated every session.
> For execution detail: [PLAN.md](PLAN.md). For architecture: [DESIGN.md](DESIGN.md).

---

## Current Focus

**Phase 2 — Complete ✅**

All Phase 2 exit criteria met: infrastructure, tokenizers, config, training loop, checkpointing, inference, perplexity metrics, seed hardening, and overfit testing. 240 unit tests passing. Ready to begin Phase 3 (Transformer decoder).

## Phase 2 Status

**✅ Phase 2 Complete (100%)**

Phase 1 complete. Phase 2 complete: infrastructure, tokenizers, config, training loop, checkpointing, inference, perplexity metrics, seed hardening, and overfit test all working. WikiText & TinyStories datasets validated end-to-end with proper boundary detection. All 240 unit tests passing.

**Completed sequence (this session additions)**:
- ✅ Overfit test: `tests/unit/test_overfit.py` with 3 comprehensive tests
- ✅ Overfit verification: Model achieves train loss < 0.1 on 10K-token subset within 500 steps
- ✅ Convergence validation: Loss trends downward with monotonic approximate decrease
- ✅ Step threshold verified: Target loss achieved within 500-step limit
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

⏭️ **Phase 3 begins**: Minimal Transformer (token/position embeddings, multi-head causal attention, FFN blocks, weight initialization)

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
| 2026-02-25 | post-`main`+unstaged | **Inference refactor + utils consolidation**: Extracted shared inference utilities into `src/inference/utils.py` (device resolution, checkpoint loading, tokenizer creation). Refactored `src/inference/chat.py` and `src/inference/run.py` to use shared helpers, removing duplicate logic. Added `tests/unit/test_inference_utils.py` with 14 tests covering device resolution, checkpoint formats, and tokenizer creation; updated checkpoint/integration tests to import new helper. All unit tests passing (272). Lint + mypy clean. |
| 2026-02-25 | post-`main`+unstaged | **Parameter tuning for real learning + interactive chat**: Created `config/experiment_curriculum.toml` with tuned parameters (hidden_size=256, num_layers=6, batch_size=8, lr=0.0005, dropout=0.1) to enable real learning without overfitting. Prepared full UTF-8 datasets: TinyStories (2.15M tokens) + WikiText-103 (2.13M tokens). Trained on 5000 steps achieving: initial_loss=24.56 → final_loss=2.56 (perplexity 46B → 12.87). Created interactive chat interface `src/inference/chat.py` with temperature/top-p/top-k sampling. Chat mode tested and working. Model generates recognizable word patterns with expected byte-level garbling (Phase 2 baseline). Results summary in `outputs/curriculum-alternating/README.md`. Ready for Phase 3 transformer. |
| 2026-02-25 | post-`main`+unstaged | **Phase 2 Complete — Overfit test (+3 tests, 237→240 passing)**: Created `tests/unit/test_overfit.py` with 3 comprehensive unit tests verifying overfitting behavior on 10K-token subset. Tests confirm: (1) train loss < 0.1 achievable within 500 steps, (2) loss convergence curve monotonically decreases, (3) target threshold reached within 500-step limit. Model config: hidden_size=128, num_layers=1, batch_size=4, lr=0.01 (overfitting LR). All 240 unit tests passing. PLAN.md Phase 2 status updated: ✅ 100% complete (exit criteria met). SESSION.md marked Phase 2 complete. Deterministic overfitting validated. Ready for Phase 3. |
| 2026-02-25 | post-`main`+unstaged | **Seed hardening complete**: Implemented comprehensive seed management via `src/utils/seed.py` with `seed_everything` utility (Python/NumPy/PyTorch CPU/CUDA). Added deterministic training with generator-based DataLoader seeding + worker_init_fn. Created `tests/unit/test_seed.py` with 8 unit tests verifying bit-identical replay across single/multi-step training. Updated `src/training/train.py` to call `seed_everything` at startup. All 237 tests passing. Phase 2 exit criteria updated: seed hardening ✅ (deterministic replay verified). PLAN.md + SESSION.md status updated to ~99% complete. |
| 2026-02-24 | `48c09c6` | **SKILLS.md consolidation + governance**: Refactored `.github/SKILLS.md` (71→62 lines, 7→4 sections). Added explicit Persona section (5 roles); consolidated Working Discipline, Session Workflow (Start/During/End/Done), Project Patterns (absorbed Protected Files), and Technical Skills (Core + When-relevant, added design principles: SOLID/DRY/KISS/YAGNI/composition). Updated frontmatter (v1.2→v1.3, agentskills.io compatible). Updated CONTRIBUTING.md to route agents to SKILLS.md + LESSONS.md. |
