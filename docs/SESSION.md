# Session Notes

> Quick-start for any contributor (human or agent). Updated every session.
> For execution detail: [PLAN.md](PLAN.md). For architecture: [DESIGN.md](DESIGN.md).

---

## Current Focus

**Phase 3 — In Progress 🔄**

Phase 3 started. BPE tokenizer (tiktoken/gpt2) implemented and integrated into data pipeline. Token and positional embedding modules implemented. BPE data prep workflow extended. Exploration training run: SimpleLM + BPE (500 steps, WikiText 2MB) establishes BPE baseline. 304 tests passing.

## Phase 3 Status

**🔄 Phase 3 In Progress (~10%)**

**Completed this session:**
- ✅ `src/tokenizer/bpe_tokenizer.py`: BPETokenizer wrapping tiktoken (gpt2/cl100k_base/o200k_base); `encode`/`decode`/`count_tokens`/`vocab_size`; accepts `**_kwargs` for config compat
- ✅ `src/tokenizer/__init__.py`: BPETokenizer registered as `"bpe"` in TokenizerFactory
- ✅ `src/models/embeddings/token_embedding.py`: TokenEmbedding (`vocab_size × d_model`), N(0, 0.02) init
- ✅ `src/models/position/learned_position.py`: LearnedPositionEmbedding (`max_seq_len × d_model`), N(0, 0.02) init, returns `(1, T, d_model)` for broadcast
- ✅ `requirements.txt`: added `tiktoken>=0.5.0`
- ✅ Data pipeline BPE support: `run_data_prep.py` + `tokenize.py` extended with `--encoding` flag for BPE tokenizer
- ✅ `scripts/data/wikitext-103/wikitext-103_bpe_gpt2_small.yaml`: YAML config for BPE WikiText prep
- ✅ `config/experiment_p3_bpe.toml`: exploration config — SimpleLM + BPE gpt2 (vocab 50304)
- ✅ Exploration run: SimpleLM + BPE, 500 steps, WikiText 442K tokens; loss 26.03→7.69 (ppl 2179)
- ✅ 304 unit tests passing (272 P2 + 19 BPE + 13 embeddings)

**Remaining Phase 3 work (next sessions):**
- ☐ CausalMultiHeadAttention (Q/K/V proj, causal mask, scaled dot-product)
- ☐ FeedForward block (Linear → GELU → Linear, 4× expansion)
- ☐ TransformerBlock (pre-norm LN → MHA → residual → LN → FFN → residual)
- ☐ DecoderLM (embeddings → N blocks → LM head, weight-tied, proper weight init)
- ☐ Training loop upgrades (LR scheduler, AMP, gradient accumulation, tokens/sec logging)
- ☐ Integration test (text → BPE → batch → DecoderLM → loss → generate)
- ☐ OpenWebText/FineWeb data pipeline, HuggingFaceDownloader, Parquet intermediate, ChunkedTokenCache

## Next Steps (Priority Order)

⏭️ **Phase 3 continues**: CausalMultiHeadAttention (TDD red→green), then FFN, TransformerBlock, DecoderLM — vertical slice ending in an overfit test with the full decoder

**Config-driven execution**: All data-prep driven by YAML configs in `scripts/data/<dataset>/`. Single runner:
```bash
python scripts/data/run_data_prep.py --config scripts/data/<dataset>/<config>.yaml
```
See [DESIGN.md — Data Pipeline Reference](DESIGN.md#data-pipeline-reference) for config anatomy and size guide.

---

## Current Session Scratch Pad

> Ephemeral — clear this section at commit time. Use for in-progress notes only.

(cleared)

---

## Running Session Log

**Retention policy**: Keep last 5 sessions. Archive older entries to `docs/archive/` or trim after merge. One row per unique commit marker; update existing row rather than duplicating.

| Date | Commit | Summary |
|---|---|---|
| 2026-02-26 | phase3 branch | **Phase 3 start — BPE tokenizer + embeddings**: Implemented `BPETokenizer` (tiktoken gpt2/cl100k/o200k) with factory registration; `TokenEmbedding` and `LearnedPositionEmbedding` modules with N(0,0.02) init. Extended data pipeline (`run_data_prep.py`, `tokenize.py`) with BPE encoding support + `--encoding` flag. Added `wikitext-103_bpe_gpt2_small.yaml` YAML config and `experiment_p3_bpe.toml`. Exploration training run: SimpleLM + BPE gpt2, 500 steps, 442K tokens, loss 26.03→7.69. Added `tiktoken>=0.5.0` to requirements. 304 tests passing, `make check` clean. |
| 2026-02-25 | post-`main`+unstaged | **Phase 2 final review + fixes**: Comprehensive pre-Phase-3 review. Found and fixed two bugs introduced by inference refactor: `scripts/evaluate_p2.py` imported `load_checkpoint` and `sample_token` from `src.inference.run` (neither existed post-refactor); corrected to `load_checkpoint_into_model`/`create_tokenizer_from_data_config` from `src.inference.utils` and `sample_token` from `src.inference.sampler`. Fixed `src/data/datasets/wikitext/normalize.py:281` line-length violation (117→98 chars). All 272 tests passing. No other issues: mypy clean, no TODOs/FIXMEs, 97–100% coverage on all critical modules. Phase 2 exit criteria fully met. Ready for Phase 3. |
| 2026-02-25 | post-`main`+unstaged | **Inference refactor + utils consolidation**: Extracted shared inference utilities into `src/inference/utils.py` (device resolution, checkpoint loading, tokenizer creation). Refactored `src/inference/chat.py` and `src/inference/run.py` to use shared helpers, removing duplicate logic. Added `tests/unit/test_inference_utils.py` with 14 tests covering device resolution, checkpoint formats, and tokenizer creation; updated checkpoint/integration tests to import new helper. All unit tests passing (272). Lint + mypy clean. |
| 2026-02-25 | post-`main`+unstaged | **Parameter tuning for real learning + interactive chat**: Created `config/experiment_curriculum.toml` with tuned parameters (hidden_size=256, num_layers=6, batch_size=8, lr=0.0005, dropout=0.1) to enable real learning without overfitting. Prepared full UTF-8 datasets: TinyStories (2.15M tokens) + WikiText-103 (2.13M tokens). Trained on 5000 steps achieving: initial_loss=24.56 → final_loss=2.56 (perplexity 46B → 12.87). Created interactive chat interface `src/inference/chat.py` with temperature/top-p/top-k sampling. Chat mode tested and working. Model generates recognizable word patterns with expected byte-level garbling (Phase 2 baseline). Results summary in `outputs/curriculum-alternating/README.md`. Ready for Phase 3 transformer. |
| 2026-02-25 | post-`main`+unstaged | **Phase 2 Complete — Overfit test (+3 tests, 237→240 passing)**: Created `tests/unit/test_overfit.py` with 3 comprehensive unit tests verifying overfitting behavior on 10K-token subset. Tests confirm: (1) train loss < 0.1 achievable within 500 steps, (2) loss convergence curve monotonically decreases, (3) target threshold reached within 500-step limit. Model config: hidden_size=128, num_layers=1, batch_size=4, lr=0.01 (overfitting LR). All 240 unit tests passing. PLAN.md Phase 2 status updated: ✅ 100% complete (exit criteria met). SESSION.md marked Phase 2 complete. Deterministic overfitting validated. Ready for Phase 3. |
| 2026-02-25 | post-`main`+unstaged | **Seed hardening complete**: Implemented comprehensive seed management via `src/utils/seed.py` with `seed_everything` utility (Python/NumPy/PyTorch CPU/CUDA). Added deterministic training with generator-based DataLoader seeding + worker_init_fn. Created `tests/unit/test_seed.py` with 8 unit tests verifying bit-identical replay across single/multi-step training. Updated `src/training/train.py` to call `seed_everything` at startup. All 237 tests passing. Phase 2 exit criteria updated: seed hardening ✅ (deterministic replay verified). PLAN.md + SESSION.md status updated to ~99% complete. |
| 2026-02-24 | `48c09c6` | **SKILLS.md consolidation + governance**: Refactored `.github/SKILLS.md` (71→62 lines, 7→4 sections). Added explicit Persona section (5 roles); consolidated Working Discipline, Session Workflow (Start/During/End/Done), Project Patterns (absorbed Protected Files), and Technical Skills (Core + When-relevant, added design principles: SOLID/DRY/KISS/YAGNI/composition). Updated frontmatter (v1.2→v1.3, agentskills.io compatible). Updated CONTRIBUTING.md to route agents to SKILLS.md + LESSONS.md. |
