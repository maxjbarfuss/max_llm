# Session Notes

> Quick-start for any contributor (human or agent). Updated every session.
> For execution detail: [PLAN.md](PLAN.md). For architecture: [DESIGN.md](DESIGN.md).

---

## Current Focus

**Phase 3 — In Progress 🔄**

Completed xFormers bug fix (LowerTriangularMask for causal masking) and comprehensive documentation consolidation. Consolidated 6 scattered optimization docs into single OPTIMIZATION.md. Applied KISS/DRY principles across DESIGN.md (17% reduction), PLAN.md, and README.md. All phase documentation synchronized and concise. Multi-backend attention (Flash/Sage/xFormers/Standard) working. Next: Data scale-up (10M→50M→100M+ tokens), extended validation runs.

## Phase 3 Status

**🔄 Phase 3 In Progress (~70%)**

**Completed (all sessions):**
- ✅ `src/tokenizer/bpe_tokenizer.py`: BPETokenizer wrapping tiktoken (gpt2/cl100k_base/o200k_base); `encode`/`decode`/`count_tokens`/`vocab_size`; accepts `**_kwargs` for config compat
- ✅ `src/tokenizer/unigram_tokenizer.py`: UnigramTokenizer wrapping SentencePiece (`model_path` load + `train_model()` helper) integrated into TokenizerFactory as `"unigram"`
- ✅ `src/tokenizer/benchmark.py` + `scripts/benchmark_tokenizers.py`: BPE vs Unigram benchmark utilities and CLI (compression + vocab diversity + encode throughput)
- ✅ Data/inference/training tokenizer wiring updated for unigram (`model_path`) and bpe (`encoding`) paths: `src/data/pipeline/tokenize.py`, `scripts/data/run_data_prep.py`, `src/inference/utils.py`, `src/training/train.py`
- ✅ Benchmark artifact generated on identical WikiText slice: `outputs/p3_tokenizer_benchmark_20260227_local.json` (BPE chars/token=4.608, ~7.28M toks/s; Unigram chars/token=4.018, ~4.84M toks/s)
- ✅ **10M interleaved data-ramp pass executed (TinyStories BPE + WikiText BPE)**:
  - Boundary-aware source extracts: `data/fast/wikitext_70m_docs.txt`, `data/fast/tinystories_70m_docs.txt`
  - Randomized interleaved pages corpus: `data/fast/interleaved_wikitext_tinystories_50m_pages.txt` (+ metadata)
  - BPE token cache: `data/fast/interleaved_wikitext_tinystories_50m_pages_bpe_gpt2.npy` (11,508,251 tokens)
  - Ramp subset artifact: `data/fast/interleaved_wikitext_tinystories_10m_tokens_bpe_gpt2.npy` (10,000,000 tokens)
  - Training run: `config/experiment_p3_interleaved_10m_bpe.toml` → `outputs/p3-interleaved-10m-bpe/checkpoint.pt`
  - Metrics (1000 steps, CUDA): loss 10.83→6.08, ppl 50,307→435, throughput ~216k–239k tokens/s, memory ~704MB
- ✅ `src/tokenizer/__init__.py`: BPETokenizer registered as `"bpe"` in TokenizerFactory
- ✅ `src/models/embeddings/token_embedding.py`: TokenEmbedding (`vocab_size × d_model`), N(0, 0.02) init
- ✅ `src/models/position/learned_position.py`: LearnedPositionEmbedding (`max_seq_len × d_model`), N(0, 0.02) init, returns `(1, T, d_model)` for broadcast
- ✅ `src/models/attention/causal_mha.py`: CausalMultiHeadAttention with Q/K/V proj, causal mask, scaled dot-product, Xavier uniform init
- ✅ `src/models/learning_model/attention_lm.py`: AttentionLM test model (tok emb + pos emb + pre-norm + attention + LM head)
- ✅ `src/models/feedforward/feedforward.py`: FeedForward (Linear → GELU → Linear, 4× expansion); 14 tests
- ✅ `src/models/transformer/transformer_block.py`: TransformerBlock (pre-norm + attention + residual + FFN + residual); 18 tests
- ✅ `src/models/learning_model/decoder_lm.py`: DecoderLM full GPT-style decoder (embeddings → N blocks → final norm → weight-tied head); 18 tests
- ✅ `src/models/learning_model/__init__.py`: Added DecoderLM to exports
- ✅ `src/config/model.py`: Added `model_type` field with validation ("simple_lm", "attention_lm", "decoder_lm", etc.)
- ✅ `src/training/train.py`: Model factory based on model_type; supports decoder_lm
- ✅ All experiment configs: Added `model_type` field to [model] section
- ✅ `config/experiment_p3_attention.toml`: AttentionLM validation config
- ✅ `config/experiment_p3_decoder_lm.toml`: DecoderLM validation config
- ✅ AttentionLM validation training: 500 steps, WikiText BPE 442K tokens; loss 10.82→6.66, ppl 50K→783
- ✅ DecoderLM validation training: 500 steps, WikiText BPE 442K tokens; loss 10.86→6.72, ppl 52K→831
- ✅ AttentionLM inference validated: chat mode generates text successfully
- ✅ `requirements.txt`: added `tiktoken>=0.5.0`
- ✅ Data pipeline BPE support: `run_data_prep.py` + `tokenize.py` extended with `--encoding` flag for BPE tokenizer
- ✅ `scripts/data/wikitext-103/wikitext-103_bpe_gpt2_small.yaml`: YAML config for BPE WikiText prep
- ✅ 373 unit tests passing (319 P2 + 54 P3 new)
- ✅ Integration tests: 4 end-to-end DecoderLM checks (train, overfit, inference, convergence)
- ✅ README/DESIGN diagrams updated for KV-cache + DPO flow and standardized layout
- ✅ `.github/LESSONS.md`: Added L007 — always activate venv for Python commands
- ✅ **NEW: Training loop upgrades (Phase 3.1)**:
  - ✅ `src/training/scheduler.py`: `get_cosine_schedule_with_warmup()` with linear warmup (0→peak) and cosine decay (peak→min_lr); 8 unit tests
  - ✅ `src/training/optimizer.py`: `configure_optimizer_param_groups()` for selective weight decay (exclude bias, LayerNorm, 1D params); 11 unit tests
  - ✅ `src/training/loop.py`: Refactored `train_step()`, added `optimizer_step()`, enhanced `train()` with:
    - Gradient accumulation over M micro-batches
    - Mixed precision (AMP) with `torch.amp.GradScaler` on CUDA, fallback to FP32
    - Gradient clipping (global norm ≤ specified limit, default 1.0)
    - LR scheduler integration with per-step `scheduler.step()`
    - Tokens/sec throughput logging
    - GPU memory usage logging (CUDA only)
  - ✅ Updated `src/training/train.py` to wire all new features:
    - Parameter groups with selective weight decay
    - LR scheduler creation from config
    - Precision schedule parsing (first entry determines AMP vs FP32)
    - Gradient clipping thresholds
    - Gradient accumulation steps
    - Throughput/memory logging flags
  - ✅ Enhanced test suite: 28 new unit tests across 3 files validating scheduler, optimizer grouping, and training loop features
  - ✅ Updated `config/experiment_p3_decoder_lm.toml` with training improvements:
    - `gradient_accumulation_steps=2` (effective batch size 8)
    - `warmup_steps=50` (10% of 500 max_steps)
    - `learning_rate=0.0005` (conservative for stability)
    - `weight_decay=0.01` (small regularization)
    - `log_interval=25` (more frequent logging)
  - ✅ All features validated: scheduler monotonicity, parameter group separation, AMP fallback, gradient accumulation, throughput tracking
- ✅ **NEW: Training optimizations (Phase 3.2)**:
  - ✅ **Flash Attention 2 integration**: Added optional Flash Attention 2 support to `CausalMultiHeadAttention`; passes through `TransformerBlock`, `DecoderLM`, `AttentionLM`; controlled via `use_flash_attention` in config `[training]` section; gracefully falls back to standard attention if unavailable
  - ✅ **DataLoader workers & prefetching**: Added `num_workers`, `prefetch_factor`, `pin_memory`, `persistent_workers` parameters to `create_simple_loaders()`; wired from `[data]` config section; eliminates data loading bottleneck that caused training hangs
  - ✅ **torch.compile support**: Added optional `torch.compile(mode="max-autotune")` call after model creation; controlled via `use_torch_compile` in config; graceful fallback on failure; **note: incompatible with Flash Attention** (choose one)
  - ✅ Created optimization documentation: `docs/OPTIMIZATION_PLAN.md` (detailed strategy), `docs/OPTIMIZATION_QUICKREF.md` (quick reference for config)
  - ✅ Created optimized test config: `config/experiment_p3_optimized_test.toml` (num_workers=4, flash_attn=true, compile=false)
  - ✅ Validated optimizations: 200-step test run successful (loss 10.83→7.62, ppl 50K→2028, throughput ~159-228k tokens/sec)
- ✅ **NEW: Multi-GPU DDP + Profiling (Phase 3.3)**:
  - ✅ **Distributed training module**: `src/training/distributed.py` with utilities for DDP initialization, rank management, model wrapping, sampler creation, and distributed synchronization
  - ✅ **Profiling module**: `src/training/profiling.py` with Timer, GPUMemoryMonitor, ThroughputMonitor, PyTorch profiler integration, and model size logging
  - ✅ **DDP integration into train.py**: Updated imports, added `--distributed` flag, distributed initialization, model wrapping, DDP-aware checkpoint saving, distributed cleanup
  - ✅ **Launcher script**: `scripts/train_ddp.sh` for easy torchrun-based multi-GPU training
  - ✅ **DDP test config**: `config/experiment_p3_ddp_test.toml` for validated multi-GPU runs
  - ✅ **Comprehensive DDP guide**: `docs/DDP_GUIDE.md` with quick start, troubleshooting, best practices
  - ✅ **Validated DDP**: 200-step test with 2 GPUs successful (identical loss across processes, gradient sync working, throughput ~50-57k per GPU)
  - ✅ **All print statements updated**: Using `print_once()` for distributed training (only rank 0 prints)
- ✅ **NEW: Multi-backend attention support (Phase 3.4)**:
  - ✅ **Comprehensive backend integration**: Extended `CausalMultiHeadAttention` with Flash Attention 2, Sage Attention, xFormers, and Standard PyTorch backends
  - ✅ **Automatic fallback**: Graceful degradation if preferred backend unavailable; config-selectable via `attention_backend` parameter
  - ✅ **xFormers bug fix**: Fixed critical memory bug (materialized `[B, num_heads, T, T]` bias tensor) → replaced with `LowerTriangularMask()` for efficient causal masking
  - ✅ **Validated all backends**: Flash, Sage, xFormers, Standard tested on training runs (200-1700+ steps)
  - ✅ **Consolidated documentation**: Created `docs/OPTIMIZATION.md` from 6 scattered optimization docs (ATTENTION_BACKENDS.md, OPTIMIZATION_PLAN.md, OPTIMIZATION_QUICKREF.md, DDP_GUIDE.md, ATTENTION_BACKENDS_INTEGRATION.md, TRAINING_OPTIMIZATIONS.md)
  - ✅ **Documentation cleanup**: Applied KISS/DRY across DESIGN.md (414→343 lines, 17% reduction), PLAN.md synchronized with DESIGN goals, README.md updated

**Remaining Phase 3 work (next sessions):**
- ☐ Tokenizer decision and migration note: document whether to keep BPE default vs promote Unigram on any axis without regressions
- ☐ Data strategy ramp (priority): continue scaling beyond completed 10M stage to 50M → 100M+ token stages
- ☐ TinyStories BPE retokenization (not yet present in `data/fast`; add BPE YAML + artifacts)
- ☐ OpenWebText/FineWeb pipeline tasks: downloader + schema discovery + dedup + Parquet intermediate + chunked token cache
- ☐ Data scale-up execution: prepare larger BPE training corpus and memory-mapped loader path
- ☐ Loss-curve export pipeline (CSV/TensorBoard) for reproducible analysis artifacts
- ☐ Extended DDP testing: longer runs (1000+ steps), multi-node setup validation
- ☐ FSDP exploration (Phase 4 prep): ZeRO-style optimizer sharding for larger models

## Next Steps (Priority Order)

⏭️ **Phase 3 continues**: data strategy ramp (10M → 50M → 100M+), finalize tokenizer decision doc (BPE vs Unigram), larger BPE corpus prep, and training artifact export (CSV/TensorBoard)

**Config-driven execution**: All data-prep driven by YAML configs in `scripts/data/<dataset>/`. Single runner:
```bash
python scripts/data/run_data_prep.py --config scripts/data/<dataset>/<config>.yaml
```
See [DESIGN.md — Data Pipeline Reference](DESIGN.md#data-pipeline-reference) for config anatomy and size guide.

---

## Current Session Scratch Pad

> Ephemeral — clear this section at commit time. Use for in-progress notes only.

**Integration test fixes + attention backend CPU fix (2026-03-02)**: 
1. Fixed create_simple_loaders() calls in 4 test files (integration tests failing due to renamed parameter `tokens→train_tokens` and 3-tuple return value unpacking).
2. Fixed test_attention.py test_multiple_heads() which was trying to use Flash Attention on CPU (doesn't support CPU). Added `attention_backend="standard"` to match other tests.
All 409 tests now passing.

---

## Running Session Log

**Retention policy**: Keep last 5 sessions. Archive older entries to `docs/archive/` or trim after merge. One row per unique commit marker; update existing row rather than duplicating.

| Date | Commit | Summary |
|---|---|---|
| 2026-02-28 | phase3 branch | **Phase 3 — xFormers fix + documentation consolidation**: Fixed xFormers causal attention bug (replaced materialized bias tensor with `LowerTriangularMask()`); validated 1700+ steps. Consolidated 6 scattered optimization docs → `docs/OPTIMIZATION.md` (backends matrix, DDP setup, torch.compile constraints, DataLoader tuning). Applied comprehensive KISS/DRY cleanup: `DESIGN.md` (414→343 lines, 17% reduction), `PLAN.md` synchronized with DESIGN goals (Phase 3 includes optimizations, Phase 4 cleaned of completed items), `README.md` updated (Phase 3 shows multi-backend attention/DDP/torch.compile completed). Deleted obsolete files: ATTENTION_BACKENDS.md, OPTIMIZATION_PLAN.md, OPTIMIZATION_QUICKREF.md, TRAINING_OPTIMIZATIONS.md, ATTENTION_BACKENDS_INTEGRATION.md, DDP_GUIDE.md. |
| 2026-02-27 | phase3 branch | **Phase 3 — training optimizations to fix hangs**: Analyzed training hang issues on dual-GPU hardware (RTX 4090 + RTX 3090 Ti). Created `docs/OPTIMIZATION_PLAN.md` with comprehensive optimization strategy. Implemented: (1) Flash Attention 2 integration in `CausalMultiHeadAttention` with config flag `use_flash_attention` (2-4x attention speedup), (2) DataLoader workers/prefetching (`num_workers`, `prefetch_factor`, `pin_memory`, `persistent_workers`) to eliminate data loading bottleneck, (3) torch.compile support with `use_torch_compile` flag (30-40% speedup, incompatible with Flash Attention). Created `config/experiment_p3_optimized_test.toml` and validated (200 steps: loss 10.83→7.62). Created `docs/OPTIMIZATION_QUICKREF.md` for config reference. Multi-GPU (DDP) and memory-mapped datasets marked for Phase 4. |
| 2026-02-27 | phase3 branch | **Phase 3 — 10M interleaved BPE ramp run (requested)**: Built boundary-aware extracts from full corpora (`/mnt/d/.../wikitext-103-raw/train.txt`, `/mnt/d/.../tinystories-gpt4-clean/train.txt`), created randomized interleaved page mix (`scripts/data/mix_interleaved_pages.py`, seed=42), tokenized with GPT-2 BPE (11.5M tokens), and extracted a 10M-token artifact (`data/fast/interleaved_wikitext_tinystories_10m_tokens_bpe_gpt2.npy`). Ran training pass with new config `config/experiment_p3_interleaved_10m_bpe.toml` for 1000 steps on CUDA; final loss 6.0763, final ppl 435.42; checkpoint at `outputs/p3-interleaved-10m-bpe/checkpoint.pt`. |
| 2026-02-27 | phase3 branch | **Phase 3 — tokenizer benchmark kickoff + unigram path**: Added `UnigramTokenizer` (SentencePiece) and registered it in `TokenizerFactory`; wired unigram/bpe kwargs through data pipeline, inference, and training paths. Added benchmark utility (`src/tokenizer/benchmark.py`) and CLI (`scripts/benchmark_tokenizers.py`) for identical-slice BPE vs Unigram comparisons. Ran benchmark on `data/fast/wikitext_2mb_normalized.txt` (200k chars) and wrote `outputs/p3_tokenizer_benchmark_20260227_local.json`: BPE 4.608 chars/token at ~7.28M toks/s vs Unigram 4.018 chars/token at ~4.84M toks/s. Added/updated tokenizer unit tests; targeted test set passing. |
| 2026-02-27 | phase3 branch | **Phase 3 — quality validation + interactive chat + workflow hardening**: Ran 3000-step combined-corpus training (`combined_wikitext_tinystories_10m_utf8.npy`) on CUDA with full training-stack upgrades enabled (warmup+cosine LR, selective weight decay, AMP bf16, gradient accumulation, clipping, throughput/memory logging). Final metrics: loss 5.7530→2.3846, perplexity 315.14→10.85, throughput ~128–137k tokens/sec, stable ~2132MB GPU memory, no NaN/Inf. Added interactive stdin/stdout chat loop at `src/inference/chat.py` (default temp/top-p/max_tokens, sliding context window, token counting, structured output). Updated `.github/SKILLS.md`, `.github/AGENTS.md`, and `CONTRIBUTING.md` to enforce single-source commit workflow requiring plan + session updates per commit. |
| 2026-02-27 | phase3 branch | **Phase 3 — Decoder stack + diagrams**: Implemented FeedForward (14 tests), TransformerBlock (18 tests), and DecoderLM full GPT-style decoder (18 unit + 4 integration tests). Added DecoderLM model factory support and `experiment_p3_decoder_lm.toml` validation config. Validated training on WikiText BPE 442K tokens (loss 10.86→6.72, ppl 52K→831). Updated README/DESIGN mermaid diagrams (KV-cache + DPO flow, layout standardization). 373 tests passing; lint/mypy/black clean. |
| 2026-02-26 | phase3 branch | **Phase 3 — BPE + embeddings + attention + model factory**: Implemented `BPETokenizer` (tiktoken gpt2/cl100k/o200k), `TokenEmbedding`, `LearnedPositionEmbedding`, `CausalMultiHeadAttention` (Q/K/V proj, causal mask, scaled dot-product, Xavier init), and `AttentionLM` test model (tok+pos+prenorm+attn+head). Added `model_type` field to ModelConfig with factory in train.py. Updated all experiment configs with model_type. Created `experiment_p3_attention.toml` and ran validation: 500 steps, 442K tokens, loss 10.82→6.66, ppl 50K→783; inference chat working. Added LESSONS.md L007: activate venv. 325 tests passing, lint/mypy/black clean. |
| 2026-02-25 | post-`main`+unstaged | **Phase 2 final review + fixes**: Comprehensive pre-Phase-3 review. Found and fixed two bugs introduced by inference refactor: `scripts/evaluate_p2.py` imported `load_checkpoint` and `sample_token` from `src.inference.run` (neither existed post-refactor); corrected to `load_checkpoint_into_model`/`create_tokenizer_from_data_config` from `src.inference.utils` and `sample_token` from `src.inference.sampler`. Fixed `src/data/datasets/wikitext/normalize.py:281` line-length violation (117→98 chars). All 272 tests passing. No other issues: mypy clean, no TODOs/FIXMEs, 97–100% coverage on all critical modules. Phase 2 exit criteria fully met. Ready for Phase 3. |
| 2026-02-25 | post-`main`+unstaged | **Inference refactor + utils consolidation**: Extracted shared inference utilities into `src/inference/utils.py` (device resolution, checkpoint loading, tokenizer creation). Refactored `src/inference/chat.py` and `src/inference/run.py` to use shared helpers, removing duplicate logic. Added `tests/unit/test_inference_utils.py` with 14 tests covering device resolution, checkpoint formats, and tokenizer creation; updated checkpoint/integration tests to import new helper. All unit tests passing (272). Lint + mypy clean. |
| 2026-02-25 | post-`main`+unstaged | **Parameter tuning for real learning + interactive chat**: Created `config/experiment_curriculum.toml` with tuned parameters (hidden_size=256, num_layers=6, batch_size=8, lr=0.0005, dropout=0.1) to enable real learning without overfitting. Prepared full UTF-8 datasets: TinyStories (2.15M tokens) + WikiText-103 (2.13M tokens). Trained on 5000 steps achieving: initial_loss=24.56 → final_loss=2.56 (perplexity 46B → 12.87). Created interactive chat interface `src/inference/chat.py` with temperature/top-p/top-k sampling. Chat mode tested and working. Model generates recognizable word patterns with expected byte-level garbling (Phase 2 baseline). Results summary in `outputs/curriculum-alternating/README.md`. Ready for Phase 3 transformer. |
| 2026-02-25 | post-`main`+unstaged | **Phase 2 Complete — Overfit test (+3 tests, 237→240 passing)**: Created `tests/unit/test_overfit.py` with 3 comprehensive unit tests verifying overfitting behavior on 10K-token subset. Tests confirm: (1) train loss < 0.1 achievable within 500 steps, (2) loss convergence curve monotonically decreases, (3) target threshold reached within 500-step limit. Model config: hidden_size=128, num_layers=1, batch_size=4, lr=0.01 (overfitting LR). All 240 unit tests passing. PLAN.md Phase 2 status updated: ✅ 100% complete (exit criteria met). SESSION.md marked Phase 2 complete. Deterministic overfitting validated. Ready for Phase 3. |
| 2026-02-25 | post-`main`+unstaged | **Seed hardening complete**: Implemented comprehensive seed management via `src/utils/seed.py` with `seed_everything` utility (Python/NumPy/PyTorch CPU/CUDA). Added deterministic training with generator-based DataLoader seeding + worker_init_fn. Created `tests/unit/test_seed.py` with 8 unit tests verifying bit-identical replay across single/multi-step training. Updated `src/training/train.py` to call `seed_everything` at startup. All 237 tests passing. Phase 2 exit criteria updated: seed hardening ✅ (deterministic replay verified). PLAN.md + SESSION.md status updated to ~99% complete. |
