# Max LLM Execution Plan

Purpose: phased execution roadmap and session tracker for both human contributors and AI agents.

**Session rule**: one session = all work since last commit.

**How to use:**
1. Read this file at session start: check Phase Progress and Next Steps sections
2. Pick work from **Next Steps** section
3. Keep temporary work notes in **Current Session Scratch Pad**
4. Log completed work in **Running Session Log** before committing

---

## Phase Progress

| Phase | Status | Focus | Effort | Risk | Data Strategy | Key Artifacts |
|-------|--------|-------|--------|------|---------------|---------------|
| **1** | ✅ Done | Foundation & Tests | M | Low (stabilized) | Setup; no training data | CI workflow, test scaffold, reproducible env notes |
| **2** | 🔄 In Progress | Skeleton | M | Low (scope clarity) | TinyStories + Wiki-103 subset (1M+) | Baseline training logs, checkpoints, tokenizer report |
| **3** | — | Transformer | L | Med (training stability) | OpenWebText subset + Gutenberg (10M+) | Decoder baseline metrics, sampling outputs, integration test evidence |
| **4** | — | Stability | L | High (scale + distributed) | FineWeb / FineWeb-Edu subset | Throughput benchmark report, tokenizer decision memo, distributed training logs |
| **5** | — | Curriculum | XL | High (data complexity) | 100M+ tokens; staged curriculum | Architecture A/B report, curriculum manifest, stage-transition metrics |
| **6** | — | Continual | L | Med (forgetting risk) | SFT instruction pairs; replay buffer | SFT runbook, LoRA adapters/merged weights, continual-learning eval report |
| **7** | — | RL Alignment | XL | High (alignment instability) | Preference data + reward model | Alignment experiment report, reward-model card, safety evaluation summary |
| **8** | — | MoE + MLA | XL | High (routing imbalance) | Partitioned SFT + curriculum routing | MoE routing diagnostics, MLA memory report, dense-vs-sparse comparison |
| **9** | — | GRU Hybrid | L | Med (benchmark variance) | Static + incremental evaluation | Hybrid comparison paper draft, NIAH results, forgetting analysis |

**Artifact naming convention**:
- Use `p<phase>_<artifact>_<yyyymmdd>_<commit>_<seed>` for all outputs (reports, checkpoints, benchmark CSVs).
- Examples: `p4_throughput_report_20260220_5aced98_s42`, `p7_reward_model_card_20260220_5aced98_s42`.

For detailed execution: read below. For architectural context: see [DESIGN.md](DESIGN.md#architecture-overview).

---

## Next Steps

**Current focus: Phase 2 — prototype baseline**

Phase 1 is complete. Phase 2 starts with a prototype-first path to get a runnable baseline quickly, then harden for reproducibility.

**Phase 2 kickoff flow (prototype-first):**

1. **Bootstrap runnable baseline (MVP)**
	- Implement `ExperimentConfig` wiring and minimal `train.py` entrypoint
	- Add character-level tokenizer (encode/decode, 128-char ASCII)
	- Add `BaseLearningModel` interface (`src/models/learning_model/`) and `SimpleLM` implementation
	- Add minimal training loop (`forward → loss → backward → step`)

2. **Wire minimal data path**
	- Source TinyStories + WikiText-103 subset (~1–10M tokens)
	- Build simple train/val split + chunked batch loader
	- Run first end-to-end train command successfully

3. **Prove quick learning signal**
	- Overfit a tiny subset (10K tokens)
	- Target: train loss < 0.1 within 500 steps

4. **Lock reproducibility and reliability**
	- Add seed control (Python, NumPy, PyTorch CPU/CUDA)
	- Add checkpoint save/restore (model + optimizer + step)
	- Verify deterministic replay (same seed → same loss trajectory)

5. **Add tests and interface guards**
	- Unit tests: config, tokenizer, data split, model forward shape, loss, checkpoint, seed
	- Add Phase 9-shaped placeholder interfaces (attention/moe/rnn/inference/alignment) with explicit stub guards
	- Add placeholder boundary tests (import-safe + `NotImplementedError` contract)

**Phase 2 exit criteria:**
- ☐ `python train.py --config config/experiment.toml` trains end-to-end, loss decreases monotonically over 100 steps
- ☐ Save/restore checkpoint with same seed produces bit-identical loss at step N+1
- ☐ ≥8 unit tests passing (config, tokenizer, data chunking, model forward shape, loss, checkpoint, seed, placeholder boundaries)
- ☐ TinyStories + WikiText-103 subset validated (token count matches expected)
- ☐ Overfit test achieves target loss < 0.1 within 500 steps

**Immediate execution order (start now):**
1. Fill `config/experiment.toml` all sections (P2 values: `hidden_size=128`, `vocab_size=128`, 1 layer); `train.py` skeleton loads config, validates, exits 0
2. `src/tokenizer/char_tokenizer.py` — 128-char printable ASCII; `encode`/`decode` with roundtrip tests
3. `src/models/learning_model/` — `BaseLearningModel` ABC; `SimpleLM` (embedding → FFN → weight-tied LM head) with shape tests
4. `src/data/loader.py` — in-memory text → token chunks; `src/training/loop.py` — forward → CE loss → backward → optimizer step
5. Wire into `train.py`; first end-to-end run: loss is finite and decreases over 10 steps
6. Overfit 10K-token subset (target: loss < 0.1 in 500 steps)
7. Seed control + checkpoint save/restore; verify deterministic replay (same seed → same loss at step N+1)
8. Phase 9-shaped placeholder stubs + contract tests; TinyStories/WikiText-103 data pipeline

**Running tests:**
- Quick mixed suite: `make test-quick` (fast Python + C++ quick gate)
- All tests: `make test` (Python + C++)
- Python only: `make test-py` (pytest)
- C++ only: `make test-cpp` (ctest)
- With coverage: `make test-cov`
- Linting/format: `make lint`, `make format-check`, `make format`

**Execution rule:** Write failing tests first (TDD), then implement minimal code to satisfy contracts.

---

### Phase 1: Foundation & Testing Scaffolding

**Goal**: Reproducible development environment, working CI, minimal model test suite (Python + C++), key dependencies installed.

**Dependencies**: None.
**Artifacts**: CI workflow status report, local setup verification log, Phase 1 completion checklist.
**Kill Criteria**: Stop if CI is still red after 2 full root-cause attempts on independent failures.
**Out of Scope**: New model architecture work; any training beyond sanity tests.
**Decision Log**: Record decisions as `P1-DEC-<n>` in Running Session Log.

**Tasks**:

Environment and tooling:
- ☑ Repository structure: vertical slices (embeddings, position, attention, moe, rnn, config, training, core)
- ☑ Config system: Pydantic-based, TOML externalization
- ☑ Data management: DataConfig integration, phase-based dataset sourcing placeholders
- ☑ Dependency management: phase-based requirements (Phase 2: torch, Phase 4+: distributed tools)
- ☑ Build system: CMake 3.20+, C++20, ninja, LTO
- ☑ CI pipeline: GitHub Actions for lint/test on every PR (Run #13 consolidated, passing)
- ☑ Key dependencies installed: flash-attn/xformers/SageAttention/torchao functional checks passing

Components and tests:
- ☑ Minimal model architecture: SimpleMLP (2-layer, 128 hidden, for testing)
- ☑ Python test suite: config validation complete (30/30 tests passing)
- ☑ C++ test suite: baseline C++ test coverage wired with GTest/CTest
- ☑ Test discovery: pytest finds tests; CTest discovers CMake tests
- ☑ Continual testing scripts: Makefile targets + monitoring tools (scripts/ci/) with stable CLI and documented outputs
- ☑ Test reporting: JUnit XML + coverage metrics generated by pytest/ctest

Quality:
- ☑ Makefile targets: lint (ruff+mypy), format (black+clang-format), test (all), test-py, test-cpp, test-cov
- ☑ Lint rules: baseline enforcement in place; strict-zero policy active (all violations fixed)

**Exit Criteria**:
- ☑ GitHub Actions CI passes: lint on every PR (Run #13 passed with consolidated single job)
- ☑ Local build and tests pass: `make test` returns 0
- ☑ `make test-py` discovers and runs baseline Python tests (30/30 passing)
- ☑ `make test-cpp` discovers and runs C++ tests (GTest/CTest, all passing)
- ☑ Coverage report: `make test-cov` computes coverage (pytest-cov integrated)
- ☑ Continual test scripts exist and are documented: Makefile targets (`test-quick`, `test-py`, `test-cpp`, `test`, `test-cov`) + monitoring scripts in scripts/ci/README.md
- ☑ Test reporting: JUnit XML (Python + C++) and coverage metrics generated automatically by make targets
- ☑ Acceleration libraries tested: flash-attn/xformers/SageAttention/torchao functional checks passing
- ☑ All documentation (README, SETUP, CONTRIBUTING, PLAN, DESIGN) is consistent
- ☑ DataConfig validates successfully with placeholder paths

### Phase 2: Skeleton & Reproducibility

**Goal**: Runnable repo, one-command training on a small dataset, reproducible losses and checkpoints.

**Dependencies**: Phase 1 complete (CI green, environment stable, tests discoverable).
**Artifacts**: Tokenizer notes, reproducibility run logs, baseline checkpoint set, overfit report.
**Kill Criteria**: Stop if deterministic replay cannot be achieved after fixing seed, dataloader order, and checkpoint state restore.
**Out of Scope**: Transformer blocks, distributed training, alignment behaviors.
**Decision Log**: Record decisions as `P2-DEC-<n>` in Running Session Log.

**Tasks**:

Data:
- ☐ `src/data/loader.py`: in-memory text → char token ids → chunked batches (MVP; no file I/O required)
- ☐ Source TinyStories + WikiText-103 subset (~1–10M tokens); wire file-based train/val split + validated token counts

Components:
- ☑ Fill `config/experiment.toml` with all sections (P2 values: `hidden_size=128`, `vocab_size=128`, 1 layer); wire into `train.py` entrypoint
- ☑ Character-level tokenizer (`src/tokenizer/char_tokenizer.py`): 128-char printable ASCII, encode/decode with roundtrip tests
- ☑ `BaseLearningModel` ABC (`src/models/learning_model/base.py`): `forward` and factory interface; evolves across phases
- ☑ `SimpleLM` (`src/models/learning_model/simple_lm.py`): token embedding → single-layer linear FFN → weight-tied LM head
- ☐ Option B stubbing (now): implement Phase 2 concrete stubs plus Phase 9-shaped placeholder interfaces (attention, moe, rnn, inference, alignment) with import/compile-safe boundaries

Training and evaluation:
- ☐ Seed control: Python, NumPy, PyTorch (CPU/CUDA)
- ☐ Training loop: forward → loss → backward → step → log
- ☐ Checkpointing: model state, optimizer state, epoch/step
- ☐ Perplexity metrics (train/val, per epoch)

Quality:
- ☐ Unit tests: config loading, tokenizer roundtrip, data splitting
- ☐ Stub boundary tests: placeholder modules import cleanly, expose stable interfaces, and fail with explicit `NotImplementedError` where expected

**Exit Criteria**:
- ☐ `python train.py --config config/experiment.toml` trains end-to-end on TinyStories, loss decreases monotonically over 100 steps
- ☐ Save/restore checkpoint with same seed produces bit-identical loss at step N+1
- ☐ ≥8 unit tests pass: config loading, tokenizer encode/decode roundtrip, data split ratios, model forward shape, loss computation, checkpoint save/load, seed determinism, perplexity calculation
- ☐ Option B stub contract passes: Phase 2 path runnable; Phase 9-shaped placeholders present with validated interfaces
- ☐ TinyStories + WikiText-103 subset (1M+ tokens) downloaded, tokenized, and validated (token count matches expected)
- ☐ Overfit test achieves train loss < 0.1 on a 10K-token subset within 500 steps

---

### Phase 3: Minimal Decoder-Only Transformer

**Goal**: Working GPT-style model, overfits small dataset, generates recognizable text.

**Dependencies**: Phase 2 reproducibility and checkpointing completed.
**Artifacts**: Decoder baseline metrics, generation samples, end-to-end integration test log.
**Kill Criteria**: Stop if causal masking correctness fails after architecture and test fixes (block release until resolved).
**Out of Scope**: Multi-GPU scaling, curriculum pretraining, RL alignment.
**Decision Log**: Record decisions as `P3-DEC-<n>` in Running Session Log.

**Tasks**:

Data:
- ☐ Prepare OpenWebText subset (10–50M tokens) and document duplication rate

Components:
- ☐ Token embedding (vocab_size × d_model)
- ☐ Learned position embedding (max_seq_len × d_model)
- ☐ Transformer block (repeat N times): pre-norm LayerNorm → multi-head causal attention → residual → FFN → residual
- ☐ Multi-head causal self-attention: Q/K/V project, scaled dot-product, upper-triangular mask
- ☐ Feed-forward: Linear → GELU → Linear (d_model → 4×d_model → d_model)
- ☐ LM head: Linear (d_model → vocab_size), weight-tied to token embedding

Training and inference:
- ☐ Cross-entropy loss (next-token prediction)
- ☐ Greedy generation (argmax sampling)
- ☐ Temperature + top-k sampling
- ☐ Weight initialization: Xavier uniform for linear layers, learned embeddings from N(0, 0.02)
- ☐ Hyperparameters: 2–4 layers, 128–256 d_model, 4 heads, 128 context

Quality:
- ☐ Shape/dtype assertions for all layers
- ☐ Integration test: full pipeline (raw text → tokenize → batch → forward → loss → generate) in a single test

**Exit Criteria**:
- ☐ Model overfits 1K-token subset (train loss < 0.5 after 1000 steps; train perplexity < 2.0)
- ☐ Generated 100-token samples contain coherent English phrases (manual inspection logged)
- ☐ All shape/dtype tests pass; causal mask verified (no future token leakage)
- ☐ Integration test passes: end-to-end pipeline from raw text to generated output
- ☐ OpenWebText subset (10–50M tokens) prepared, deduplicated, and deduplication rate documented

---

### Phase 4: Training Stability & Usability

**Goal**: Stable training on 50–100M token datasets, smooth convergence, measured throughput baseline.

**Dependencies**: Phase 3 decoder baseline and generation pipeline validated.
**Artifacts**: Throughput benchmark report, tokenizer selection memo, distributed-consistency test results.
**Kill Criteria**: Stop scaling if NaN/Inf recurs >2 times after stability mitigations (LR, clip, precision, batch schedule).
**Out of Scope**: Architecture upgrades (RMSNorm/RoPE/SwiGLU/GQA), SFT and alignment.
**Decision Log**: Record decisions as `P4-DEC-<n>` in Running Session Log.

**Tasks**:

Data:
- ☐ Download and prepare FineWeb / FineWeb-Edu subset (~50–100M tokens)
- ☐ Implement heuristic data filters for length, language, and perplexity
- ☐ Benchmark BPE vs Unigram on identical corpus slices; select tokenizer for Phase 5+
- ☐ Implement memory-mapped data reads and DataLoader shuffling at scale

Components and training:
- ☐ Learning rate scheduler (linear warmup → cosine decay)
- ☐ Gradient clipping (global norm ≤ 1.0)
- ☐ Weight decay on all params except bias and LayerNorm
- ☐ Mixed precision: `torch.cuda.amp` autocast + GradScaler (fallback to fp32)
- ☐ Gradient accumulation over M micro-batches
- ☐ `torch.compile(mode="max-autotune")` integration; measure gain vs eager mode and document graph breaks
- ☐ Multi-GPU training: DDP for 100–300M params, FSDP for 300–500M

Evaluation and quality:
- ☐ Logging: tokens/sec, GPU memory, eval every N steps
- ☐ Loss curves to CSV or TensorBoard
- ☐ Multi-GPU consistency validation (same seed, same loss across ranks)

**Exit Criteria**:
- ☐ 50–100M token training for 10K+ steps, no NaN/Inf; gradient norm stays within 2× of moving average
- ☐ Loss curve smooth: no single-step spike > 3× running average over any 100-step window
- ☐ Throughput baseline documented: tokens/sec on target hardware (single-GPU and multi-GPU)
- ☐ `torch.compile` throughput gain measured and documented (target: ≥15% over eager mode)
- ☐ Tokenizer benchmark complete: BPE vs Unigram decision documented with compression ratio, vocab size, and throughput
- ☐ FineWeb subset integrated, deduplicated, and memory-mapped; data loading does not bottleneck training
- ☐ Multi-GPU: DDP training produces identical loss to single-GPU at same seed for first 100 steps

---

### Phase 5: Llama-Style Architecture Upgrades + Full Pretraining Corpus

**Goal**: Same param count, better perplexity, longer context handling; assemble full unrestricted world model with multi-phase curriculum.

**Dependencies**: Phase 4 tokenizer decision and stable high-volume training established.
**Artifacts**: Architecture A/B report, curriculum manifest, per-stage training curves, memory/perf summary.
**Kill Criteria**: Stop curriculum progression if stage transition harms val perplexity by >15% without recovery in 1K steps.
**Out of Scope**: Instruction tuning, preference optimization, post-training alignment.
**Decision Log**: Record decisions as `P5-DEC-<n>` in Running Session Log.

**Tasks**:

Data:
- ☐ Assemble 100–500M token corpus with staged curriculum: 75% FineWeb/Wikipedia/GitHub/UCI + curated data (Cosmopedia), 20% adult/controversial, 5% harmful
- ☐ Create source manifest (URLs, licenses, curriculum stage assignments, deduplication stats)
- ☐ Implement dynamic data mixing and sampling weights by curriculum stage
- ☐ Validate curriculum with per-stage loss curves and stage-transition logs; benchmark vs random mixing

Components:
- ☐ RMSNorm (replace LayerNorm, no mean, learnable gain)
- ☐ RoPE on Q/K with explicit formulation (complex or sin/cos pairs)
- ☐ RoPE length extrapolation beyond training context
- ☐ SwiGLU FFN (hidden = 4×d_model×2/3, rounded to 256 multiples)
- ☐ GQA with configurable KV head count (1 = MQA, N = MHA, between = GQA)
- ☐ Flash Attention 2 integrated with GQA forward pass

Evaluation and quality:
- ☐ Per-component unit tests: RMSNorm, RoPE, SwiGLU, GQA
- ☐ A/B comparison script: Phase 4 vs Phase 5 on same data/seed/param count
- ☐ Comparison logs: parameter count, perplexity delta, tokens/sec, peak memory

**Exit Criteria**:
- ☐ Llama-style model achieves lower val perplexity than Phase 4 baseline (same param count, same data, same training steps)
- ☐ RoPE handles 2× training context length without perplexity degradation > 10%
- ☐ A/B results logged per Reproducibility Contract; reproducible across runs
- ☐ Flash Attention 2 active in GQA; memory reduction vs naive attention documented
- ☐ 100–500M token corpus assembled and partitioned across P5a/P5b/P5c; source manifest complete
- ☐ Curriculum stage transitions trigger correctly; per-stage loss curves show continued improvement

---

### Phase 6: Inference Optimization & Fine-Tuning + Continual Learning

**Goal**: 2× generation speedup, SFT-based adaptation, LoRA efficiency, continual learning for domain adaptation.

**Dependencies**: Phase 5 architecture frozen with reproducible checkpoints and selected eval baselines.
**Artifacts**: SFT dataset manifest, LoRA adapter bundle, merged inference checkpoint, continual-learning evaluation report.
**Kill Criteria**: Stop continual SFT if forgetting metric Δ worsens for 3 consecutive evaluations.
**Out of Scope**: Preference-model training and PPO/GRPO rollout optimization.
**Decision Log**: Record decisions as `P6-DEC-<n>` in Running Session Log.

**Tasks**:

Data:
- ☐ Curate/download 1–5M SFT instruction-response pairs (OpenAssistant, Self-Instruct, ShareGPT)
- ☐ Prepare domain-specific subsets for continual-learning experiments

Components:
- ☐ KV-cache for autoregressive decoding (process only new token per step)
- ☐ Top-p (nucleus) sampling and repetition penalty
- ☐ Sampling configurable via `InferenceConfig`
- ☐ Prompt templates (ChatML or Alpaca-style)
- ☐ `ChatFormatter` for multi-turn conversation inference
- ☐ LoRA adapters on Q/K/V/output projections; freeze base model; merge adapters for deployment

Training and evaluation:
- ☐ Continual SFT with replay buffer (10% memory), online domain adaptation, and forgetting measurement
- ☐ Replay buffer memory management (reservoir sampling or ring buffer)
- ☐ Loss masking on assistant-response tokens only
- ☐ Held-out evaluation set and side-by-side generation comparison
- ☐ Select 2–3 standard benchmarks before Phase 6 (e.g., HellaSwag, MMLU subset)
- ☐ Integrate `lm-eval-harness` or equivalent custom benchmark loop
- ☐ Track continual-learning metric: Δ = max(0, perf_before - perf_after)

**Exit Criteria**:
- ☐ KV-cache provides > 2× generation speedup at sequence length 512
- ☐ LoRA-tuned model better instruction-following than base
- ☐ LoRA adds < 1% trainable parameters
- ☐ SFT dataset (1–5M pairs) successfully formatted and tokenized with ChatML/Alpaca templates
- ☐ Continual learning replay buffer implemented and functional
- ☐ Evaluation harness runs successfully on selected benchmarks (e.g., HellaSwag, MMLU)
- ☐ Catastrophic forgetting metric (Δ) calculated and documented

---

### Phase 7: Alignment with DPO/RL + Reward Modeling + Safety Testing

**Goal**: Preference-based + RL-based alignment with learned reward model, >60% preference accuracy on held-out; robust rejection of harmful inputs; reward model calibration.

**Dependencies**: Phase 6 SFT baseline and evaluation harness operational.
**Artifacts**: Preference dataset card, reward-model calibration report, alignment training logs, safety evaluation summary.
**Kill Criteria**: Stop alignment run if KL divergence explodes or reward hacking detected (metric drift without quality gains).
**Out of Scope**: MoE/MLA architecture migration and GRU hybridization.
**Decision Log**: Record decisions as `P7-DEC-<n>` in Running Session Log.

**Tasks**:

Data:
- ☐ Assemble 10K–100K preference pairs (HH-RLHF, UltraFeedback) with 5–10% harmful examples
- ☐ Label 5K–10K examples with multi-dimensional reward scores (quality, safety, factuality)
- ☐ Evaluate optional synthetic preference generation via self-critique (experimental at 100–500M scale)

Components:
- ☐ Train reward model (linear probe or small MLP on last hidden state)
- ☐ Calibrate reward model across domains (length-bias mitigation, domain balance)
- ☐ Implement either DPO (frozen reference) or PPO/GRPO (reward-model-guided)
- ☐ Expose β temperature and LR as alignment config knobs

Training and evaluation:
- ☐ Apply replay buffer (10% of preference data) during alignment to reduce forgetting
- ☐ Collect RL trajectories (if PPO/GRPO): generate rollouts, score with reward model, compute gradients
- ☐ Track reward margin, preference accuracy, KL divergence, reward-model confidence, reward-signal correlation
- ☐ Evaluate win rate vs SFT, adversarial safety behavior, reward-model MSE, and continual-learning degradation

**Exit Criteria**:
- ☐ DPO model > 60% preference accuracy on held-out pairs
- ☐ Generation quality better or safer than SFT baseline
- ☐ Reward margin trend positive throughout training
- ☐ 10K–100K preference pairs and 5K–10K reward labels curated and validated
- ☐ Reward model achieves < 0.1 MSE on held-out test set (if applicable)
- ☐ Safety evaluation: refusal rate on adversarial prompt suite > 80% AND false-refusal rate on benign prompts < 10% (both documented)

---

### Phase 8: MoE + MLA (DeepSeek-Style) + Continual Expert Routing

**Goal**: Upgrade attention from GQA to MLA, add sparse MoE for capacity scaling, enable continual expert specialization.

**Dependencies**: Phase 7 aligned baseline available for dense-vs-sparse comparison.
**Artifacts**: Expert-routing diagnostics, MLA KV-cache reduction report, MoE capacity/overflow analysis.
**Kill Criteria**: Stop sparse rollout if token drop >5% or expert collapse persists beyond 3 mitigation attempts.
**Out of Scope**: Hybrid recurrent architecture and long-context recurrent benchmarking.
**Decision Log**: Record decisions as `P8-DEC-<n>` in Running Session Log.

**Tasks**:

Data:
- ☐ Route SFT/preference data through token-to-expert assignments; measure specialization (entropy, load distribution)

Components:
- ☐ MLA: low-rank KV compression, latent projections, decoupled RoPE, compressed KV-cache path
- ☐ MoE: N routed experts + 1 shared expert, router network with auxiliary load-balance loss, top-k gating
- ☐ MoE capacity controls: token capacity factor, overflow handling, expert config (count, k, hidden dim)
- ☐ Inference routing optimization: efficient expert selection and dispatch during generation

Training and evaluation:
- ☐ Track per-expert utilization and token-dropping rates
- ☐ Integrate MLA + MoE (DeepSeek-V2/V3 style)
- ☐ Measure continual expert-routing drift and load imbalance over phases
- ☐ Compare against dense baseline (same total params): perplexity, throughput, memory, KV-cache reduction, utilization histograms

**Exit Criteria**:
- ☐ MLA achieves smaller KV cache than GQA at comparable perplexity
- ☐ MoE val perplexity ≤ dense at same FLOPs
- ☐ Expert utilization balanced (5–30% per expert with 8 experts)
- ☐ Load-balance loss converges
- ☐ SFT/preference data successfully partitioned for expert routing
- ☐ Expert specialization drift documented across curriculum phases
- ☐ Token dropping rate remains < 1% during training and inference

---

### Phase 9: GRU Hybrid Architecture + Continual Learning Evaluation

**Goal**: Linear-complexity sequence modeling, memory scaling, tradeoff characterization; comprehensive continual learning benchmarking; catastrophic forgetting analysis.

**Dependencies**: Phase 8 sparse architecture stabilized with reproducible evaluation pipeline.
**Artifacts**: Transformer-vs-GRU-vs-hybrid comparison report, NIAH benchmark outputs, catastrophic-forgetting analysis.
**Kill Criteria**: Stop hybrid variants that underperform transformer baseline by >15% perplexity with no memory benefit.
**Out of Scope**: New alignment objectives or additional data-curriculum redesign.
**Decision Log**: Record decisions as `P9-DEC-<n>` in Running Session Log.

**Tasks**:

Data:
- ☐ Finalize static benchmark suite (HellaSwag, MMLU, TruthfulQA)
- ☐ Design incremental domain stream (5–10 domains, ~1K examples each)

Components:
- ☐ GRU-only baseline: embed → stacked GRU → LM head (hidden state aligned to d_model)
- ☐ Hybrid transformer-GRU variants: interleaved and parallel options
- ☐ Reset GRU states at document boundaries

Evaluation:
- ☐ Run continual-learning comparison across transformer, GRU, and hybrid on static + streaming domains
- ☐ Run NIAH at 512/1024/2048 for all models; test extrapolation to 4096 for GRU/hybrid
- ☐ Compare at same data, param count, and training steps
- ☐ Track metrics: perplexity, train/infer tokens-sec, latency per token (ms), peak memory vs sequence length
- ☐ Characterize tradeoff: attention O(n²) vs GRU O(n) vs hybrid
- ☐ Run ablations: GRU/attention layer ratios, placement, RoPE on attention path

**Exit Criteria**:
- ☐ Hybrid matches/beats pure-transformer perplexity with better memory scaling
- ☐ Comparison tables (perplexity, throughput, memory)
- ☐ Generated text samples from all three (transformer, GRU, hybrid)
- ☐ Incremental domain task stream (5–10 domains) fully integrated into evaluation suite
- ☐ Catastrophic forgetting rate formally compared between Transformer, GRU, and Hybrid architectures
- ☐ NIAH retrieval accuracy ≥90% at 2048 tokens (all architectures); GRU/hybrid maintains ≥80% at 4096 tokens (extrapolation)

---

## Current Session Scratch Pad

> *Ephemeral — clear this section at commit time. Use for in-progress notes only.*

(empty)

---

## Running Session Log (Brief)

**Retention policy**: Keep last 5 sessions here. Archive older entries to `design/SESSION_ARCHIVE.md` or trim after merge. Use one row per unique commit marker (for example `post-`main`+unstaged`); if the same marker appears again, update the existing row instead of adding a duplicate.

| Date | Commit | Summary |
|---|---|---|
| 2026-02-23 | `HEAD` (Phase 2 steps 1–3) | Step 1: `config/experiment.toml` + `train.py`. Step 2: `CharTokenizer` 14 tests. Step 3: `BaseLearningModel` ABC + `SimpleLM` (token emb + pos emb + FFN + weight-tied LM head), 12 tests. 64 Python tests passing, lint clean. |
| 2026-02-23 | post-`main`+unstaged | Consolidated session summary: setup hardening landed (CUDA preflight, post-venv CUDA path export, Step 4 torch-dependent install ordering, and improved build parallelism controls), then Phase 1 closure/docs governance cleanup completed (`CONTRIBUTING` refactor, checklist canonicalized there, `CONTRIBUTORS` now policy + guidance pointer, torchao coverage added to acceleration tests). Local validation passed: `make lint`, `make format-check`, `make test` (35 Python + 1 C++). |
| 2026-02-20 | `main` | ✅ **Phase 1 near-complete**: Fixed Makefile pytest invocation (`python -m pytest` instead of bare `pytest`) resolving test-py-quick ImportError. Verified: tests (30 Python + 1 C++ PASSED, 1.990s total), quick sanity (0.440s <3 min), CI passing (Run #13), documentation consistent. Acceleration libs still pending functional tests at that time. |
| 2026-02-20 | post-`f31f5cc`+unstaged | CI consolidation final pass: Merged test-ci→main; deleted test-ci branch; trimmed Python matrix to 3.12 only; consolidated lint+test into single `ci` job to eliminate redundant pip installs. Run #13 passed with unified job structure. Updated PLAN.md Phase 1 status (CI pipeline ☑, lint rules ☑, tests ✓30/30). |
| 2026-02-20 | `f31f5cc` | CI consolidation: Replaced 2 separate jobs (lint-and-type-check + test with job dependency) with single `ci` job containing all steps (install→lint→test). Reduced file from 71→39 lines. Run #13 confirms single-job structure working. Eliminated redundant pip installs (~50% CI time savings). |

---

## Canonical References

- [DESIGN.md](DESIGN.md) (architecture, engineering standards, testing strategy, agent workflow)
- [PLAN.md](PLAN.md) (this file — execution plan for phases, sessions, and daily work)
- [CONTRIBUTING.md](../CONTRIBUTING.md#workflow) (workflow and validation gates)
- [config/*.toml](../config) (authoritative runtime values)
- [Makefile](../Makefile) (build, test, lint targets)
