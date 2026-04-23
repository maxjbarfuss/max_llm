# Max LLM Execution Plan

Purpose: phased execution roadmap for human contributors and AI agents.

**How to use:**
1. Read [MEMORY.md](../.github/MEMORY.md) first — current focus and agent working state; read [SESSION_LOG.md](../.github/SESSION_LOG.md) for recent completed-session history and [SESSION_LOG_ARCHIVE.md](../.github/SESSION_LOG_ARCHIVE.md) for older history when needed
2. Check Phase Progress table and the current phase's task list for execution detail
3. For architecture and design decisions: [DESIGN.md](DESIGN.md)
4. For completed phases (full history): [PHASE_1_CLOSEOUT.md](PHASE_1_CLOSEOUT.md) | [PHASE_2_CLOSEOUT.md](PHASE_2_CLOSEOUT.md) | [PHASE_3_CLOSEOUT.md](PHASE_3_CLOSEOUT.md) | [PHASE_4_CLOSEOUT.md](PHASE_4_CLOSEOUT.md)

---

## Phase Progress

| Phase | Status | Focus | Effort | Risk | Data Strategy | Key Artifacts |
|-------|--------|-------|--------|------|---------------|---------------|
| **1** | ✅ Done | Foundation | M | Low (stabilized) | Setup; no training data | CI workflow, test scaffold, env notes. [Phase 1 Closeout](PHASE_1_CLOSEOUT.md) |
| **2** | ✅ Done | Skeleton & Reproducibility | M | Low (scope clarity) | TinyStories + WikiText-103 (1–10M tokens) | Tokenizer, data pipeline, training loop, checkpointing, seed control, overfit test. [Phase 2 Closeout](PHASE_2_CLOSEOUT.md) |
| **3** | ✅ Done | Capable GPT-2-like model (~60M params, coherent output) | L | Medium | Mixed corpus: TinyStories (~10%), WikiText-103 (full), OpenWebText (~12%), FineWeb-Edu (partial); Unigram 8K tokenizer | Architecture + optimization stack complete. Two milestone runs: p3_final_unigram (ppl 24.0, 12K steps) and p3_final_27b_merge50 (ppl 28.9, 10,836 steps on 27B-token corpus). Coherent output gate passed. [Phase 3 Closeout](PHASE_3_CLOSEOUT.md) |
| **4** | ✅ Done | Llama Architecture + Scale-Up Training | L | Medium | Wikipedia → Cosmopedia-v2 → mixed curriculum; existing 27B corpus for P3 comparison | P3 vs P4 ppl comparison, 5-stage curriculum loss curves, simplicity anneal; final checkpoint (`p4_final_anneal_20260422`, val_loss 2.323, ppl ~9.2). [Phase 4 Closeout](PHASE_4_CLOSEOUT.md) |
| **5** | — | Post-Training | XL | High (forgetting + alignment) | SFT, grounding, preference data | LoRA adapters, grounding benchmark, reward-model card, safety evaluation |
| **6** | — | MoE + MLA | XL | High (routing imbalance) | Partitioned SFT + preference with curriculum | MoE routing diagnostics, MLA memory report, dense-vs-sparse comparison |
| **7** | — | Dual-Stream Reasoning | XL | High (training-inference mismatch) | Reasoning trace triples + STaR | Dual-stream comparison, reasoning accuracy delta, GRU overhead benchmark |

**Artifact naming convention**:
- Use `p<phase>_<artifact>_<yyyymmdd>_<commit>_<seed>` for all outputs (reports, checkpoints, benchmark CSVs).
- Examples: `p4_throughput_report_20260220_5aced98_s42`, `p5_reward_model_card_20260220_5aced98_s42`.
- Note: Phase numbering updated Feb 2026 (8→7 phases); legacy p6/p7/p8 artifacts may exist in outputs/.

For detailed execution: read below. For architectural context: see [DESIGN.md](DESIGN.md#architecture-overview).

---

### Phase 1: Foundation

**Status**: ✅ Complete
**Goal**: Reproducible environment, CI, testing baseline, and dependency/tooling foundation.
**Historical detail**: see [PHASE_1_CLOSEOUT.md](PHASE_1_CLOSEOUT.md).

**Exit Criteria**:
- ✅ CI baseline stable and green.
- ✅ Build/test/lint targets operational (`make` workflow established).
- ✅ Python + C++ test discovery/reporting validated.
- ✅ Acceleration stack dependency checks completed.

### Phase 2: Skeleton & Reproducibility

**Status**: ✅ Complete
**Goal**: Runnable small-scale training with deterministic replay and checkpoint integrity.
**Historical detail**: see [PHASE_2_CLOSEOUT.md](PHASE_2_CLOSEOUT.md).

**Exit Criteria**:
- ✅ End-to-end training run converged on milestone config.
- ✅ Seeded replay and checkpoint restore behavior verified.
- ✅ Overfit/inference sanity gates passed.
- ✅ Phase 2 milestone artifacts promoted and documented.

### Phase 3: Capable GPT-2-like Model

**Status**: ✅ Complete
**Goal**: Coherent-output decoder model with stable long-run training stack.
**Historical detail**: see [PHASE_3_CLOSEOUT.md](PHASE_3_CLOSEOUT.md).

**Exit Criteria**:
- ✅ Coherent output gate passed on milestone checkpoint.
- ✅ Optimization stack stability validated (Flash/bf16/AdamW-fused/WSD/DDP).
- ✅ Phase 3 milestone runs completed and promoted.
- ✅ Phase 3 closeout captures tokenizer/model/data arc and evidence.

### Phase 4: Llama Architecture + Scale-Up Training

**Status**: ✅ Complete
**Goal**: Llama-class architecture + curriculum training + final anneal, outperforming Phase 3 baseline.
**Historical detail**: see [PHASE_4_CLOSEOUT.md](PHASE_4_CLOSEOUT.md).

**Exit Criteria**:
- ✅ 5-stage curriculum completed and consolidated.
- ✅ P3 baseline surpassed (ppl improvement documented).
- ✅ Final anneal checkpoint promoted with validated metrics.
- ✅ Methodology findings (plateau escape, curriculum ordering) documented in closeout and optimization learnings.

---

### Phase 5: Post-Training

**Goal**: KV-cache, SFT with LoRA, grounding (math/logic/world-model/games), DPO or PPO/GRPO, continual learning. Also: training-stack upgrade (Muon optimizer, Z-loss, μP), data-pipeline efficiency (sequence packing), and architecture extensions (YaRN context extension, MoD routing).

**Dependencies**: Phase 4 architecture frozen with reproducible checkpoints and selected eval baselines.
**Artifacts**: SFT dataset manifest, LoRA adapter bundle, grounding benchmark report, preference dataset card, reward-model calibration report, alignment training logs, safety evaluation summary, continual-learning evaluation report; Muon vs AdamW comparison; μP proxy sweep result.
**Kill Criteria**: Stop if forgetting metric Δ worsens for 3 consecutive evaluations OR if alignment causes >20% degradation on base capabilities.
**Out of Scope**: Forward architecture changes (Phase 6: MoE), reasoning pipeline architecture (Phase 7).
**Decision Log**: Record decisions as `P5-DEC-<n>` in Running Session Log.

**Tasks**:

Execution order (locked for Phase 5):

Wave 0 — data gates (must complete before new pretraining or stack tuning):
- ✅ **Remove WikiText-103** from future data configs (covered by full Wikipedia; overlapping fragments add noise)
- ✅ **Deduplicate FineWeb ↔ FineWeb-Edu** and choose one primary policy (quality-first Edu subset vs full FineWeb without Edu overlap)
:  Policy decision: use **full OpenWebText + full FineWeb-Edu** as Wave 0 core web sources, then apply hygiene + dedup gates before training.
- ✅ **OWT hygiene pass** in `TextFormatReader`: punctuation-ended line threshold, duplicate-line threshold, symbol/word threshold
- ☐ **Raise OWT min_length** from 50 to 100-150 tokens
- ☐ **Language filter** for OWT/FineWeb during tokenization (fastText lid.176)
- ☐ **MinHash LSH near-dedup** across OWT + FineWeb (Jaccard threshold 0.8); record dedup rate
- ☐ **Sequence packing** for short documents with block-diagonal causal masking (target 20-40% throughput gain)
- ☐ **Per-source repetition budget tracking** across stages (avoid overexposure/memorization)
- ☐ **Document Cosmopedia provenance** as synthetic LLM-generated data in corpus docs

Wave 0 quick-start (simple):
1. Activate env/auth and create one ephemeral prep config (`config/ephemeral/p5_wave0_data_gates_20260423.toml`).
2. Apply corpus hygiene in prep pipeline: remove WikiText-103, include full OpenWebText + full FineWeb-Edu (+ selected companion datasets), add OWT quality filters + min_length + language filter.
3. Add near-dedup (MinHash) and sequence packing; record drop/dedup/packing metrics in stats output.
4. Run a small-slice prep first, then targeted prep tests (`test_preparation_config`, `test_preparation_strategies`, `test_preparation_pipeline`, `test_preparation_e2e`).
5. If metrics are good, run full prep and document final policy + provenance note in `src/data/README.md`.

Wave 1 — tokenizer + corpus shape decision (depends on Wave 0):
- ☐ **Tokenizer decision checkpoint**:
	- Train new **32K Unigram tokenizer** on cleaned Phase 4 distribution if doing any additional pretraining in Phase 5
	- Or explicitly defer tokenizer rebuild if Phase 5 is LoRA-only SFT/grounding/alignment on frozen base
- ☐ **Optional new sources (only after cleanup policy is stable)**:
	- Long-form books (Project Gutenberg) for long-horizon coherence
	- Code corpus (The Stack v2 / StarCoder2 permissive subsets) for structured reasoning signal

Wave 2 — low-risk architecture and inference wins (easy wins first):
- ☐ **rope_base standardization**: set milestone configs from 13892 to 500000 (Llama-3 style)
- ☐ **YaRN RoPE scaling** for 2× context extension at inference; validate coherence at 2048 tokens
- ☐ **KV-cache path** for autoregressive decoding, including MLA latent-cache variant (cache latent `z`)
- ☐ Prompt templates (ChatML or Alpaca-style) and `ChatFormatter` for multi-turn inference

Wave 3 — training stack upgrades (high impact):
- ☐ **Z-loss** in `compute_loss_with_smoothing()` for logit-scale stabilization (`1e-4 * log(sum(exp(logits)))^2`)
- ☐ **Muon optimizer** for 2-D weight matrices; keep AdamW for embeddings/head/bias/1-D params; run 5K-step Muon vs AdamW comparison
- ☐ **μP (Maximal Update Parameterization)**:
	- Update init + per-layer LR scaling for width transfer
	- Run LR sweep on 6L/256H proxy (~5M params)
	- Transfer LR directly to 12L/1024H and validate transfer error vs direct sweep

Wave 4 — data products for post-training (depends on Waves 0-1):
- ☐ **SFT data**: curate 1-5M instruction-response pairs; produce domain subsets for continual learning
- ☐ **Grounding data**: curate 50K-500K examples (math/logic/world-models/games/causal chains)
- ☐ Grounding loader with structured input -> explanation -> answer format
- ☐ **Alignment data**: assemble 10K-100K preference pairs (+5-10% harmful), label 5K-10K reward targets, optionally test synthetic preferences

Wave 5 — post-training components and training loops:
- ☐ **Fine-tuning**: LoRA on Q/K/V/O projections, base frozen, merge adapters for deployment
- ☐ Loss masking for assistant tokens (SFT) and answer tokens (grounding)
- ☐ **Alignment stack**: reward model, calibration, DPO or PPO/GRPO, β and LR knobs
- ☐ Continual-learning loop: replay buffer (10%), online adaptation, forgetting measurement

Wave 6 — evaluation and release gate:
- ☐ Held-out eval set + side-by-side generation comparisons
- ☐ Benchmark harness integration (`lm-eval-harness` or equivalent) with 2-3 standard benchmarks (for example HellaSwag and MMLU subset)
- ☐ Track forgetting metric: Δ = max(0, perf_before - perf_after)
- ☐ Track alignment diagnostics: reward margin, preference accuracy, KL, reward confidence, reward correlation
- ☐ Safety + quality checks: win rate vs base, adversarial safety behavior, reward-model MSE, continual-learning degradation

Architecture extensions (defer until core Phase 5 stack is stable):
- ☐ **Mixture of Depths (MoD)** token router for variable-compute inference
- ☐ Interleaved SWA + full attention ablation against pure MLA baseline

**Exit Criteria**:
- ☐ Corpus hygiene: WikiText-103 removed; FineWeb/FineWeb-Edu dedup resolved; OWT quality filters + min_length + language filter active
- ☐ Near-dedup: MinHash LSH run on OWT + FineWeb; dedup rate + updated stats.json committed
- ☐ Sequence packing: throughput gain vs non-packed baseline measured and documented
- ☐ Tokenizer decision: 32K trained + validated, or explicit defer with note
- ☐ Muon + Z-loss first run: curve vs AdamW baseline documented
- ☐ μP proxy sweep: LR transfer from 5M proxy to 12L/1024H validated
- ☐ KV-cache: > 2× generation speedup at seq_len 512
- ☐ YaRN: coherent output at 2048 tokens (2× training length)
- ☐ LoRA SFT: better instruction-following than base; < 1% trainable parameters
- ☐ Grounding: ppl improves on held-out grounding examples vs pre-grounding baseline
- ☐ DPO/GRPO: > 60% preference accuracy on held-out pairs; reward margin trend positive
- ☐ Safety eval: refusal rate > 80% on adversarial suite; false-refusal < 10% on benign prompts

---

### Phase 6: MoE

**Goal**: Sparse MoE on the Phase 5 stack with continual expert specialization. (MLA already in Phase 4.)

**Dependencies**: Phase 5 baseline available for dense-vs-sparse comparison.
**Kill Criteria**: Stop if token drop > 5% or expert collapse persists beyond 3 mitigation attempts.

**Exit Criteria**:
- ☐ MoE val ppl ≤ dense baseline at same FLOPs
- ☐ Expert utilization balanced (5–30% per expert, 8 experts); load-balance loss converges
- ☐ Token drop < 1% during training and inference
- ☐ Dense-vs-sparse comparison report committed

---

### Phase 7: Dual-Stream Reasoning

**Goal**: GRU reasoning stream parallel to transformer, gated fusion combiner, STaR bootstrap, inference feedback loop.

**Dependencies**: Phase 6 sparse architecture stabilized.
**Kill Criteria**: Stop if reasoning model fails to beat GRU-zeroed baseline on GSM8K, or if inference overhead > 30%.

**Exit Criteria**:
- ☐ Reasoning-enabled model > 10% accuracy improvement over Phase 6 baseline on GSM8K
- ☐ GRU-zeroed model matches Phase 6 baseline (graceful degradation verified)
- ☐ GRU inference overhead < 20% vs transformer-only at seq_len 512
- ☐ STaR bootstrap completed; scheduled teacher forcing curve documented

---

## Canonical References

- [MEMORY.md](../.github/MEMORY.md) (current agent working state and focus) + [SESSION_LOG.md](../.github/SESSION_LOG.md) (recent completed-session history) + [SESSION_LOG_ARCHIVE.md](../.github/SESSION_LOG_ARCHIVE.md) (older history)
- [DESIGN.md](DESIGN.md) (architecture, engineering standards, testing strategy, agent workflow)
- [PLAN.md](PLAN.md) (this file — phased execution roadmap)
- [CONTRIBUTING.md](../CONTRIBUTING.md#standard-workflow) (workflow and validation gates)
- [config/*.toml](../config) (authoritative runtime values)
- [Makefile](../Makefile) (build, test, lint targets)

