# Max LLM Execution Plan

Purpose: phased execution roadmap for human contributors and AI agents. Keep experiment evidence in closeout docs; keep this file focused on goals, status, and next work.

**How to use:**
1. Read [MEMORY.md](../.github/MEMORY.md) first for current active work.
2. Use this file for phase boundaries, execution order, and unfinished work.
3. Use closeouts for evidence and learning notes: [Phase 1](PHASE_1_CLOSEOUT.md), [Phase 2](PHASE_2_CLOSEOUT.md), [Phase 3](PHASE_3_CLOSEOUT.md), [Phase 4](PHASE_4_CLOSEOUT.md), [Phase 5](PHASE_5_CLOSEOUT.md).
4. For architecture details, see [DESIGN.md](DESIGN.md); for optimization notes, see [OPTIMIZATION.md](OPTIMIZATION.md).

---

## Phase Progress

| Phase | Status | Focus | Data Strategy | Key Artifacts |
|-------|--------|-------|---------------|---------------|
| **1** | Done | Foundation | Setup; no training data | CI workflow, test scaffold, env notes. [Phase 1 Closeout](PHASE_1_CLOSEOUT.md) |
| **2** | Done | Skeleton & Reproducibility | TinyStories + WikiText-103 | Tokenizer, data pipeline, training loop, checkpoints, deterministic replay. [Phase 2 Closeout](PHASE_2_CLOSEOUT.md) |
| **3** | Done | Capable GPT-2-like model | Mixed corpus + Unigram 8K tokenizer | Coherent decoder baseline; p3_final_unigram and p3_final_27b_merge50. [Phase 3 Closeout](PHASE_3_CLOSEOUT.md) |
| **4** | Done | Llama Architecture + Scale-Up | Wikipedia -> Cosmopedia-v2 -> mixed curriculum | Final checkpoint `p4_final_anneal_20260422`, val_loss `2.323`, ppl ~`9.2`. [Phase 4 Closeout](PHASE_4_CLOSEOUT.md) |
| **5** | In progress | Post-Training | SFT, grounding, preference data; 32K-tokenizer stack comparison | P4 32K baseline, LoRA adapters, grounding benchmark, reward-model card, safety eval. [Phase 5 Notes](PHASE_5_CLOSEOUT.md) |
| **6** | Not started | MoE + MLA | Partitioned SFT + preference curriculum | MoE routing diagnostics, dense-vs-sparse comparison |
| **7** | Not started | Dual-Stream Reasoning | Reasoning trace triples + STaR | Dual-stream comparison, reasoning accuracy delta, GRU overhead benchmark |

Artifact naming: use `p<phase>_<artifact>_<yyyymmdd>_<commit>_<seed>` for reports, checkpoints, benchmark CSVs, and promoted configs.

---

## Completed Phases

### Phase 1: Foundation

**Goal**: Reproducible environment, CI, testing baseline, and dependency/tooling foundation.
**Historical detail**: [PHASE_1_CLOSEOUT.md](PHASE_1_CLOSEOUT.md)

**Exit Criteria Met**:
- CI baseline stable and green.
- Build/test/lint targets operational through `make`.
- Python + C++ test discovery/reporting validated.
- Acceleration stack dependency checks completed.

### Phase 2: Skeleton & Reproducibility

**Goal**: Runnable small-scale training with deterministic replay and checkpoint integrity.
**Historical detail**: [PHASE_2_CLOSEOUT.md](PHASE_2_CLOSEOUT.md)

**Exit Criteria Met**:
- End-to-end training converged on milestone config.
- Seeded replay and checkpoint restore behavior verified.
- Overfit/inference sanity gates passed.
- Phase 2 milestone artifacts promoted and documented.

### Phase 3: Capable GPT-2-like Model

**Goal**: Coherent-output decoder model with stable long-run training stack.
**Historical detail**: [PHASE_3_CLOSEOUT.md](PHASE_3_CLOSEOUT.md)

**Exit Criteria Met**:
- Coherent output gate passed on milestone checkpoint.
- Optimization stack stability validated: Flash, bf16, fused AdamW, WSD, DDP.
- Phase 3 milestone runs completed and promoted.
- Phase 3 closeout captures tokenizer/model/data arc and evidence.

### Phase 4: Llama Architecture + Scale-Up Training

**Goal**: Llama-class architecture, curriculum training, and final anneal outperforming Phase 3.
**Historical detail**: [PHASE_4_CLOSEOUT.md](PHASE_4_CLOSEOUT.md)

**Exit Criteria Met**:
- 5-stage curriculum completed and consolidated.
- P3 baseline surpassed.
- Final anneal checkpoint promoted with validated metrics.
- Methodology findings documented in closeout and optimization learnings.

---

## Phase 5: Post-Training

**Goal**: Prepare for SFT, grounding, preference optimization, and continual-learning evaluation without destabilizing the Phase 4 base model. Also includes Phase 5 training-stack upgrades: Muon, Z-loss, packed training, YaRN/KV-cache, looped blocks, interleaved attention, MoD, and the norm-preserving generalization filter.

**Dependencies**: Phase 4 architecture frozen with reproducible checkpoints and selected eval baselines.
**Artifacts**: SFT dataset manifest, LoRA adapter bundle, grounding benchmark report, preference dataset card, reward-model calibration report, alignment logs, safety evaluation summary, continual-learning report, Muon/AdamW comparison, μP proxy sweep, P4 32K-tokenizer baseline comparison.
**Evidence**: [PHASE_5_CLOSEOUT.md](PHASE_5_CLOSEOUT.md) holds compact result tables and current learnings.
**Kill Criteria**: Stop if forgetting metric Δ worsens for 3 consecutive evaluations, or if alignment causes >20% degradation on base capabilities.
**Out of Scope**: MoE promotion belongs to Phase 6; dual-stream reasoning belongs to Phase 7.

### Current Baseline

Carry forward the no-MoD P4 mixed-corpus recipe for the next 32K-tokenizer comparison:

- 12L/1024H, `looped_num_blocks = 4`
- Interleaved `swa/mla/swa/mla`
- `attn_res_fused`, bf16, fused AdamW, torch compile
- Generalization filter: interval `4`, validation batches `1`, damping `0.5`, preserve norm

MoD is implemented but not promoted; tune router/capacity separately because validation lagged the no-MoD baseline.

### Phase 5 Work Plan

| Wave | Status | Work | Useful Detail |
|------|--------|------|---------------|
| 0 | Done | Data gates | WikiText-103 removed; OWT/FineWeb-Edu policy set; OWT hygiene, `min_length=120`, language filter, MinHash near-dedup, sequence packing, repetition budgets, and Cosmopedia provenance are in place. |
| 1 | Done | 32K tokenizer | 32K Unigram tokenizer trained on 215K docs / ~287M tokens from OWT, FineWeb-Edu, Wikipedia, Cosmopedia, Gutenberg, `ir_python`, `owm`, and NuminaMath. |
| 1 | Next | 32K-tokenizer baseline | Run current no-MoD + norm-preserving gen-filter P4 recipe on `/mnt/d/Dev/data/prepared/p5_wave1_tokenizer_32k_20260424/`; compare against the 8K-tokenized P4 mixed-corpus curve. |
| 1 | Not done | Production source preps | Generate and validate 32K-tokenized Gutenberg, `ir_python`, `owm`, and NuminaMath-CoT outputs. Check prose/code/math quality, dedup/overlap, repetition budgets, formatting, and split hygiene before adding them to training mixes. |
| 2 | Done | Low-risk inference/architecture | RoPE base standardized, YaRN implemented, KV-cache implemented and hardened across MHA/MLA/residual paths. |
| 2 | Not done | Prompt templates / `ChatFormatter` | Add shared formatting for training, inference, and eval. Support ChatML-style roles plus an Alpaca-style fallback; produce assistant-token loss masks and preserve system/user/assistant boundaries. |
| 3 | Done | Training stack upgrades | Z-loss, Muon, chunked LM loss, memory-efficient xIELU, FFN chunking, checkpointing modes, packed masks, varlen Flash, looped blocks, interleaved SWA/MLA, MoD implementation, and generalization filter are implemented. |
| 3 | Not done | Throughput tuning | Profile dataloader/forward/backward/optimizer/eval cadence; tune sequence length, microbatch, grad accumulation, backend, compile mode, checkpointing, precision, and optimizer before long Phase 5 runs. |
| 3 | Not done | FSDP / larger-param memory path | Revisit full-shard or per-block wrapping after the dense 32K baseline; current model-level `shard_grad_op` lowers memory slightly but does not raise activation-limited microbatch ceiling. |
| 3 | Not done | μP | Add width-transfer init/LR scaling, sweep a 6L/256H proxy, transfer LR to 12L/1024H, and promote only if it reduces sweep cost without hurting the baseline curve. |
| 4 | Not done | SFT data | Curate 1-5M instruction-response pairs with domain subsets, held-out instruction eval, and source/provenance tracking. |
| 4 | Not done | Grounding data + loader | Curate 50K-500K math/logic/world-model/game/causal examples; implement structured input -> explanation -> answer loading with answer-token masks and held-out grounding eval. |
| 4 | Not done | Alignment data | Assemble 10K-100K preference pairs, including 5-10% harmful/adversarial; label 5K-10K reward targets and validate any synthetic preferences against a reviewed subset. |
| 5 | Not done | Fine-tuning loop | LoRA on Q/K/V/O with base frozen; compare adapter vs merged inference quality/speed; verify SFT loss masks exclude system/user tokens. |
| 5 | Not done | Alignment stack | Reward model, calibration, DPO or PPO/GRPO, β/LR knobs, reward-margin and KL guardrails before alignment training. |
| 5 | Not done | Continual learning | Replay buffer (~10%), online adaptation loop, forgetting metric, and rollback criteria for degradation. |
| 6 | Not done | Evaluation/release gate | Held-out eval set, side-by-side generations, benchmark harness, forgetting metric, alignment diagnostics, safety checks, win rate, false-refusal tracking. |

### Phase 5 Exit Gates

- 32K-tokenizer baseline measured against the current P4 recipe.
- Production Wave 1 corpora prepared and validated.
- Prompt formatter shared by training, inference, and eval.
- μP proxy sweep either validated or explicitly deferred.
- LoRA SFT improves instruction-following with <1% trainable parameters.
- Grounding improves held-out grounding loss/quality vs pre-grounding baseline.
- Preference optimization reaches >60% held-out preference accuracy with positive reward-margin trend.
- Safety eval passes: >80% adversarial refusal and <10% benign false refusal.

---

## Phase 6: MoE

**Goal**: Sparse MoE on the stabilized Phase 5 stack with continual expert specialization. MLA is already in the dense stack.

**Dependencies**: Phase 5 dense 32K-tokenizer baseline available for dense-vs-sparse comparison.
**Kill Criteria**: Stop if token drop >5% or expert collapse persists beyond 3 mitigation attempts.

**Planned work**:
- Add sparse expert layers with load-balancing loss and expert-utilization logging.
- Keep MoE routing separate from Phase 5 MoD token-depth routing until both are understood.
- Compare sparse-vs-dense quality at similar compute using the Phase 5 dense baseline as control.
- Track token drop, expert collapse, throughput, memory, and base-capability degradation.

**Exit Criteria**:
- MoE val ppl <= dense baseline at same FLOPs.
- Expert utilization balanced, roughly 5-30% per expert for 8 experts.
- Token drop <1% during training and inference.
- Dense-vs-sparse comparison report committed.

---

## Phase 7: Dual-Stream Reasoning

**Goal**: GRU reasoning stream parallel to transformer, gated fusion combiner, STaR bootstrap, and inference feedback loop.

**Dependencies**: Phase 6 sparse architecture stabilized.
**Kill Criteria**: Stop if reasoning model fails to beat GRU-zeroed baseline on GSM8K, or if inference overhead >30%.

**Planned work**:
- Add GRU reasoning stream and gated fusion without changing transformer-only fallback.
- Build reasoning trace triples and STaR-style bootstrap loop.
- Compare reasoning-enabled model against transformer-only and GRU-zeroed baselines.
- Measure reasoning quality, runtime overhead, teacher-forcing schedule behavior, and graceful degradation.

**Exit Criteria**:
- Reasoning-enabled model >10% accuracy improvement over Phase 6 baseline on GSM8K.
- GRU-zeroed model matches Phase 6 baseline.
- GRU inference overhead <20% vs transformer-only at seq_len 512.
- STaR bootstrap completed; scheduled teacher-forcing curve documented.

---

## Canonical References

- [MEMORY.md](../.github/MEMORY.md): current agent working state and focus
- [SESSION_LOG.md](../.github/SESSION_LOG.md): recent completed-session history
- [SESSION_LOG_ARCHIVE.md](../.github/SESSION_LOG_ARCHIVE.md): older completed-session history
- [DESIGN.md](DESIGN.md): architecture, engineering standards, testing strategy, agent workflow
- [OPTIMIZATION.md](OPTIMIZATION.md): optimization notes
- [PHASE_1_CLOSEOUT.md](PHASE_1_CLOSEOUT.md), [PHASE_2_CLOSEOUT.md](PHASE_2_CLOSEOUT.md), [PHASE_3_CLOSEOUT.md](PHASE_3_CLOSEOUT.md), [PHASE_4_CLOSEOUT.md](PHASE_4_CLOSEOUT.md), [PHASE_5_CLOSEOUT.md](PHASE_5_CLOSEOUT.md): phase summaries and learning notes
- [CONTRIBUTING.md](../CONTRIBUTING.md#standard-workflow): workflow and validation gates
- [config/*.toml](../config): authoritative runtime values
- [Makefile](../Makefile): build, test, lint targets
