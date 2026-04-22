# Phase 4: Llama Architecture + Scale-Up Training — Closeout Report

**Date**: 2026-04-22
**Goal**: Llama-style architecture on a 5-stage curriculum (Wikipedia → Cosmopedia-v2 → FineWeb-10BT → OWT → mixed 27B), outperform P3 baseline, then anneal on clean structured text to consolidate.

---

## Objectives ✅

| Objective | Evidence |
|---|---|
| Llama architecture (RMSNorm/RoPE/SwiGLU/GQA) | Implemented + validated |
| MLA attention + block_attn residuals + xIELU FFN | Best-class stack confirmed in run |
| FlashNorm (parameter-free norm) | `norm_type="flash"` used in final config |
| 5-stage curriculum training | All 5 stages complete |
| P3 baseline beaten | P3 ppl 24.0 → P4 ppl ~14.3 (mixed final) |
| DDP 10K+ step validated | Multiple runs; no divergence |
| Simplicity anneal (Wikipedia + Cosmopedia-v2) | 15K steps SGDR; ppl ~9.2 |
| Repetition penalty + sampling tuning | Implemented + calibrated |

---

## Architecture

Best-class stack (`config/milestones/p4_final_anneal_20260422.toml`): FlashNorm, RoPE, MLA, xIELU FFN, block-attn residuals, 12 layers, 1024 hidden, 16 heads / 4 KV heads, 1024 context, Unigram 8K tokenizer. Model size is ~60M parameters, which preserves direct ppl comparison with Phase 3.

---

## Training Arc

### 5-Stage Curriculum

Stages 3 and 4 were ended early and carried forward deliberately as warm-starts into the next stage.

| Stage | Run | Corpus | Steps | Val Loss | Notes |
|---|---|---|---|---|---|
| 1 | `p4_wiki_curriculum_s1_wsd_20260410` | Wikipedia EN | 28,500 | 2.352 | WSD |
| 2 | `p4_cosmopedia_curriculum_s2_wsd_20260412` | Cosmopedia-v2 | 26,600 | 1.975 | WSD |
| 3 | `p4_fw10bt_curriculum_s3_wsd_20260415` | FineWeb-10BT | 21,000 | 2.923 | WSD, ended early |
| 4 | `p4_owt_curriculum_s3b_wsd_20260416` | OpenWebText | 18,000 | 3.083 | WSD, ended early |
| 5 | `p4_final_mixed_27b_wsd_20260416` | Mixed 27B | 75,000 | **2.660** | WSD, **final mixed checkpoint** |

Stage 1→2 reflects factual consolidation; stages 3–4 show expected domain-transition bumps; stage 5 consolidates the curriculum and sets the best pre-anneal checkpoint.

### Simplicity Anneal

- Corpus: Wikipedia EN (7.4B) + fresh Cosmopedia-v2 (3.98B, seed=31415, ~77% unseen) → 11.4B tokens.
- Schedule: SGDR, 3 cycles, LR `5e-5 → 2.5e-6`, 15K steps, 200 warmup steps.
- Batch: `10 × 16 accum × 1024 ctx × 2 GPU` = 327,680 tokens/step.
- Result: step 15K reached `val_loss=2.323`, ppl ~9.2.

Why it worked: low-LR re-exposure to clean structured text consolidated factual and narrative behavior without erasing the broader curriculum.

---

## P3 vs P4 Comparison

| Checkpoint | Val Loss | Val PPL | Steps | Notes |
|---|---|---|---|---|
| P3 Run 1 (p3_final_unigram) | ~3.178 | 24.0 | 12K | Original mixed corpus |
| P3 Final (p3_final_27b_merge50) | ~3.364 | 28.9 | 10,836 | 27B corpus, merged init |
| P4 Final mixed (stage 5) | 2.660 | ~14.3 | 75K | P4 Llama best-class stack |
| **P4 Anneal (final)** | **2.323** | **~9.2** | 15K post-stage-5 | Wikipedia+Cosmopedia anneal |

P4 vs P3 best: **ppl 9.2 vs 24.0 — 2.6× improvement**.

Primary drivers: Llama architecture (MLA+xIELU+block_attn+FlashNorm), larger corpus (27B vs ~10B effective), and simplicity annealing on clean text.

---

## External Benchmark Snapshot (P4 Final Checkpoint)

Internal harness run (`python -m src.eval.run`) on 128-example validation slices:

| Task | P4 final (~60M, ours) | TinyLlama 1.1B | SmolLM2 1.7B (base) |
|---|---|---|---|
| HellaSwag | **32.8%** | 60.3% | 68.7% |
| PIQA | **66.4%** | — | 77.6% |
| ARC-Easy | **35.9%** | 33.9%* | 60.5%** |

Comparison readout: the Phase 4 checkpoint is materially behind strong 1B–2B public small-model baselines on HellaSwag and PIQA, with ARC performance depending on benchmark variant.

\* TinyLlama reports ARC (Challenge), not ARC-Easy.

\** SmolLM2 reports ARC average in the model card table, not ARC-Easy specifically.

---

## Methodology Findings

**P4-DEC-1: RMSNorm confirmed as Phase 4 default** — LayerNorm vs RMSNorm A/B at 6L/1024H/2K steps showed RMSNorm lower peak memory with similar speed. FlashNorm (parameter-free variant) used in final config.

**P4-DEC-2: Multi-axis perturbation escapes plateaus** — When val loss plateaued, a coordinated weight soup (averaging step_8K + step_6K checkpoints) + data seed reset + SGDR scheduler restart produced immediate improvement. Simultaneous perturbation of weight space, data order, and LR trajectory escapes local basins more effectively than any single change.

**P4-DEC-3: Domain transfer causes coherence regression despite PPL improvement** — FW-Edu-only training improved FW-Edu PPL but degraded factual recall and structural diversity vs the mixed checkpoint. PPL on a domain-matched val set is a misleading signal for general quality. Solution: factual curriculum anchor (Wikipedia → Cosmopedia) before mixing, and domain-diverse val set.

---

## Artifacts

| Artifact | Location |
|---|---|
| **Milestone config** | `config/milestones/p4_final_anneal_20260422.toml` |
| **Final checkpoint** | `outputs/milestones/p4_final_anneal_20260422/checkpoint.pt` |
| **Loss curve** | `outputs/milestones/p4_final_anneal_20260422/loss_curve.csv` |
| Stage 5 (mixed 27B) checkpoint | `outputs/ephemeral/p4_final_mixed_27b_wsd_20260416/checkpoint.pt` |

---

## Completion Checklist

- [x] Llama-style architecture implemented: RMSNorm/FlashNorm, RoPE, xIELU FFN, GQA, MLA attention
- [x] block_attn residuals, make_norm/make_ffn/make_attention factories
- [x] All attention backends (Flash/xFormers/Sage/Standard) working; CPU fallback to standard
- [x] 5-stage curriculum completed: Wikipedia → Cosmopedia-v2 → FineWeb-10BT → OWT → Mixed-27B
- [x] DDP validated: 10K+ steps, 2 GPUs, no divergence
- [x] P3 vs P4 comparison: ppl 9.2 vs 24.0 — **P4 wins by 2.6×**
- [x] Repetition-penalty sampling calibrated for coherent inference on the final checkpoint
- [x] Simplicity anneal: 15K SGDR steps on Wikipedia+fresh-Cosmopedia, val_loss 2.323, ppl ~9.2
- [x] Inference configuration captured in milestone config and sampler implementation
- [x] Milestone config + checkpoint promoted to `config/milestones/` + `outputs/milestones/`

---

## Phase 4 → Phase 5

**What Phase 4 proved**: Llama-class architecture (MLA + xIELU + block_attn + FlashNorm) at 60M params on a clean 5-stage curriculum reaches ppl ~9.2 after annealing — a 2.6× improvement over P3. Curriculum structure matters: Wikipedia/Cosmopedia factual anchoring before mixing prevents domain-transfer coherence regression (P4-DEC-3). Simplicity annealing at the tail is effective and cheap.

**What Phase 5 adds**: SFT (LoRA adapters), grounding on math/logic/games, DPO or GRPO preference alignment, continual-learning evaluation, and a set of efficiency and optimizer improvements identified in the Phase 4 architecture review. The anneal checkpoint is the starting point.

---

## Architecture Review: Gaps and Forward Priorities

Notable strengths:
- MLA + xIELU + block_attn + FlashNorm is an unusual combined stack.
- P4-DEC-2 established a practical plateau-escape protocol.
- The anneal used an explicitly measured fresh-data fraction (~77% unseen), not a vague clean-data pass.

### Techniques Not Yet Incorporated (Prioritized)

| Technique | Phase | Effort | Payoff | Notes |
|---|---|---|---|---|
| **YaRN RoPE scaling** | Now | Trivial | Context extension w/o retraining | Current `rope_base=13892` is undocumented; YaRN or `rope_base=500000` fixes this |
| **Z-loss regularization** | Next run | Trivial | Stability + small PPL improvement | `1e-4 × log(Σexp(logits))²` appended to CE loss; from PaLM/Gemini training |
| **Muon optimizer** | Next run | Low | ~10–30% sample efficiency vs AdamW | Replace AdamW on 2-D weight matrices only; keep AdamW for embeddings/biases |
| **Sequence packing** | Data pipeline | Medium | ~20–40% throughput on short-doc corpora | Pack multiple docs per context window with block-diagonal causal mask |
| **Mixture of Depths (MoD)** | P5 fine-tune | Medium | Variable-compute inference efficiency | Per-layer top-k token router; can be fine-tuned onto frozen base |
| **μP (Maximal Update Parameterization)** | P5 training | Medium | HP transfer across model sizes | Run LR sweep on 5M proxy; transfer to 300M+ without re-sweep |
| **Interleaved SWA + full attention** | Next run | Low | Long-context memory efficiency | Already implemented in `src/models/attention/sliding_window_attention.py` |

### rope_base Audit

`rope_base=13892` in the final anneal config has no documented rationale (likely a warm-start artifact near the default 10000). Recommend `rope_base=500000` (Llama 3 style) for the next training run, or apply YaRN scaling from the current checkpoint for immediate context extension.

---

## Data Review: Gaps and Forward Priorities

### Actual Phase 4 Corpus Composition

| Source | Docs | Tokens | % Tokens | Median doc (tokens) |
|---|---|---|---|---|
| OpenWebText | 79.6M | 8.87B | 33% | **84** |
| FineWeb | 7.5M | 6.72B | 25% | 537 |
| Cosmopedia v2 | 6.2M | 5.37B | 20% | 796 |
| FineWeb-Edu | 2.3M | 2.68B | **10%** | 717 |
| Wikipedia | 1.7M | 1.88B | 7% | 532 |
| WikiText-103 | 7.4M | 1.07B | **4%** | 135 |
| TinyStories | 2.7M | 0.13B | 0.5% | **48** |
| stories_young_children | 0.28M | 0.13B | 0.5% | 472 |

### What's Working

- NFKC + control-char stripping.
- Two-stage UNK filtering.
- Stratified per-source validation split.
- Spill-cache tokenization and resume behavior.
- Factual-first curriculum ordering.
- Fresh-subset anneal selection (~77% unseen Cosmopedia docs).

### Structural Problems Found

- **FineWeb-Edu duplicates FineWeb**: the corpus currently double-exposes educational pages.
- **WikiText-103 duplicates Wikipedia**: it adds short noisy fragments already covered by the full dump.
- **OWT wastes context**: median length is 84 tokens, so packing and stronger filtering are high leverage.

### Data Gaps Not Yet Addressed (Prioritized)

| Gap | Effort | Impact | When |
|---|---|---|---|
| Remove WikiText-103 (fully covered by Wikipedia) | Trivial | Med | Next data run |
| Deduplicate FineWeb ↔ FineWeb-Edu (Edu is a subset) | Low | Med-High | Next data run |
| Heuristic OWT quality filters (punctuation rate, line repetition, symbol ratio) | Low | High | Next data run |
| Language detection filter — fastText on OWT/FineWeb | Low | Med | Next data run |
| Raise OWT min_length to ~100–150 tokens | Trivial | Med | Next data run |
| MinHash LSH near-dedup within+across OWT/FineWeb | High | High | Phase 5 pre-train |
| Sequence packing (already planned for P5) | Med | High | Phase 5 |
| New 32K Unigram tokenizer on Phase 4 distribution | Med | High (Phase 5+) | Phase 5 decision |
| Add long-form books source (Project Gutenberg) | Low | Med | Phase 5+ |
| Add code source (The Stack v2) | Low | Med | Phase 5+ |
| Per-source repetition budget tracking | Low | Med | Next data run |
| Document Cosmopedia synthetic provenance | Trivial | Low | Now |

### Cosmopedia Note

Cosmopedia v2 (20% of the corpus) is synthetic text generated by HuggingFace with Mistral. That provenance should be recorded anywhere this corpus is described publicly.

### Tokenizer Mismatch

The Unigram 8K tokenizer was trained on the older Phase 3 distribution and missed Cosmopedia, which became 20% of the Phase 4 corpus. A 32K tokenizer trained on the full cleaned Phase 4 distribution is the highest-leverage data investment before more pretraining.
