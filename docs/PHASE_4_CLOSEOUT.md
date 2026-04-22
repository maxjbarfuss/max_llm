# Phase 4: Llama Architecture + Scale-Up Training — Closeout Report

**Date**: 2026-04-22 | **Status**: ✅ Complete — 5-stage curriculum + simplicity anneal done; final ppl ~9.2
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

**Model config** (best-class stack — `config/milestones/p4_final_anneal_20260422.toml`):

```toml
hidden_size        = 1024
num_heads          = 16
num_kv_heads       = 4          # GQA (4:1 ratio)
vocab_size         = 8192       # Unigram 8K (same tokenizer as P3)
max_seq_length     = 1024
num_layers         = 12
norm_type          = "flash"    # FlashNorm (parameter-free RMSNorm)
ffn_type           = "xielu"    # piecewise quadratic/exp activations
intermediate_size  = 4096
attn_type          = "mla"      # Multi-head Latent Attention (DeepSeek-V2 style)
mla_latent_dim     = 512
res_type           = "block_attn"  # block-summary attention residuals
attn_res_num_blocks = 6
pos_type           = "rope"
rope_base          = 13892
dropout            = 0.0
```

~60M parameters. Same Unigram 8K tokenizer as P3 (enables direct ppl comparison).

---

## Training Arc

### 5-Stage Curriculum

Stages 3 and 4 were ended early; their final checkpoints were carried forward as warm-starts into the next stage (not failures — deliberate curriculum continuation).

| Stage | Run | Corpus | Steps | Val Loss | Notes |
|---|---|---|---|---|---|
| 1 | `p4_wiki_curriculum_s1_wsd_20260410` | Wikipedia EN | 28,500 | 2.352 | WSD |
| 2 | `p4_cosmopedia_curriculum_s2_wsd_20260412` | Cosmopedia-v2 | 26,600 | 1.975 | WSD |
| 3 | `p4_fw10bt_curriculum_s3_wsd_20260415` | FineWeb-10BT | 21,000 | 2.923 | WSD, ended early |
| 4 | `p4_owt_curriculum_s3b_wsd_20260416` | OpenWebText | 18,000 | 3.083 | WSD, ended early |
| 5 | `p4_final_mixed_27b_wsd_20260416` | Mixed 27B | 75,000 | **2.660** | WSD, **final mixed checkpoint** |

Stage 1→2 val loss drops clearly (2.352→1.975), reflecting the model gaining denser factual knowledge from Cosmopedia. Stage 3–4 upward bumps are expected domain-transition artifacts when val set doesn't match training distribution; the mixed-27B stage (5) consolidates all prior knowledge and achieves the final curriculum best.

### Simplicity Anneal

**Config**: `config/milestones/p4_final_anneal_20260422.toml`
**Resume from**: `p4_final_mixed_27b_wsd_20260416/checkpoint.pt`

```
Corpus   : Wikipedia EN (7.4B tokens) + fresh Cosmopedia-v2 (3.98B, seed=31415, ~77% unseen)
           → blended 11.4B token dataset
Schedule : SGDR, 3 cycles (sgdr_cycle_decay=0.8), LR=5e-5→2.5e-6 per cycle
Steps    : 15,000 | Warmup: 200 | min_lr_ratio=0.05
Batch    : 10 × 16 accum × 1024 ctx × 2 GPU DDP = 327,680 tokens/step

Step  1000 : ppl ~10.0  (resume point after power outages)
Step 15000 : val_loss=2.323, ppl ~9.2   ← final
```

**Why anneal**: Llama 3 and MiniCPM both show that re-exposing a near-final checkpoint to clean, structured text at low LR consolidates factual and narrative patterns without catastrophic forgetting. Cosmopedia seed=31415 gives ~77% fresh documents vs the prior 7B-token batch — maximises new signal on familiar distribution.

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

## Methodology Findings

**P4-DEC-1: RMSNorm confirmed as Phase 4 default** — LayerNorm vs RMSNorm A/B at 6L/1024H/2K steps showed RMSNorm lower peak memory with similar speed. FlashNorm (parameter-free variant) used in final config.

**P4-DEC-2: Multi-axis perturbation escapes plateaus** — When val loss plateaued, a coordinated weight soup (averaging step_8K + step_6K checkpoints) + data seed reset + SGDR scheduler restart produced immediate improvement. Simultaneous perturbation of weight space, data order, and LR trajectory escapes local basins more effectively than any single change.

**P4-DEC-3: Domain transfer causes coherence regression despite PPL improvement** — FW-Edu-only training improved FW-Edu PPL but degraded factual recall and structural diversity vs the mixed checkpoint. PPL on a domain-matched val set is a misleading signal for general quality. Solution: factual curriculum anchor (Wikipedia → Cosmopedia) before mixing, and domain-diverse val set.

---

## Sampling Configuration

Repetition penalty implemented in `src/inference/sampler.py` (added this phase). Calibrated inference settings for the anneal checkpoint:

```toml
temperature        = 0.25
top_p              = 0.5
top_k              = 33
repetition_penalty = 1.15
repetition_window  = 256
```

Low temperature (near-greedy) + tight top_k=33 keeps output coherent. Wide repetition_window=256 suppresses medium-range loops that small-vocab models are prone to.

---

## Artifacts

| Artifact | Location |
|---|---|
| **Milestone config** | `config/milestones/p4_final_anneal_20260422.toml` |
| **Final checkpoint** | `outputs/milestones/p4_final_anneal_20260422/checkpoint.pt` |
| **Loss curve** | `outputs/milestones/p4_final_anneal_20260422/loss_curve.csv` |
| Stage 5 (mixed 27B) checkpoint | `outputs/ephemeral/p4_final_mixed_27b_wsd_20260416/checkpoint.pt` |
| Stage 5 training status | `outputs/ephemeral/p4_final_mixed_27b_wsd_20260416/training_status.json` |
| Anneal training status | `outputs/ephemeral/p4_wiki_cosmo_anneal_sgdr_20260421/training_status.json` |

---

## Completion Checklist

- [x] Llama-style architecture implemented: RMSNorm/FlashNorm, RoPE, xIELU FFN, GQA, MLA attention
- [x] block_attn residuals, make_norm/make_ffn/make_attention factories
- [x] All attention backends (Flash/xFormers/Sage/Standard) working; CPU fallback to standard
- [x] 5-stage curriculum completed: Wikipedia → Cosmopedia-v2 → FineWeb-10BT → OWT → Mixed-27B
- [x] DDP validated: 10K+ steps, 2 GPUs, no divergence
- [x] P3 vs P4 comparison: ppl 9.2 vs 24.0 — **P4 wins by 2.6×**
- [x] Repetition penalty implemented and wired through sampler + config + chat + run
- [x] Simplicity anneal: 15K SGDR steps on Wikipedia+fresh-Cosmopedia, val_loss 2.323, ppl ~9.2
- [x] Inference sampling calibrated: coherent output at `temp=0.25, top_k=33, rep_penalty=1.15, rep_window=256`
- [x] Milestone config + checkpoint promoted to `config/milestones/` + `outputs/milestones/`

---

## Phase 4 → Phase 5

**What Phase 4 proved**: Llama-class architecture (MLA + xIELU + block_attn + FlashNorm) at 60M params on a clean 5-stage curriculum reaches ppl ~9.2 after annealing — a 2.6× improvement over P3. Curriculum structure matters: Wikipedia/Cosmopedia factual anchoring before mixing prevents domain-transfer coherence regression (P4-DEC-3). Simplicity annealing at the tail is effective and cheap.

**What Phase 5 adds**: SFT (LoRA adapters), grounding on math/logic/games, DPO or GRPO preference alignment, and continual-learning evaluation. The anneal checkpoint is the starting point.

**Phase 4 Status**: ✅ Complete — final val ppl ~9.2 (anneal); P4 > P3 by 2.6×; coherent output confirmed
