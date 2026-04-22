# Phase 2: Skeleton & Reproducibility — Closeout Report

**Date**: 2026-03-07 | **Status**: ✅ COMPLETE
**Goal**: Runnable training on small dataset with reproducible losses and checkpoints

---

## Objectives ✅

| Objective | Evidence |
|---|---|
| Reproducible training loop | Seed control, deterministic data loading, checkpoint restoration |
| SimpleLM architecture | Embedding-only model → consistent losses |
| Char tokenization (ASCII-127) | 128-token vocab, 100% English coverage |
| Large-scale validation | 100M token training corpus |
| Loss convergence | Loss → 2.92 at step 2000, reproducible |
| Checkpoint save/restore | Weights + optimizer state + step preserved |

---

## Architecture Validation

### ASCII-127 — Production Baseline ✅

**Config**: `config/milestones/p2_ascii127.toml`

```
Corpus: TinyStories (100M tokens) | Tokenization: ASCII-127 char-level
Architecture: Token Emb (128→1024) + Pos Emb → LM Head (1024→128, weight-tied)
Batch: 128 × 4 grad_accum (effective 512) | Precision: bf16

Step 1:    loss=4.85,  ppl=127.6
Step 500:  loss=3.27,  ppl=26.5
Step 1000: loss=3.06,  ppl=21.2
Step 2000: loss=2.92,  ppl=18.6   ← still improving (not plateaued)

Throughput: 13.5M tok/s | GPU: 1.4 GB | Checkpoint: 14 MB
```

**Optimizer**: `lr=0.015, weight_decay=0.0, warmup=100, cosine_schedule`

**Key Result**: Loss continues dropping at step 2000 — model has capacity to learn further.
Embedding-only architecture (num_layers=0) demonstrates reproducible training pipeline.

---

## Lessons Learned

Detailed project-wide lessons now live in [LESSONS.md](../.github/LESSONS.md).

Phase 2-specific takeaways are captured there and in the validation evidence above; this closeout keeps the historical record focused on outcomes, metrics, and artifacts.

---

## Artifacts

| Artifact | Location |
|---|---|
| ASCII-127 checkpoint | `outputs/milestones/p2-ascii127-2000/checkpoint.pt` (14 MB) |
| ASCII-127 loss curve | `outputs/milestones/p2-ascii127-2000/loss_curve.csv` |
| ASCII-127 config | `config/milestones/p2_ascii127.toml` |

---

## Completion Checklist

- [x] SimpleLM implemented and validated
- [x] ASCII-127 char tokenization working end-to-end
- [x] Training loop with loss logging, checkpointing, seed control
- [x] Loss converges reproducibly (ASCII-127: 2.92 @ step 2000)
- [x] Hyperparameter optimization: lr, batch_size, hidden_size tuned
- [x] Data scaling validated (100M tokens)
- [x] Training optimizations: bf16, grad_accum, rapid iteration
- [x] Interactive inference pipeline validated

**Phase 2 Status**: ✅ COMPLETE — `outputs/milestones/p2-ascii127-2000/` (loss 2.92)
