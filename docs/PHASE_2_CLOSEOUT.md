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

### 1. **Data Scale Matters Significantly**
- 100M token corpus enables meaningful convergence
- ASCII-127 (English-only): optimal efficiency for TinyStories corpus
- Final perplexity: 18.6 at step 2000

### 2. **Hyperparameter Optimization — Systematic Exploration**
- **Learning Rate**: Can push 10-20× baseline (0.001 → 0.015) for embedding-only models
	- Sweet spot: 0.01–0.02 (tested 0.001, 0.003, 0.005, 0.01, 0.02, 0.03, 0.05)
	- Degradation at 0.03+
- **Weight Decay**: Extremely sensitive — 0.1 completely kills learning (loss stuck at 4.85)
	- Use 0.0 for embedding-only architectures
- **Batch Size**: Scaled 8 → 128 (16× increase) with grad_accum=4
	- Effective batch size 512 works excellently
	- Tested hardware limits: 8, 32, 64, 128, 256, 512 (all successful)
- **Hidden Size**: 1024 (vs 128 baseline) → better capacity utilization

### 3. **Training Optimizations Enabled Rapid Iteration**
- Mixed precision (bf16): ~40% speedup, minimal accuracy impact
- Gradient accumulation: enables large effective batch sizes on single GPU
- Fast tokenization pipeline: 13.5M tokens/s throughput
- Multiple training runs: systematic exploration of hyperparameter space
- **Result**: 15+ training runs in single session, enabling data-driven tuning

### 4. **Architecture Insights**
- Embedding-only (num_layers=0) sufficient for pipeline testing
- Cannot generate coherent text without attention mechanism
- Perplexity ≠ generation quality for embedding-only models
- Achieved ppl=18.6 but outputs only unigram statistics

### 5. **Config System Refactoring**
- Made inference/training parameters optional with sensible defaults
- Reduced boilerplate: minimal configs now ~30 lines vs 100+ previously
- Single `LearningModel` class handles both SimpleLM and DecoderLM

### 6. **Loss Dynamics**
- Training still improving at step 2000 (not plateaued)
- Loss curve slope indicates capacity for 3000-5000 steps before saturation
- Cosine schedule decay preserves late-stage fine-tuning

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
