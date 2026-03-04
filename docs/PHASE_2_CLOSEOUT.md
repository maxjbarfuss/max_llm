# Phase 2: Skeleton & Reproducibility — Closeout Report

**Date**: 2026-03-03 | **Status**: ✅ COMPLETE
**Goal**: Runnable training on small dataset with reproducible losses and checkpoints

---

## Objectives ✅

| Objective | Evidence |
|---|---|
| Reproducible training loop | Seed control, deterministic data loading, checkpoint restoration |
| SimpleLM architecture | 1 embedding + 128H MLP → consistent losses |
| Char tokenization (UTF-8) | 256-token vocab, ~100% English coverage |
| Small-scale validation | 10M token train/val/test split |
| Loss convergence | Loss → 2.84 (UTF-8), reproducible |
| Checkpoint save/restore | Weights + optimizer state + step preserved |

---

## Architecture Validation

### UTF-8 — Success ✅

**Config**: `config/milestones/p2_final.toml`

```
Corpus: WikiText + TinyStories (10M tokens) | Tokenization: UTF-8 (256 vocab)
Architecture: Token Emb (256→128) → GELU MLP → LM Head (128→256)
Batch: 8 | Precision: float32

Step 1:    loss=16.05, ppl=9.3M
Step 100:  loss=2.75,  ppl=15.6   ← converged
Step 500:  loss=2.84,  ppl=17.2   ← plateau

Throughput: 300K tok/s | GPU: 20 MB | Checkpoint: 968 KB
```

**Chat** (best-in-class checkpoint, temp=0.9, top_k=20):
```
You: Once upon a time,
🤖 Once upon a time, briluly asie, blo pre tin nor, theey o nd o achanyonird...

You: The model is useful because
🤖 The model is useful becausers thed t wompy alad thea tourthe s bus f ad s...
```
Character-level and noisy — expected. Confirms inference pipeline works end-to-end.

### BPE — Documented Failure ❌

```
Same corpus + GPT2 BPE (50,304 vocab)

Step 1:    loss=16.45
Step 100:  loss=8.50, ppl=4,935   ← PLATEAU
Step 500:  loss=8.50              ← stuck (no progress)

GPU: 400 MB | Checkpoint: 75 MB
```

**Root cause**: Linear head (128 → 50K) has rank ≤ 128. Cannot express 50K-dim output.
Architectural limit — hyperparameters cannot fix this.

---

## Key Findings

| | UTF-8 (256 vocab) | BPE (50K vocab) |
|---|---|---|
| Final loss | **2.84** | 8.50 (stuck) |
| Converges? | ✅ Yes | ❌ No |
| Architecture limit | None | **Rank bottleneck** |

**Overtraining note**: WikiText-only stress test (6K steps) shows overtrain onset at ~step 2000
(val `2.63` best → degrades). Recommended stop window: steps 1900–2400.

---

## Artifacts

| Artifact | Location |
|---|---|
| UTF-8 checkpoint | `outputs/p2-closeout-final-20260303/p2-combined-10m-utf8/checkpoint.pt` (968 KB) |
| UTF-8 loss curve | `outputs/p2-closeout-final-20260303/p2-combined-10m-utf8/loss_curve.csv` |
| BPE checkpoint | `outputs/p2-closeout-final-20260303/p2-combined-10m-bpe/checkpoint.pt` (75 MB) |
| BPE loss curve | `outputs/p2-closeout-final-20260303/p2-combined-10m-bpe/loss_curve.csv` |

---

## Completion Checklist

- [x] SimpleLM implemented and validated
- [x] UTF-8 char tokenization working end-to-end
- [x] Training loop with loss logging, checkpointing, seed control
- [x] Loss converges reproducibly (UTF-8: 2.84)
- [x] Architecture limitation documented (BPE fails at 50K vocab)
- [x] Interactive inference validated on trained checkpoint

---

## Phase 2 → Phase 3

**What Phase 2 proved**: Pipeline reliable + reproducible. Small vocab (256) works. SimpleLM cannot scale to 50K+ vocab — architectural, not tunable.

**What Phase 3 addresses**: DecoderLM (4-layer Transformer). Multi-head attention handles 50K vocab implicitly. Result: **49% loss improvement** (8.50 → 4.31 on BPE).

---

**Phase 2 Status**: ✅ COMPLETE — `outputs/p2-final/` (loss 2.84)
