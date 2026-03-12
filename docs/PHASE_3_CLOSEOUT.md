# Phase 3: DecoderLM & Stability — Closeout Report

**Date**: 2026-03-12 | **Status**: 🔄 BPE validation complete — Unigram training active (~4500/12000 steps, val loss ~3.59, ppl ~36)
**Goal**: Validate DecoderLM on BPE tokenization, prove stability with full optimization stack; achieve coherent output via Unigram 8K tokenizer on clean mixed corpus

---

## Objectives ✅

| Objective | Evidence |
|---|---|
| DecoderLM architecture | 4L Transformer (256H, 4 heads) — GQA-ready |
| BPE tokenization | GPT2, 50,304 vocab, full coverage |
| Large vocab scaling | Loss 4.31 vs SimpleLM failure (8.50) — 49% improvement |
| Multi-backend attention | Flash Attention 2 integrated and validated |
| BF16 mixed precision | Stable 15K+ steps, zero NaN/Inf |
| Gradient accumulation | Effective batch 32 (8×4) |
| Early stopping | Patience-based validation monitoring |
| Label smoothing | ε=0.1, prevents overconfident predictions |
| torch.compile | AOT ready, 10–20% expected speedup |
| Reproducibility | Identical seeds → identical trajectories |

---

## Architecture Validation: First Working → Best-of-Breed

Two experiments bracket Phase 3. Chat sessions use matched prompts for direct comparison.

### Experiment 1 — First Working Run (Feb 27)

**Checkpoint**: `outputs/p3-decoder-lm-test/checkpoint.pt` (step 500)

```
Corpus: WikiText BPE (442K tokens) | Architecture: 2L / 128H / standard attention
LR: 0.001 | Batch: 4 | Steps: 500

Step 500:  loss=6.72,  ppl=831       ← proof of concept; not converged
```

**Chat** (temp=0.9, top_k=20, max_tokens=80):
```
You: Once upon a time,
🤖 Once upon a time. the he of the the the the the the the the the a a " first the...

You: The model learned to
🤖 The model learned to " " He the the the a a the the " the he a an a the a " to...
```

Degenerate repetition — model learned word frequency but lacks capacity for coherent phrases.

### Experiment 2 — Best-of-Breed Run (Feb 28 – Mar 2)

**Checkpoint**: `outputs/p3-bpe-convergence/checkpoint.pt` (step 5000)
**Config**: `config/milestones/p3_baseline.toml`

```
Corpus: WikiText + TinyStories interleaved (2.15M train / 375K val tokens)
Architecture: 4L / 256H / Flash Attention 2
LR: 0.0003 cosine (200-step warmup) | Batch: 8 | Grad accum: 2 → eff. batch 16
BF16 | Dropout: 0.1 | Steps: 5000

Step 500:  loss=4.41           (already below Exp. 1 final at step 500)
Throughput: 165K tok/s (Flash) | GPU: 1566 MB
```

**Chat** (same prompts, same sampling):
```
You: Once upon a time,
🤖 Once upon a time there was a little girl named Lily. She loved to play in the park
   with her friends. One day, she saw a big dog barking at a cat. She was scared and
   ran away. The dog was barking and barking. " Stop it! " she said.

You: The model learned to
🤖 The model learned to misogyny. They were best friends.
```

Real English words and sentence structure. Hallucinations remain — expected at 5K steps.

| | Exp. 1 (First Working) | Exp. 2 (Best-of-Breed) |
|---|---|---|
| Architecture | 2L / 128H | 4L / 256H |
| Dataset | 442K BPE tokens | 2.15M BPE tokens |
| Attention | Standard | Flash Attention 2 |
| Steps | 500 | 5000 |
| Final loss | 6.72 | **4.31** |
| Output quality | Degenerate repetition | Coherent English + narrative fragments |

**Key driver**: data scale (442K → 2.15M) + model capacity (128H/2L → 256H/4L).
Flash Attention enables the throughput (165K tok/s) that makes 5K steps practical.

### Additional Validation Runs

| Run | Config | Result | Purpose |
|---|---|---|---|
| Baseline | `config/milestones/p3_baseline.toml` | 2K steps, 4L/256H | Canonical P3 starting point (Flash, BF16, grad accum) |
| Full-stack safety | `config/milestones/p3_combined_convergence.toml` | loss 7.79 at 15K steps | torch.compile + label smoothing ε=0.1 + early stopping |

Full-stack run at 15,937+ steps (eff. batch 32, torch.compile, label smoothing ε=0.1,
early stopping patience=5) confirms all safety infrastructure works.

---

## Technology Stack

| Feature | Status | Key Result |
|---|---|---|
| Flash Attention 2 | ✅ | Throughput 50K → 165K tok/s |
| BF16 mixed precision | ✅ | Stable 15K+ steps, no NaN/Inf |
| Gradient accumulation | ✅ | Eff. batch up to 120 (12×10), smooth convergence |
| Early stopping | ✅ | Patience counter working (triggers at plateau) |
| Label smoothing ε=0.1 | ✅ | Smoother loss curves, no sudden jumps |
| Selective weight decay | ✅ | Bias/LayerNorm excluded (ndim < 2); works with DDP |
| torch.compile | ✅ | AOT compilation ready; ~126K tok/s measured |
| Reproducibility | ✅ | Identical seeds → identical trajectories |
| AdamW fused | ✅ | Fused CUDA kernel, β=(0.9, 0.95); replaces Adam (P3) |
| WSD scheduler | ✅ | Warmup-Stable-Decay with sqrt/linear/lowered-linear shapes; supports run continuation (P3) |
| Fused QKV | ✅ | Single `nn.Linear(d, 3d)` + `.chunk(3)`; no Q/K/V bias; fewer kernel launches (P3.8) |
| Scaled residual init | ✅ | `out_proj` + FFN `linear2`: `N(0, 0.02/√(2L))` GPT-2 style; prevents variance explosion at depth (P3.8) |
| Gradient norm logging | ✅ | `grad_norm` in CSV + TensorBoard; enables stability monitoring (P3.8) |
| Chunked CE loss | ✅ | Iterates (B·T, V) in 4096-token chunks; saves ~800 MB at B=24, T=2048, V=8192 (P3.9) |
| FSDP | ✅ | Model sharding integrated alongside DDP; advanced from P4 (P3.9) |
| Random sequence offset | ✅ | `TokenDataset.set_epoch(epoch)` varies sequence boundaries each epoch (P3) |
| NFKC/unk filtering | ✅ | Unicode normalization + unknown token filtering in data prep; cleaner corpus (P3.9) |
| DistributedSampler | ✅ | Proper data sharding across DDP ranks; `set_epoch` called per epoch (P3) |

**Root cause of Phase 2 BPE failure**: SimpleLM head is rank-128 into 50K-dim vocab — structurally
impossible. DecoderLM's multi-head attention produces implicit high-rank representations.

| Architecture | Vocab | Final Loss | Result |
|---|---|---|---|
| SimpleLM (P2) | 256 | 2.84 | Works |
| SimpleLM (P2) | 50K | 8.50 | Fails (rank bottleneck) |
| DecoderLM (P3) | 50K | **4.31** | ✅ 49% better than SimpleLM |

**Conclusion**: Architecture, not hyperparameters, determines vocabulary scalability.

---

## Unigram 8K Training Run (Phase 3 Final)

**Config**: `config/ephemeral/p3_final.toml` | **Status**: 🔄 In progress

```
Architecture : 8–10L / 1024H / 16H / 2048 ctx — ~105M params
Tokenizer    : Unigram 8K (SentencePiece) with EOS per doc, NFKC normalization
Corpus       : TinyStories (~10%) + WikiText-103 (full) + OpenWebText (~12%) + FineWeb-Edu (partial)
Optimizer    : AdamW fused, β=(0.9, 0.95), lr=4e-3, wd=0.05
Scheduler    : WSD (warmup=500, stable=55%, decay=35%, sqrt shape)
Grad clip    : 0.5 | Batch: 12 × 10 accum = eff. 120 | Precision: BF16
Max steps    : 12,000 | Currently: ~4500 (stable phase, decay not yet started)

Step 4500 : val_loss=3.59, ppl=36   — still in stable plateau, not yet decaying
```

**Tokenizer decision**: Unigram 8K supersedes BPE — reaches val loss 6.85 at step 80 vs BPE's 7.92 at same step.
Unigram encodes denser semantic units for Wikipedia/news/story mix, leading to faster early convergence.

---

## Artifacts

| Artifact | Location |
|---|---|
| BPE BoB checkpoint | `outputs/p3-bpe-convergence/checkpoint.pt` (193 MB) |
| BPE BoB loss curve | `outputs/p3-bpe-convergence/loss_curve.csv` (5K steps) |
| First-working checkpoint | `outputs/p3-decoder-lm-test/checkpoint.pt` (79 MB) |
| Unigram active run | `outputs/ephemeral/p3-unigram-h16-8layers-wsd-safe-20260311/` |
| Unigram loss curve | `outputs/ephemeral/p3-unigram-h16-8layers-wsd-safe-20260311/loss_curve.csv` |

---

## Completion Checklist

- [x] DecoderLM 4L/256H implemented and validated (BPE)
- [x] BPE tokenization end-to-end (50K vocab)
- [x] 49% improvement over SimpleLM proven (4.31 vs 8.50)
- [x] Full optimization stack validated (Flash, BF16, grad accum, early stop, label smooth)
- [x] Extended optimization stack: AdamW fused, WSD scheduler, fused QKV, scaled residual init, chunked CE loss, FSDP, NFKC filtering, grad norm logging
- [x] torch.compile integration ready
- [x] Reproducibility confirmed
- [x] BPE BoB checkpoint chat-tested: generates coherent English words and narrative fragments
- [x] Dataset rebuilt with EOS tokens + NFKC normalization (Unigram 8K)
- [ ] Unigram training run to convergence (step ~4500/12000 — in progress)
- [ ] **GATE**: Unigram checkpoint chat-tested: coherent output at 8K vocab scale

---

## Phase 3 → Phase 4

**What Phase 3 proved**: DecoderLM scales to 50K+ vocab. Full optimization stack
(Flash/BF16/AdamW-fused/WSD/fused-QKV/scaled-residual-init/chunked-CE/FSDP) is production-ready.
Convergence dynamics smooth; dataset with proper EOS + NFKC filtering trains stably.

**What Phase 4 adds**: Llama-style upgrades — RMSNorm, RoPE, SwiGLU FFN, GQA — and
scale to 300M+ parameters with FSDP (infrastructure already validated in P3).

**Phase 3 Status**: 🔄 BPE validation complete (loss 4.31, ppl 74.5); Unigram run in progress (~4500/12000 steps, val ppl ~36) — awaiting coherent output gate
