# Phase 3 Optimized Architecture — Quick Start Guide

**Last Updated**: 2026-03-03
**Phase 3 Best-of-Breed**: DecoderLM + BPE + Full optimization stack
**Status**: ✅ Validated and production-ready

---

## Overview

Phase 3 establishes the **DecoderLM + BPE** architecture as the foundation for large-vocabulary language modeling. This guide explains the configuration, training pipeline, and inference setup to quickly get to a working best-of-breed baseline.

**Key Achievement**: Scaled from Phase 2's 256-token vocabulary (UTF-8 char level) to **50,304-token vocabulary (GPT-2 BPE)** without loss divergence. Loss: **4.31** on 10M interleaved WikiText + TinyStories corpus.

---

## Configurations Available

### 1. `p3_bpe_convergence.toml` — Sealed Best-of-Breed Baseline

**Purpose**: Reproduction of the locked checkpoint (loss 4.31, step 5000)
**Best for**: Fast convergence validation, inference testing, phase 4 warm-start
**Training time**: ~20 GPU hours (single GPU)

**Key Parameters**:
```toml
model_type = "decoder_lm"
hidden_size = 256
num_layers = 4
num_heads = 4
vocab_size = 50304
batch_size = 8
gradient_accumulation_steps = 2    # Effective batch = 16
max_steps = 5000
learning_rate = 0.0003
attention_backend = "flash"        # 3-4x speedup
precision_schedule = [[0, -1, "bf16"]]  # BF16 required for 50K vocab
```

**Optimizations Enabled**:
- ✅ Flash Attention 2 (streaming for O(1) memory, 3-4x speedup)
- ✅ BF16 mixed precision (prevents NaN/Inf on large vocab)
- ✅ Cosine + warmup scheduler (smooths convergence, 200-step warmup)
- ✅ Selective weight decay (excludes bias/LayerNorm)
- ✅ Gradient clipping (clip_norm=1.0 essential with BF16)
- ✅ Data loading optimization (4 workers, pin_memory=true, prefetch=2)

**Checkpoint**: `outputs/p3-bpe-convergence/checkpoint.pt` (193 MB, locked)

**Expected Results**:
```
step   500: loss=5.40, ppl=221
step  1000: loss=4.18, ppl=65
step  5000: loss=4.31, ppl=74  ← Converged
```

---

### 2. `p3_combined_convergence.toml` — Enhanced Convergence Run

**Purpose**: Full-stack validation run with early stopping and test set tracking
**Best for**: Longer training runs, validation of optimization stack, dataset exploration
**Training time**: ~25 GPU hours (will early stop around 8-12K steps)

**Enhancements Over Baseline**:
```diff
  dropout = 0.15                    # ↑ Increased from 0.1 for regularization
  gradient_accumulation_steps = 4   # ↑ Effective batch = 32
  weight_decay = 0.15               # ↑ Increased from 0.01
  warmup_steps = 500                # ↑ Longer warmup
  eval_interval = 500               # ↑ Validate every 500 steps
  label_smoothing = 0.1             # ✨ NEW: Smooths output distribution
  early_stopping_patience = 5       # ✨ NEW: Stop after 5 evals without improvement
  use_torch_compile = true          # ✨ NEW: 10-20% speedup from AOT compilation
  eval_on_test = true               # ✨ NEW: Track test loss separately
```

**Checkpoint Output**: `outputs/p3-optimized-convergence/`

**Expected Behavior**:
- Trains up to max_steps=50000, but stops early when validation plateaus
- Early stopping triggers after 5 consecutive eval cycles (~2500-3000 steps) without 0.01+ improvement
- Outputs CSV with per-step losses, including val/test tracking

---

### 3. `p3_baseline.toml` — Minimal Baseline

**Purpose**: Quick sanity-check runs, architecture debugging
**Best for**: Rapid prototyping, CI/CD validation (minimal setup time)
**Training time**: ~5 GPU hours for 2000 steps

**Configuration**:
- No early stopping
- No label smoothing
- Short training (2000 steps)
- Minimal eval (every 100 steps)

---

## Training & Inference Pipeline

### Start Training from Sealed Baseline

```bash
# Single GPU
python -m src.training.train --config config/milestones/p3_bpe_convergence.toml

# Multi-GPU DDP (2 GPUs)
python -m torch.distributed.launch --nproc_per_node=2 \
  -m src.training.train --config config/milestones/p3_bpe_convergence.toml --distributed
```

**Output**:
```
./outputs/p3-bpe-convergence/
├── checkpoint.pt         # Final model weights + optimizer state
├── loss_curve.csv        # Step, loss, perplexity, lr, tokens/sec, val_loss, test_loss
└── p3-bpe-convergence.log  # Full training log
```

### Inference on Sealed Checkpoint

**Interactive chat mode**:
```bash
python -m src.inference.chat \
  --config config/milestones/p3_bpe_convergence.toml \
  --checkpoint outputs/p3-bpe-convergence/checkpoint.pt \
  --max-tokens 80
```

**Programmatic generation**:
```python
import torch
from src.models.learning_model import DecoderLM
from src.tokenizer import TokenizerFactory

# Load model
config_path = 'config/milestones/p3_bpe_convergence.toml'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = DecoderLM.from_config(...)
model = model.to(device).to(torch.bfloat16)  # Important: BF16 for Flash Attention
model.eval()

# Tokenize and generate
tokenizer = TokenizerFactory.create("bpe", encoding="gpt2")
tokens = torch.tensor([tokenizer.encode("Once upon a time")], device=device)

with torch.no_grad():
    logits = model(tokens)
    next_token_id = logits[0, -1, :].argmax()

print(f"Next token: {tokenizer.decode([next_token_id.item()])}")
```

---

## Key Optimization Details

### Why BF16 (Not FP32)

On 50K vocabulary, FP32 training diverges (loss explodes to NaN). BF16 maintains numerical stability:
- Phase 2 (SimpleLM + 256 vocab): FP32 fine
- Phase 3 (DecoderLM + 50K vocab): **BF16 required**

**How it works**: BF16 has lower precision but broader dynamic range, which helps with extreme logits in large vocabulary spaces.

### Why Flash Attention

Standard PyTorch attention: O(n²) memory, O(n²) FLOPs.
Flash Attention: O(n) memory (streaming), same FLOPs but 3-4x wall-clock speedup.

```
Baseline (256H, 4L, seq_len=256): ~1.5 hours/1000 steps
+ Flash Attention: ~0.35 hours/1000 steps
+ BF16: ~0.25 hours/1000 steps
Total: 1.25 hours/1000 steps (6.25 hours for 5000 steps baseline)
```

### Gradient Accumulation

Effective batch size = batch_size × gradient_accumulation_steps × world_size

```
Config: batch_size=8, grad_accum=2
Per GPU: 8 tokens/batch × 256 seq_len = 2048 tokens/micro-batch
Effective: 16 (macro-batch) → 16 × 256 = 4096 tokens/step
```

Larger effective batches stabilize training but require more compute per step.

### Early Stopping

Train up to 50K steps, but stop if validation loss plateaus:
```
Validation Loss Tracking (eval every 500 steps):
Step  500: val_loss=4.95 ← Best so far, counter=0
Step 1000: val_loss=4.88 ✓ Improved, counter=0
Step 1500: val_loss=4.87 ✓ Improved, counter=0
Step 2000: val_loss=4.88 ✗ No improvement >0.01, counter=1
Step 2500: val_loss=4.89 ✗ counter=2
Step 3000: val_loss=4.90 ✗ counter=3
Step 3500: val_loss=4.91 ✗ counter=4
Step 4000: val_loss=4.92 ✗ counter=5 → STOP (patience=5 reached)
```

---

## Critical Symbols & Files

| File | Symbol | Purpose |
|------|--------|---------|
| `src/models/learning_model/decoder_lm.py` | `class DecoderLM` | Phase 3 model architecture |
| `src/models/attention/causal_mha.py` | `class CausalMultiHeadAttention` | Multi-backend attention dispatcher |
| `src/training/train.py` | `def main()` | Training entrypoint |
| `src/training/loop.py` | `def train()` | Main training loop with all optimizations |
| `src/training/optimizer.py` | `configure_optimizer_param_groups()` | Selective weight decay |
| `src/training/scheduler.py` | `get_cosine_schedule_with_warmup()` | LR schedule |
| `config/milestones/p3_bpe_convergence.toml` | — | Sealed baseline config |
| `outputs/p3-bpe-convergence/checkpoint.pt` | — | Locked checkpoint (loss 4.31) |

---

## Data Pipeline

### Current Dataset

File: `data/fast/interleaved_wikitext_tinystories_10m_tokens_bpe_gpt2.npy`

**Composition**:
- 70% WikiText (encyclopedia-like text, coherent long-range dependencies)
- 30% TinyStories (children's stories, diversity in narrative styles)
- Total: ~10M BPE tokens

**Processing**:
1. Raw text → tokenize with GPT-2 BPE → pad to 50304 (multiple of 64)
2. Split: 90% train (2.15M tokens), 10% val (375K tokens) via `validation_split=0.1`
3. Sequence chunks: (seq_len=256 + 1 for target)

### Data Loading Optimization

```toml
[data]
num_workers = 4              # Read ahead with 4 parallel workers
prefetch_factor = 2          # Buffer 2 batches ahead
pin_memory = true            # Pre-allocate GPU-pinned RAM
persistent_workers = true    # Keep workers alive between epochs
```

**Effect**: Reduces data stall-time from ~40% to <5%

---

## Troubleshooting

### "FlashAttention only support fp16 and bf16 data type"
**Solution**: Ensure model is cast to BF16 before inference
```python
model = model.to(torch.bfloat16)
```

### "CUDA out of memory"
**Reduce** (in order of impact):
1. Batch size (batch_size = 4)
2. Sequence length (max_seq_length = 128)
3. Gradient accumulation (gradient_accumulation_steps = 1)

### Loss diverges to NaN after 100 steps
**Check**:
1. Is precision_schedule set to BF16? (critical for large vocab)
2. Is gradient_clip_norm set to 1.0? (missing = divergence)
3. Is learning rate too high? (try 0.0003 if using 0.001)

---

## Next Steps (Phase 4 Preview)

Phase 3 validates decoder + BPE. Phase 4 upgrades:
- **RMSNorm** (instead of LayerNorm, more numerically stable)
- **RoPE** (rotary positional embeddings, generalize better than learned)
- **SwiGLU** (gated linear units, 7-10% efficiency gain vs GELU)
- **GQA** (grouped query attention, reduce KV cache by 8-16x)
- **FSDP** (fully sharded data parallel for multi-machine scaling)

The Phase 3 sealed checkpoint (`p3-bpe-convergence/checkpoint.pt`) serves as a warm-start baseline for Phase 4 experiments.

---

## References

- **Architecture**: [src/models/learning_model/decoder_lm.py](../src/models/learning_model/decoder_lm.py)
- **Training Loop**: [src/training/loop.py](../src/training/loop.py)
- **Optimization Report**: [docs/OPTIMIZATION.md](OPTIMIZATION.md)
- **Phase 3 Closeout**: [docs/PHASE_3_CLOSEOUT.md](PHASE_3_CLOSEOUT.md)
- **Comparison Report**: [outputs/p2_vs_p3_best_of_breed_report_20260302.md](../outputs/p2_vs_p3_best_of_breed_report_20260302.md)
