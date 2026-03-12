# Training Optimization Guide

**Purpose**: Maximize training throughput and minimize memory usage on dual-GPU hardware (RTX 4090 + RTX 3090 Ti, 48GB VRAM).

## Quick Start

```toml
[training]
attention_backend = "flash"           # flash|sage|xformers|standard
use_torch_compile = false            # incompatible with flash/sage
batch_size = 12
gradient_accumulation_steps = 10     # effective batch = 12 × 10 = 120
scheduler_type = "wsd"               # cosine|wsd
wsd_stable_fraction = 0.55
wsd_decay_fraction = 0.35
wsd_decay_shape = "sqrt"             # linear|sqrt|lowered_linear
precision_schedule = [[0, -1, "bf16"]]
use_distributed = true               # enables DDP/FSDP via torchrun
distributed_backend = "ddp"          # ddp|fsdp

[data]
num_workers = 4                      # 8 for dual-GPU
prefetch_factor = 4
pin_memory = true
persistent_workers = false           # false for single-pass LLM training
```

**Single-GPU**: `python -m src.training.train --config config/your_config.toml`
**Multi-GPU**: `torchrun --standalone --nnodes=1 --nproc_per_node=2 -m src.training.train --config config/your_config.toml --distributed`

## Attention Backends

Choose one based on your constraints:

| Backend | Speed | Memory | torch.compile | Use When |
|---------|-------|--------|---------------|----------|
| **flash** | 2-4x | Best | ❌ No | Default for training (recommended) |
| **sage** | 2-4x | Best | ❌ No | A/B test vs Flash Attention |
| **xformers** | 1.5-2x | Good | ✅ Yes | Need torch.compile compatibility |
| **standard** | 1x | Baseline | ✅ Yes | Debugging, CPU, fallback |

**Critical constraint**: `flash` and `sage` cannot use `torch.compile`. Choose one path:
- **Path A**: `attention_backend = "flash"` + `use_torch_compile = false` (recommended)
- **Path B**: `attention_backend = "xformers"` + `use_torch_compile = true`

### Installation

```bash
pip install flash-attn --no-build-isolation  # Flash Attention 2
pip install sageattention                    # Sage Attention (optional)
pip install xformers                         # xFormers (already installed)
```

All backends auto-fallback to `standard` if unavailable. Training never fails on missing dependencies.

### Benchmark Backends

```bash
python scripts/benchmark_attention.py \
  --d-model 256 \
  --num-heads 8 \
  --seq-len 256 \
  --batch-size 8 \
  --backends flash,xformers,standard
```

## Multi-GPU Training (DDP / FSDP)

**Status**: ✅ DDP implemented and production-tested with 2 GPUs; FSDP integrated (P3.9)

Enables near-linear scaling via data parallelism. Expected: ~1.8x throughput (30-35% sync overhead).
Use FSDP (`distributed_backend = "fsdp"`) for models too large to fit per-GPU with DDP.

### Usage

```bash
# DDP (default, recommended for P3 model sizes)
source .venv/bin/activate
torchrun --standalone --nnodes=1 --nproc_per_node=2 \
    -m src.training.train \
    --config config/your_config.toml \
    --distributed

# FSDP (for larger models — set distributed_backend = "fsdp" in config)
torchrun --standalone --nnodes=1 --nproc_per_node=2 \
    -m src.training.train \
    --config config/your_config.toml \
    --distributed
```

### Config Adjustments

```toml
[training]
use_distributed = true
distributed_backend = "ddp"  # or "fsdp"

[data]
num_workers = 8  # 4 per GPU (2 × 4)
```

### How It Works

1. `torchrun` spawns one process per GPU
2. Each process gets unique rank (0, 1, ..., N-1)
3. Model replicated to each GPU
4. Data automatically sharded across processes
5. Gradients synchronized after backward pass
6. Only rank 0 saves checkpoints

### Expected Performance

- **Single GPU** (RTX 4090): ~150-170k tokens/sec
- **Dual GPU** (4090 + 3090 Ti): ~270-300k tokens/sec total (~1.8x)

### Troubleshooting

**"Address already in use"**: Change port with `export MASTER_PORT=29501`

**Training hangs at init**: Ensure both GPUs are idle (`nvidia-smi`)

**Loss divergence**: Verify same seed across all processes

**OOM**: Reduce `batch_size` or increase `gradient_accumulation_steps`

## DataLoader Optimization

**Purpose**: Eliminate CPU data loading bottleneck by parallelizing with GPU training.

**Status**: ✅ Production-ready

```toml
[data]
num_workers = 4              # Start with 4, increase if GPU util < 95%
prefetch_factor = 2          # Batches to prefetch per worker
pin_memory = true            # Faster CPU→GPU transfer (CUDA only)
persistent_workers = false   # true for multi-epoch, false for single-pass
```

**Impact**: ~1.2-1.5x throughput, prevents training hangs

**Guidelines**:
- Single-GPU: `num_workers = 4`
- Dual-GPU: `num_workers = 8`(4 per GPU)
- Reduce if OOM or high CPU usage

## torch.compile

**Purpose**: Kernel fusion and graph optimization for 30-40% speedup.

**Status**: ✅ Implemented (xformers/standard only)

```toml
[training]
attention_backend = "xformers"  # or "standard"
use_torch_compile = true
```

**Constraints**:
- ❌ **Cannot** be used with `flash` or `sage` attention
- Adds ~10-30s compilation overhead on first step
- Requires PyTorch 2.x

## Learning Rate Schedulers

Two schedules are supported via `scheduler_type`.

### Cosine (default)

```toml
[training]
scheduler_type = "cosine"
warmup_steps = 200
min_lr_ratio = 0.1   # decay floor = 10% of peak lr
```

Linear warmup → cosine decay to `min_lr_ratio * lr`.

### WSD (Warmup-Stable-Decay) — recommended for long runs

```toml
[training]
scheduler_type = "wsd"
warmup_steps = 500
wsd_stable_fraction = 0.55   # 55% of steps at peak lr
wsd_decay_fraction = 0.35    # 35% of steps decaying
wsd_decay_shape = "sqrt"     # linear | sqrt | lowered_linear
min_lr_ratio = 0.05
```

WSD holds LR at peak during the stable phase, then decays. This makes long runs more predictable and allows resuming from a checkpoint mid-decay by adjusting `wsd_stable_fraction`. Decay shapes:
- `sqrt`: fast initial drop, slower tail (recommended)
- `linear`: uniform decay
- `lowered_linear`: stays high longer, drops sharply at end (`wsd_lowered_linear_alpha` controls shape)

## Gradient Accumulation

**Purpose**: Simulate larger batch sizes without OOM.

```toml
[training]
batch_size = 8                      # Micro-batch per step
gradient_accumulation_steps = 4     # Effective batch = 8 × 4 = 32
```

Accumulates gradients across multiple forward/backward passes before optimizer step.

## Mixed Precision (AMP)

**Purpose**: Reduce memory usage and increase throughput via bf16 computation.

**Status**: ✅ Automatic when precision_schedule specified

```toml
[training]
precision_schedule = [[0, -1, "bf16"]]  # bf16 from step 0 to end
```

**Expected**: ~1.5-2x memory reduction, ~1.2-1.4x throughput (requires Ampere+ GPU for bf16)

## Recommended Configurations

### Phase 3: Production Run (~100M params, 2-GPU DDP)
```toml
[training]
attention_backend = "flash"
use_torch_compile = false
use_distributed = true
distributed_backend = "ddp"
batch_size = 12
gradient_accumulation_steps = 10   # effective batch = 120
scheduler_type = "wsd"
wsd_stable_fraction = 0.55
wsd_decay_fraction = 0.35
wsd_decay_shape = "sqrt"
learning_rate = 0.004
betas = [0.9, 0.95]
weight_decay = 0.05
gradient_clip_norm = 0.5
precision_schedule = [[0, -1, "bf16"]]

[data]
num_workers = 4
prefetch_factor = 4
pin_memory = true
persistent_workers = false
```
**Expected**: ~58K tok/s per GPU × 2 GPUs = ~116K tok/s effective (RTX 4090 + RTX 3090Ti, 1024H, 8-10L, seq_len=2048)

### Phase 3: Small-Model Debug Run (single GPU, fast iteration)
```toml
[training]
attention_backend = "flash"
use_torch_compile = false
batch_size = 8
gradient_accumulation_steps = 2
scheduler_type = "cosine"
precision_schedule = [[0, -1, "bf16"]]

[data]
num_workers = 4
prefetch_factor = 2
pin_memory = true
```
**Expected**: ~150-180K tok/s (RTX 4090, 256H, 4L, seq_len=256)

### Alternative: torch.compile Path
```toml
[training]
attention_backend = "xformers"
use_torch_compile = true
```
**When**: Flash Attention unavailable or validating compile benefits

## Profiling

Enable detailed profiling for model analysis:

```bash
python -m src.training.train \
    --config config/your_config.toml \
    --profile
```

Output: Model size, parameter count, memory usage per layer

## Performance Hierarchy

**Small model** (RTX 4090, 256H, 4L, seq_len=256):

1. **Flash Attention + Workers**: ~175k tokens/sec ⚡⚡⚡
2. **xFormers + compile + Workers**: ~145k tokens/sec ⚡⚡
3. **Standard + compile + Workers**: ~110k tokens/sec ⚡
4. **Baseline (no optimizations)**: ~80k tokens/sec ⚫

**Phase 3 production** (RTX 4090 + RTX 3090Ti, 1024H, 8-10L, seq_len=2048, BF16, Flash):

- **Per GPU**: ~58K tokens/sec
- **2-GPU DDP total**: ~116K tokens/sec effective (both GPUs combined)

*Throughput decreases with larger models and longer sequences; FSDP reduces this further but enables larger-than-VRAM models.*

## Common Issues

### Training hangs after initialization
**Cause**: `num_workers = 0` blocks on CPU
**Fix**: Set `num_workers = 4`

### "Flash Attention requested but not available"
**Cause**: Not installed
**Fix**: `pip install flash-attn --no-build-isolation`
**Note**: Auto-falls back to standard (training continues)

### torch.compile error with flash
**Cause**: Known incompatibility
**Fix**: Set `use_torch_compile = false` or switch to `attention_backend = "xformers"`

### OOM with num_workers > 0
**Cause**: Each worker allocates memory
**Fix**: Reduce `num_workers`, `batch_size`, or increase `gradient_accumulation_steps`

### DDP: NCCL error
**Fix**: Ensure CUDA available: `python -c "import torch; print(torch.cuda.is_available())"`

## Future Optimizations

- [ ] Gradient checkpointing — trade compute for memory at 300M+ params (Phase 4)
- [ ] FP8 linear layers — RTX 4090 supports FP8 matmuls (Phase 4)
- [ ] Activation offloading (Phase 4)
- [ ] KV-cache for autoregressive inference (Phase 5)
- [ ] Automatic backend selection based on hardware

## References

- Flash Attention: https://arxiv.org/abs/2205.14135
- Sage Attention: https://github.com/thu-ml/SageAttention
- xFormers: https://github.com/facebookresearch/xformers
- torch.compile: https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html
- DDP: https://pytorch.org/tutorials/intermediate/ddp_tutorial.html
